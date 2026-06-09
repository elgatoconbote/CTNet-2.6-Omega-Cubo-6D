#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTNet 2.6 Omega Cubo 6D mass-contextual trainer.

This trainer replaces the first smoke-test objective "only pull out.z toward a
byte target" with a CTNet-style objective:

    the true continuation must be the most coherent reinscription of the
    contextual mass produced by CTNet.

It still uses the same zero-disk online input path:
- no Hugging Face datasets / hub,
- no pyarrow,
- no Parquet/Xet cache,
- no corpus materialization,
- final CTNet deformation saved to disk by default.

The target text is not treated as a Transformer decoder target. It is treated as
the positive reinscription in the same continuous Z chart already used by CTNet.
Wrong / corrupted reinscriptions are negatives.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from ctnet_omega_cubo6d_plegado_ctnet26 import (
    FoldLayout,
    FoldedCTNetOmegaCubo26,
    FoldedOmegaCuboState,
    count_params,
)
from train_vram_strict_ctnet import (
    DEFAULT_URLS,
    OnlineSample,
    _text_tensor,
    batch_to_state,
    online_blocks,
    slot_variance,
)


def _roll_like(x: torch.Tensor, frac: float) -> torch.Tensor:
    flat = x.reshape(x.shape[0], -1)
    shift = max(1, int(flat.shape[-1] * frac))
    return torch.roll(flat, shifts=shift, dims=-1).reshape_as(x)


def make_negative_z(prompt_z: torch.Tensor, target_z: torch.Tensor) -> List[torch.Tensor]:
    """Negative reinscriptions in the same Z chart.

    These are not external memory. They are deliberately wrong chart candidates:
    shifted, inverted, prompt-copy, and batch-mismatched targets.
    """
    negs: List[torch.Tensor] = [
        _roll_like(target_z, 0.137),
        _roll_like(target_z, 0.381),
        -target_z,
        prompt_z.detach(),
    ]
    if target_z.shape[0] > 1:
        negs.append(torch.roll(target_z, shifts=1, dims=0))
    return negs


def candidate_energy(
    model: FoldedCTNetOmegaCubo26,
    out: FoldedOmegaCuboState,
    candidate_z: torch.Tensor,
    *,
    lambda_fit: float,
    lambda_cand_coh: float,
    lambda_cand_omega: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Energy of a proposed textual reinscription for the current CTNet mass.

    Lower is better. The candidate is evaluated as a possible visible Z-chart
    reading of the internal CTNet state. The model is not asked to decode tokens;
    it is asked whether this candidate belongs to the current closed regime.
    """
    fit = F.mse_loss(out.z, candidate_z)

    obs = model.cubo(candidate_z, out.memory, out.relations)
    cand_state = FoldedOmegaCuboState(
        z=candidate_z,
        memory=out.memory,
        relations=out.relations,
        cubo=obs["vector"].to(device=out.z.device, dtype=out.z.dtype),
        pad=out.pad,
    )
    cand_xi = model.pack(cand_state)
    cand_coh, cand_speed, cand_info = model.core.coherence_energy(cand_xi)
    cand_omega = obs["omega"].mean()

    energy = lambda_fit * fit + lambda_cand_coh * cand_coh + lambda_cand_omega * cand_omega
    metrics = {
        "fit": float(fit.detach().cpu()),
        "cand_coh": float(cand_coh.detach().cpu()),
        "cand_speed": float(cand_speed.detach().cpu()),
        "cand_info": float(cand_info.detach().cpu()),
        "cand_omega": float(cand_omega.detach().cpu()),
        "cand_residual": float(obs["residual"].mean().detach().cpu()),
        "cand_absorption": float(obs["absorption"].mean().detach().cpu()),
        "cand_closure": float(obs["closure_score"].mean().detach().cpu()),
    }
    return energy, metrics


def contextual_mass_loss(
    model: FoldedCTNetOmegaCubo26,
    state: FoldedOmegaCuboState,
    out: FoldedOmegaCuboState,
    target_z: torch.Tensor,
    *,
    tau: float,
    lambda_fit: float,
    lambda_cand_coh: float,
    lambda_cand_omega: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Contrastive closure loss: true continuation must be lowest energy."""
    energies: List[torch.Tensor] = []
    metric_rows: List[Dict[str, float]] = []

    true_e, true_m = candidate_energy(
        model,
        out,
        target_z,
        lambda_fit=lambda_fit,
        lambda_cand_coh=lambda_cand_coh,
        lambda_cand_omega=lambda_cand_omega,
    )
    energies.append(true_e)
    metric_rows.append(true_m)

    for neg_z in make_negative_z(state.z, target_z):
        e, m = candidate_energy(
            model,
            out,
            neg_z,
            lambda_fit=lambda_fit,
            lambda_cand_coh=lambda_cand_coh,
            lambda_cand_omega=lambda_cand_omega,
        )
        energies.append(e)
        metric_rows.append(m)

    e_stack = torch.stack(energies)  # [1 + negatives]
    logits = (-e_stack / max(tau, 1e-6)).unsqueeze(0)
    label = torch.zeros(1, dtype=torch.long, device=out.z.device)
    loss = F.cross_entropy(logits, label)

    with torch.no_grad():
        neg_min = e_stack[1:].min()
        margin = neg_min - e_stack[0]
        correct = (e_stack[0] < neg_min).to(torch.float32)

    metrics = {
        "mass_loss": float(loss.detach().cpu()),
        "true_energy": float(e_stack[0].detach().cpu()),
        "neg_min_energy": float(neg_min.detach().cpu()),
        "energy_margin": float(margin.detach().cpu()),
        "mass_correct": float(correct.detach().cpu()),
        "true_fit": metric_rows[0]["fit"],
        "true_cand_coh": metric_rows[0]["cand_coh"],
        "true_cand_omega": metric_rows[0]["cand_omega"],
        "true_cand_closure": metric_rows[0]["cand_closure"],
    }
    return loss, metrics


def train(args: argparse.Namespace) -> Dict:
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if args.fp64 else torch.float32

    layout = FoldLayout(
        N=args.N,
        d=args.d,
        z_tokens=args.z_tokens,
        z_dim=args.z_dim,
        mem_slots=args.mem_slots,
        mem_dim=args.mem_dim,
        rel_edges=args.rel_edges,
        rel_dim=args.rel_dim,
    )
    layout.validate()

    model = FoldedCTNetOmegaCubo26(
        layout=layout,
        fractal_steps=args.fractal_steps,
        latent_steps=args.latent_steps,
        cubo_shear=args.cubo_shear,
    ).to(device=device, dtype=dtype)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    stream = online_blocks(args.url, block_bytes=args.block_bytes, timeout=args.timeout)

    print("=== CTNet mass-contextual zero-disk training ===", flush=True)
    print("objective=true continuation is lowest-energy coherent reinscription", flush=True)
    print("loader=no datasets/no huggingface_hub/no pyarrow/no xet", flush=True)
    print(f"device={device} dtype={dtype} params={count_params(model)}", flush=True)
    print(f"layout capacity={layout.capacity} semantic_size={layout.semantic_size} pad_size={layout.pad_size}", flush=True)
    print(f"fixed memory M=[B,{layout.mem_slots},{layout.mem_dim}] relations R=[B,{layout.rel_edges},{layout.rel_dim}]", flush=True)
    print(f"urls={args.url or DEFAULT_URLS}", flush=True)
    print(f"save_final={args.save_final}", flush=True)

    t0 = time.time()
    last: Dict = {}

    for step in range(1, args.steps + 1):
        samples: List[OnlineSample] = [next(stream) for _ in range(args.batch)]
        state, target_z, regimes = batch_to_state(model, samples, device=device, dtype=dtype, max_bytes=args.max_bytes)

        optimizer.zero_grad(set_to_none=True)
        out = model.forward_state(state)
        xi_out = model.pack(out)

        loss_mass, mass_metrics = contextual_mass_loss(
            model,
            state,
            out,
            target_z,
            tau=args.mass_tau,
            lambda_fit=args.lambda_candidate_fit,
            lambda_cand_coh=args.lambda_candidate_coh,
            lambda_cand_omega=args.lambda_candidate_omega,
        )

        # Anchor is no longer the whole philosophy. It is a weak chart stabilizer.
        loss_anchor = F.mse_loss(out.z, target_z)
        loss_coh, speed, info = model.core.coherence_energy(xi_out)
        obs = model.cubo_observation(out)
        loss_omega = obs["omega"].mean()
        loss_cubo_track = F.mse_loss(out.cubo, obs["vector"].detach())

        mem_var = slot_variance(out.memory)
        rel_var = slot_variance(out.relations)
        loss_structure = F.relu(args.min_slot_var - mem_var) + F.relu(args.min_slot_var - rel_var)

        if args.reversibility_loss_every > 0 and step % args.reversibility_loss_every == 0:
            recovered = model.inverse_state(out)
            loss_rev = F.mse_loss(model.pack(recovered), model.pack(state))
        else:
            loss_rev = torch.zeros((), device=device, dtype=dtype)

        loss = (
            args.lambda_mass * loss_mass
            + args.lambda_anchor * loss_anchor
            + args.lambda_coh * loss_coh
            + args.lambda_omega * loss_omega
            + args.lambda_cubo * loss_cubo_track
            + args.lambda_structure * loss_structure
            + args.lambda_rev * loss_rev
        )
        loss.backward()

        if args.coherence_grad_scale:
            with torch.no_grad():
                scale = float(torch.clamp(speed.detach().to(torch.float32), args.grad_scale_min, args.grad_scale_max).cpu())
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        if args.empty_cache_every > 0 and step % args.empty_cache_every == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            last = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "loss_mass": float(loss_mass.detach().cpu()),
                "loss_anchor": float(loss_anchor.detach().cpu()),
                "loss_coh": float(loss_coh.detach().cpu()),
                "loss_omega": float(loss_omega.detach().cpu()),
                "loss_rev": float(loss_rev.detach().cpu()),
                "speed": float(speed.detach().cpu()),
                "info": float(info.detach().cpu()),
                "mem_slot_var": float(mem_var.detach().cpu()),
                "rel_slot_var": float(rel_var.detach().cpu()),
                "omega": float(obs["omega"].mean().detach().cpu()),
                "residual": float(obs["residual"].mean().detach().cpu()),
                "absorption": float(obs["absorption"].mean().detach().cpu()),
                "closure_score": float(obs["closure_score"].mean().detach().cpu()),
                "elapsed_sec": elapsed,
                "source": samples[0].source,
                "regimes": regimes,
                **mass_metrics,
            }
            print(
                f"step {step:6d} | loss={last['loss']:.6e} mass={last['loss_mass']:.6e} "
                f"anchor={last['loss_anchor']:.3e} trueE={last['true_energy']:.3e} "
                f"negMin={last['neg_min_energy']:.3e} margin={last['energy_margin']:.3e} "
                f"ok={last['mass_correct']:.0f} omega={last['omega']:.2e} "
                f"coh={last['loss_coh']:.3e} rev={last['loss_rev']:.1e} time={elapsed:.1f}s",
                flush=True,
            )

    if args.save_final:
        path = Path(args.save_final)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "last": last, "args": vars(args)}, path)
        print(f"saved_final={path}", flush=True)

    return last


def main() -> None:
    p = argparse.ArgumentParser(description="Mass-contextual CTNet trainer: true response is coherent reinscription.")
    p.add_argument("--url", action="append", default=[], help="Direct text URL to stream. Can be passed multiple times.")
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--block-bytes", type=int, default=2048)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--fp64", action="store_true")
    p.add_argument("--max-bytes", type=int, default=2048)

    p.add_argument("--N", type=int, default=64)
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--z-tokens", type=int, default=32)
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--mem-slots", type=int, default=8)
    p.add_argument("--mem-dim", type=int, default=16)
    p.add_argument("--rel-edges", type=int, default=8)
    p.add_argument("--rel-dim", type=int, default=16)
    p.add_argument("--fractal-steps", type=int, default=4)
    p.add_argument("--latent-steps", type=int, default=2)
    p.add_argument("--cubo-shear", type=float, default=0.05)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--coherence-grad-scale", action="store_true")
    p.add_argument("--grad-scale-min", type=float, default=0.5)
    p.add_argument("--grad-scale-max", type=float, default=5.0)

    p.add_argument("--mass-tau", type=float, default=0.15)
    p.add_argument("--lambda-mass", type=float, default=1.0)
    p.add_argument("--lambda-anchor", type=float, default=0.05)
    p.add_argument("--lambda-candidate-fit", type=float, default=1.0)
    p.add_argument("--lambda-candidate-coh", type=float, default=0.02)
    p.add_argument("--lambda-candidate-omega", type=float, default=0.25)

    p.add_argument("--lambda-coh", type=float, default=0.02)
    p.add_argument("--lambda-omega", type=float, default=0.25)
    p.add_argument("--lambda-cubo", type=float, default=0.05)
    p.add_argument("--lambda-structure", type=float, default=0.10)
    p.add_argument("--lambda-rev", type=float, default=0.10)
    p.add_argument("--reversibility-loss-every", type=int, default=10)
    p.add_argument("--min-slot-var", type=float, default=1e-8)

    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--empty-cache-every", type=int, default=25)
    p.add_argument("--save-final", default="checkpoints/ctnet_mass_state_final.pt")

    args = p.parse_args()
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()

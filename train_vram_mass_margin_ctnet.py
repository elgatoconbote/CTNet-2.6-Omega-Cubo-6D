#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTNet 2.6 Omega Cubo 6D mass-margin trainer.

This is a stricter replacement for train_vram_mass_ctnet.py.

The previous mass loss used softmax energy ranking. It reduced global coherence,
but the true continuation often did not beat the hard negative. This trainer adds
an explicit margin constraint:

    true_energy + margin < every_negative_energy

So the model is not merely asked to lower energy. It is asked to make the true
reinscription belong more strongly than false reinscriptions.

Still zero-disk corpus mode:
- no Hugging Face datasets / hub,
- no pyarrow,
- no Parquet/Xet cache,
- final CTNet deformation saved to disk by default.
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
    batch_to_state,
    online_blocks,
    slot_variance,
)


def flatten_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.reshape(a.shape[0], -1) - b.reshape(b.shape[0], -1)).pow(2).mean(dim=-1)


def roll_like(x: torch.Tensor, frac: float) -> torch.Tensor:
    flat = x.reshape(x.shape[0], -1)
    shift = max(1, int(flat.shape[-1] * frac))
    return torch.roll(flat, shifts=shift, dims=-1).reshape_as(x)


def make_internal_negatives(prompt_z: torch.Tensor, target_z: torch.Tensor) -> List[torch.Tensor]:
    negs = [
        prompt_z.detach(),
        roll_like(target_z, 0.137).detach(),
        roll_like(target_z, 0.381).detach(),
        (-target_z).detach(),
    ]
    if target_z.shape[0] > 1:
        negs.append(torch.roll(target_z, shifts=1, dims=0).detach())
    return negs


def candidate_energy_vector(
    model: FoldedCTNetOmegaCubo26,
    out: FoldedOmegaCuboState,
    candidate_z: torch.Tensor,
    *,
    lambda_fit: float,
    lambda_omega: float,
    lambda_residual: float,
) -> torch.Tensor:
    fit = flatten_mse(out.z, candidate_z)
    obs = model.cubo(candidate_z, out.memory, out.relations)
    omega = obs["omega"]
    residual = obs["residual"]
    absorption = obs["absorption"]
    excess_residual = torch.relu(residual - absorption)
    return lambda_fit * fit + lambda_omega * omega + lambda_residual * excess_residual


def mass_margin_loss(
    model: FoldedCTNetOmegaCubo26,
    state: FoldedOmegaCuboState,
    out: FoldedOmegaCuboState,
    target_z: torch.Tensor,
    negative_z_list: List[torch.Tensor],
    *,
    margin: float,
    tau: float,
    lambda_fit: float,
    lambda_omega: float,
    lambda_residual: float,
    lambda_softmax: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    true_e = candidate_energy_vector(
        model,
        out,
        target_z,
        lambda_fit=lambda_fit,
        lambda_omega=lambda_omega,
        lambda_residual=lambda_residual,
    )

    neg_es = []
    for neg_z in negative_z_list:
        neg_es.append(
            candidate_energy_vector(
                model,
                out,
                neg_z,
                lambda_fit=lambda_fit,
                lambda_omega=lambda_omega,
                lambda_residual=lambda_residual,
            )
        )
    neg_stack = torch.stack(neg_es, dim=0)  # [K,B]

    # Hard margin: all negatives must be above the true candidate by margin.
    # Shape [K,B].
    hinge = torch.relu(true_e.unsqueeze(0) - neg_stack + margin).mean()

    # Soft ranking is auxiliary only; margin is the real belonging constraint.
    e_all = torch.cat([true_e.unsqueeze(0), neg_stack], dim=0).transpose(0, 1)  # [B,1+K]
    logits = -e_all / max(tau, 1e-6)
    labels = torch.zeros(e_all.shape[0], dtype=torch.long, device=e_all.device)
    soft = F.cross_entropy(logits, labels)

    loss = hinge + lambda_softmax * soft

    with torch.no_grad():
        neg_min = neg_stack.min(dim=0).values
        raw_margin = neg_min - true_e
        ok = (raw_margin > 0).to(torch.float32).mean()
        ok_margin = (raw_margin > margin).to(torch.float32).mean()

    return loss, {
        "mass_loss": float(loss.detach().cpu()),
        "hinge_loss": float(hinge.detach().cpu()),
        "soft_loss": float(soft.detach().cpu()),
        "true_energy": float(true_e.mean().detach().cpu()),
        "neg_min_energy": float(neg_min.mean().detach().cpu()),
        "energy_margin": float(raw_margin.mean().detach().cpu()),
        "ok": float(ok.detach().cpu()),
        "ok_margin": float(ok_margin.detach().cpu()),
    }


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

    print("=== CTNet mass-margin zero-disk training ===", flush=True)
    print("objective=true_energy + margin < every_negative_energy", flush=True)
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

        negative_z_list = make_internal_negatives(state.z, target_z)

        # Extra online negatives: other continuations from the stream, encoded in the same chart.
        if args.extra_negatives > 0:
            extra_samples = [next(stream) for _ in range(args.extra_negatives)]
            _, extra_target_z, _ = batch_to_state(model, extra_samples, device=device, dtype=dtype, max_bytes=args.max_bytes)
            for j in range(extra_target_z.shape[0]):
                neg = extra_target_z[j : j + 1].repeat(args.batch, 1, 1).detach()
                negative_z_list.append(neg)

        optimizer.zero_grad(set_to_none=True)
        out = model.forward_state(state)
        xi_out = model.pack(out)

        loss_mass, mass_metrics = mass_margin_loss(
            model,
            state,
            out,
            target_z,
            negative_z_list,
            margin=args.mass_margin,
            tau=args.mass_tau,
            lambda_fit=args.lambda_candidate_fit,
            lambda_omega=args.lambda_candidate_omega,
            lambda_residual=args.lambda_candidate_residual,
            lambda_softmax=args.lambda_softmax,
        )

        # Weak chart attachment. This is not the full philosophy; it prevents the
        # mass objective from drifting into a chart that cannot be read later.
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
                f"step {step:6d} | loss={last['loss']:.6e} mass={last['loss_mass']:.3e} "
                f"hinge={last['hinge_loss']:.3e} anchor={last['loss_anchor']:.3e} "
                f"trueE={last['true_energy']:.3e} negMin={last['neg_min_energy']:.3e} "
                f"margin={last['energy_margin']:.3e} ok={last['ok']:.0f} okM={last['ok_margin']:.0f} "
                f"omega={last['omega']:.1e} coh={last['loss_coh']:.3e} rev={last['loss_rev']:.1e} time={elapsed:.1f}s",
                flush=True,
            )

    if args.save_final:
        path = Path(args.save_final)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "last": last, "args": vars(args)}, path)
        print(f"saved_final={path}", flush=True)

    return last


def main() -> None:
    p = argparse.ArgumentParser(description="Mass-margin CTNet trainer: true response must beat negatives by margin.")
    p.add_argument("--url", action="append", default=[], help="Direct text URL to stream. Can be passed multiple times.")
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--block-bytes", type=int, default=2048)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--extra-negatives", type=int, default=2)
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

    p.add_argument("--mass-margin", type=float, default=0.05)
    p.add_argument("--mass-tau", type=float, default=0.15)
    p.add_argument("--lambda-mass", type=float, default=1.0)
    p.add_argument("--lambda-anchor", type=float, default=0.25)
    p.add_argument("--lambda-softmax", type=float, default=0.10)
    p.add_argument("--lambda-candidate-fit", type=float, default=1.0)
    p.add_argument("--lambda-candidate-omega", type=float, default=0.25)
    p.add_argument("--lambda-candidate-residual", type=float, default=0.25)

    p.add_argument("--lambda-coh", type=float, default=0.02)
    p.add_argument("--lambda-omega", type=float, default=0.25)
    p.add_argument("--lambda-cubo", type=float, default=0.05)
    p.add_argument("--lambda-structure", type=float, default=0.10)
    p.add_argument("--lambda-rev", type=float, default=0.10)
    p.add_argument("--reversibility-loss-every", type=int, default=10)
    p.add_argument("--min-slot-var", type=float, default=1e-8)

    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--empty-cache-every", type=int, default=25)
    p.add_argument("--save-final", default="checkpoints/ctnet_mass_margin_state_final.pt")

    args = p.parse_args()
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train CTNet 2.6 Omega + Cubo 6D from online streaming datasets.

Strict data rule
----------------
The training corpus is never downloaded as a local dataset. Samples are consumed
online through Hugging Face streaming (`load_dataset(..., streaming=True)`).

Local outputs are only:
  - checkpoints
  - JSONL metrics
  - optional tiny preview/report files

Architecture rule
-----------------
CTNet memory is fixed-size by construction. The layout defines:

    Z:        [B, z_tokens, z_dim]
    M:        [B, mem_slots, mem_dim]
    R:        [B, rel_edges, rel_dim]
    C6:       [B, 29]
    pad:      [B, pad_size]
    Xi:       [B, N, d]

No append, no KV-cache, no vector-store, no growing relation list.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
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
from ctnet_streaming_datasets import StreamMixConfig, make_online_regime_stream, preview as preview_stream


def _byte_signal(text: str, size: int, *, max_bytes: int = 8192) -> torch.Tensor:
    """Map text to a fixed real vector without downloading any tokenizer.

    This is deliberately byte-level and local: no tokenizer model, no vocabulary
    file, no external artifacts. It is a minimal next-token-like interface for
    early CTNet regime training.
    """
    raw = (text or "").encode("utf-8", errors="ignore")[:max_bytes]
    if not raw:
        raw = b"<empty>"
    v = torch.zeros(size, dtype=torch.float32)
    for i, b in enumerate(raw):
        j = i % size
        depth = 1.0 + (i // size)
        v[j] += ((float(b) / 127.5) - 1.0) / math.sqrt(depth)
    phase = torch.linspace(0, 2.0 * math.pi, size, dtype=torch.float32)
    v = torch.tanh(v + 0.015 * torch.sin(phase) + 0.0075 * torch.cos(2.0 * phase))
    return v


def _text_tensor(text: str, shape: Tuple[int, ...], *, amp: float = 1.0, max_bytes: int = 8192) -> torch.Tensor:
    n = 1
    for s in shape:
        n *= int(s)
    return (amp * _byte_signal(text, n, max_bytes=max_bytes)).reshape(*shape)


def _pad_anchor(batch: int, pad_size: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if pad_size <= 0:
        return torch.zeros(batch, 0, dtype=dtype, device=device)
    phase = torch.linspace(0, 2.0 * math.pi, pad_size, dtype=dtype, device=device)
    pad = 0.01 * (torch.sin(phase) + 0.5 * torch.cos(2.0 * phase))
    return pad.unsqueeze(0).repeat(batch, 1)


def batch_to_state(
    model: FoldedCTNetOmegaCubo26,
    samples: List[Dict],
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_bytes: int,
) -> Tuple[FoldedOmegaCuboState, torch.Tensor, List[str]]:
    L = model.layout
    batch = len(samples)

    z_rows = []
    mem_rows = []
    rel_rows = []
    target_z_rows = []
    regimes = []

    for ex in samples:
        regime = str(ex.get("regime", "unknown"))
        source = str(ex.get("source", "unknown"))
        x = str(ex.get("x", ""))
        y = str(ex.get("y", ""))

        z_rows.append(_text_tensor(f"<regime>{regime}</regime>\n{x}", (L.z_tokens, L.z_dim), amp=1.0, max_bytes=max_bytes))
        mem_rows.append(_text_tensor(f"<source>{source}</source>\n<regime>{regime}</regime>\n{x}", (L.mem_slots, L.mem_dim), amp=0.01, max_bytes=max_bytes))
        rel_rows.append(_text_tensor(f"<relations>{regime}|{source}</relations>\n{x[:2048]}", (L.rel_edges, L.rel_dim), amp=0.01, max_bytes=max_bytes))
        target_z_rows.append(_text_tensor(y, (L.z_tokens, L.z_dim), amp=1.0, max_bytes=max_bytes))
        regimes.append(regime)

    z = torch.stack(z_rows, dim=0).to(device=device, dtype=dtype)
    memory = torch.stack(mem_rows, dim=0).to(device=device, dtype=dtype)
    relations = torch.stack(rel_rows, dim=0).to(device=device, dtype=dtype)
    target_z = torch.stack(target_z_rows, dim=0).to(device=device, dtype=dtype)
    pad = _pad_anchor(batch, L.pad_size, dtype=dtype, device=device)

    with torch.no_grad():
        cubo0 = model.cubo(z, memory, relations)["vector"].to(device=device, dtype=dtype)

    return FoldedOmegaCuboState(z=z, memory=memory, relations=relations, cubo=cubo0, pad=pad), target_z, regimes


def slot_variance(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-2] <= 1:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return x.var(dim=-2, unbiased=False).mean()


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

    stream_config = StreamMixConfig(
        fineweb_prob=args.p_fineweb,
        openwebmath_prob=args.p_openwebmath,
        numina_prob=args.p_numina,
        swe_prob=args.p_swe,
        seed=args.seed,
        fineweb_name=args.fineweb_name,
        use_fineweb=not args.no_fineweb,
        use_openwebmath=not args.no_openwebmath,
        use_numina=not args.no_numina,
        use_swe=not args.no_swe,
    )
    stream = make_online_regime_stream(stream_config)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "layout": {
                    "N": layout.N,
                    "d": layout.d,
                    "z_tokens": layout.z_tokens,
                    "z_dim": layout.z_dim,
                    "mem_slots": layout.mem_slots,
                    "mem_dim": layout.mem_dim,
                    "rel_edges": layout.rel_edges,
                    "rel_dim": layout.rel_dim,
                    "capacity": layout.capacity,
                    "semantic_size": layout.semantic_size,
                    "pad_size": layout.pad_size,
                },
                "stream": asdict(stream_config),
                "args": vars(args),
                "fixed_memory": True,
                "no_corpus_download": True,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=== CTNet 2.6 Omega Cubo 6D online-stream training ===")
    print("repo mode: online streaming only; no corpus materialization")
    print(f"device={device} dtype={dtype} params={count_params(model)}")
    print(f"layout capacity={layout.capacity} semantic_size={layout.semantic_size} pad_size={layout.pad_size}")
    print(f"fixed memory M=[B,{layout.mem_slots},{layout.mem_dim}] relations R=[B,{layout.rel_edges},{layout.rel_dim}]")
    print(f"metrics={metrics_path}")

    t0 = time.time()
    last = {}

    for step in range(1, args.steps + 1):
        samples = [next(stream) for _ in range(args.batch)]
        state, target_z, regimes = batch_to_state(model, samples, device=device, dtype=dtype, max_bytes=args.max_bytes)

        optimizer.zero_grad(set_to_none=True)

        out = model.forward_state(state)
        xi_out = model.pack(out)

        loss_task = F.mse_loss(out.z, target_z)
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
            args.lambda_task * loss_task
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

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                audit = model.audit(batch=min(args.batch, 2), dtype=dtype, device=device, steps=1, seed=args.seed + step)
            elapsed = time.time() - t0
            sources = {}
            for ex in samples:
                sources[ex.get("mixture_name", "unknown")] = sources.get(ex.get("mixture_name", "unknown"), 0) + 1

            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "loss_task": float(loss_task.detach().cpu()),
                "loss_coh": float(loss_coh.detach().cpu()),
                "loss_omega": float(loss_omega.detach().cpu()),
                "loss_cubo_track": float(loss_cubo_track.detach().cpu()),
                "loss_structure": float(loss_structure.detach().cpu()),
                "loss_rev": float(loss_rev.detach().cpu()),
                "speed": float(speed.detach().cpu()),
                "info": float(info.detach().cpu()),
                "mem_slot_var": float(mem_var.detach().cpu()),
                "rel_slot_var": float(rel_var.detach().cpu()),
                "memory_shape_ok": audit["memory_shape_ok"],
                "relations_shape_ok": audit["relations_shape_ok"],
                "packed_mae_audit": audit["packed_mae"],
                "packed_rel_audit": audit["packed_rel"],
                "sources": sources,
                "regimes": regimes,
                "elapsed_sec": elapsed,
            }
            last = row
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"step {step:6d} | loss={row['loss']:.6e} task={row['loss_task']:.6e} "
                f"omega={row['loss_omega']:.6e} coh={row['loss_coh']:.6e} "
                f"rev={row['loss_rev']:.2e} mem_shape={row['memory_shape_ok']:.0f} "
                f"rel_shape={row['relations_shape_ok']:.0f} audit_mae={row['packed_mae_audit']:.2e} "
                f"time={elapsed:.1f}s"
            )

        if args.save_every > 0 and step % args.save_every == 0:
            ckpt_path = ckpt_dir / f"ctnet_stream_step_{step:08d}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "layout": {
                        "N": layout.N,
                        "d": layout.d,
                        "z_tokens": layout.z_tokens,
                        "z_dim": layout.z_dim,
                        "mem_slots": layout.mem_slots,
                        "mem_dim": layout.mem_dim,
                        "rel_edges": layout.rel_edges,
                        "rel_dim": layout.rel_dim,
                    },
                    "stream_config": asdict(stream_config),
                    "args": vars(args),
                    "last_metrics": last,
                },
                ckpt_path,
            )
            print(f"saved {ckpt_path}")

    final_path = ckpt_dir / "ctnet_stream_final.pt"
    torch.save(
        {
            "step": args.steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "layout": {
                "N": layout.N,
                "d": layout.d,
                "z_tokens": layout.z_tokens,
                "z_dim": layout.z_dim,
                "mem_slots": layout.mem_slots,
                "mem_dim": layout.mem_dim,
                "rel_edges": layout.rel_edges,
                "rel_dim": layout.rel_dim,
            },
            "stream_config": asdict(stream_config),
            "args": vars(args),
            "last_metrics": last,
        },
        final_path,
    )
    print(f"final checkpoint: {final_path}")
    return {"checkpoint": str(final_path), "metrics": str(metrics_path), "last": last}


def main() -> None:
    p = argparse.ArgumentParser(description="Train CTNet 2.6 Omega Cubo 6D from fully online streaming datasets.")

    p.add_argument("--preview", action="store_true", help="Preview online stream samples and exit.")
    p.add_argument("--preview-n", type=int, default=5)

    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--out-dir", default="runs/online_stream")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--fp64", action="store_true")
    p.add_argument("--max-bytes", type=int, default=8192)

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

    p.add_argument("--lambda-task", type=float, default=1.0)
    p.add_argument("--lambda-coh", type=float, default=0.05)
    p.add_argument("--lambda-omega", type=float, default=0.25)
    p.add_argument("--lambda-cubo", type=float, default=0.05)
    p.add_argument("--lambda-structure", type=float, default=0.10)
    p.add_argument("--lambda-rev", type=float, default=0.10)
    p.add_argument("--reversibility-loss-every", type=int, default=10)
    p.add_argument("--min-slot-var", type=float, default=1e-8)

    p.add_argument("--p-fineweb", type=float, default=0.55)
    p.add_argument("--p-openwebmath", type=float, default=0.25)
    p.add_argument("--p-numina", type=float, default=0.15)
    p.add_argument("--p-swe", type=float, default=0.05)
    p.add_argument("--fineweb-name", default="sample-10BT")
    p.add_argument("--no-fineweb", action="store_true")
    p.add_argument("--no-openwebmath", action="store_true")
    p.add_argument("--no-numina", action="store_true")
    p.add_argument("--no-swe", action="store_true")

    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500)

    args = p.parse_args()

    if args.preview:
        print(json.dumps(preview_stream(n=args.preview_n, seed=args.seed), indent=2, ensure_ascii=False))
        return

    result = train(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

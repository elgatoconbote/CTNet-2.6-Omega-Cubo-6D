#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probe a trained CTNet 2.6 Omega Cubo 6D checkpoint.

This is intentionally not a Transformer-style tokenizer decoder. The current
CTNet trainer is byte-signal / structural-state training, so the safest probe is:

1. load trained CTNet state_dict,
2. encode a prompt into fixed CTNet state (Z,M,R,C6,pad),
3. run one or more reversible CTNet ticks,
4. report closure/coherence/reversibility,
5. optionally rank candidate replies by CTNet structural fit,
6. print an approximate byte projection of the output state for smoke testing.

For real conversational text, add a proper decoder head later. This probe is for
checking whether the trained CTNet state loads and how it scores/responds
structurally after training.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from ctnet_omega_cubo6d_plegado_ctnet26 import (
    FoldLayout,
    FoldedCTNetOmegaCubo26,
    FoldedOmegaCuboState,
)


PRINTABLE = "\n " + "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,;:!?-_'\"()[]{}<>/\\@#$%&*+=|"


def _byte_signal(text: str, size: int, *, max_bytes: int = 2048) -> torch.Tensor:
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


def _text_tensor(text: str, shape: Tuple[int, ...], *, amp: float = 1.0, max_bytes: int = 2048) -> torch.Tensor:
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


def build_state(
    model: FoldedCTNetOmegaCubo26,
    prompt: str,
    *,
    source: str,
    regime: str,
    device: torch.device,
    dtype: torch.dtype,
    max_bytes: int,
) -> FoldedOmegaCuboState:
    L = model.layout
    z = _text_tensor(f"<regime>{regime}</regime>\n{prompt}", (L.z_tokens, L.z_dim), amp=1.0, max_bytes=max_bytes)
    memory = _text_tensor(
        f"<source>{source}</source>\n<regime>{regime}</regime>\n{prompt}",
        (L.mem_slots, L.mem_dim),
        amp=0.01,
        max_bytes=max_bytes,
    )
    relations = _text_tensor(
        f"<relations>{regime}|{source}</relations>\n{prompt[:1024]}",
        (L.rel_edges, L.rel_dim),
        amp=0.01,
        max_bytes=max_bytes,
    )
    z = z.unsqueeze(0).to(device=device, dtype=dtype)
    memory = memory.unsqueeze(0).to(device=device, dtype=dtype)
    relations = relations.unsqueeze(0).to(device=device, dtype=dtype)
    pad = _pad_anchor(1, L.pad_size, dtype=dtype, device=device)
    with torch.no_grad():
        cubo = model.cubo(z, memory, relations)["vector"].to(device=device, dtype=dtype)
    return FoldedOmegaCuboState(z=z, memory=memory, relations=relations, cubo=cubo, pad=pad)


def approx_decode_z(z: torch.Tensor, *, chars: int = 256) -> str:
    """Very rough smoke-test projection from continuous z to printable chars.

    This is not a true language decoder. It maps output-state values to a fixed
    printable alphabet so the user can see whether the state changes after load.
    """
    flat = z.detach().to(torch.float32).flatten().cpu()
    if flat.numel() == 0:
        return ""
    # Normalize robustly, then map to printable alphabet.
    flat = flat - flat.median()
    scale = flat.abs().quantile(0.95).clamp_min(1e-6)
    flat = torch.tanh(flat / scale)
    idx = (((flat + 1.0) * 0.5) * (len(PRINTABLE) - 1)).round().long().clamp(0, len(PRINTABLE) - 1)
    text = "".join(PRINTABLE[i] for i in idx[:chars].tolist())
    return text


def make_model_from_args(saved_args: Dict, *, device: torch.device, dtype: torch.dtype) -> FoldedCTNetOmegaCubo26:
    layout = FoldLayout(
        N=int(saved_args.get("N", 64)),
        d=int(saved_args.get("d", 16)),
        z_tokens=int(saved_args.get("z_tokens", saved_args.get("z-tokens", 32))),
        z_dim=int(saved_args.get("z_dim", saved_args.get("z-dim", 16))),
        mem_slots=int(saved_args.get("mem_slots", saved_args.get("mem-slots", 8))),
        mem_dim=int(saved_args.get("mem_dim", saved_args.get("mem-dim", 16))),
        rel_edges=int(saved_args.get("rel_edges", saved_args.get("rel-edges", 8))),
        rel_dim=int(saved_args.get("rel_dim", saved_args.get("rel-dim", 16))),
    )
    layout.validate()
    return FoldedCTNetOmegaCubo26(
        layout=layout,
        fractal_steps=int(saved_args.get("fractal_steps", saved_args.get("fractal-steps", 4))),
        latent_steps=int(saved_args.get("latent_steps", saved_args.get("latent-steps", 2))),
        cubo_shear=float(saved_args.get("cubo_shear", saved_args.get("cubo-shear", 0.05))),
    ).to(device=device, dtype=dtype)


def score_candidate(
    model: FoldedCTNetOmegaCubo26,
    out: FoldedOmegaCuboState,
    candidate: str,
    *,
    prompt: str,
    max_bytes: int,
) -> Dict:
    L = model.layout
    target = _text_tensor(candidate, (L.z_tokens, L.z_dim), amp=1.0, max_bytes=max_bytes).unsqueeze(0).to(
        device=out.z.device, dtype=out.z.dtype
    )
    task = F.mse_loss(out.z, target)
    obs = model.cubo_observation(out)
    xi = model.pack(out)
    coh, speed, info = model.core.coherence_energy(xi)
    # Lower is better: fit to candidate plus residual/coherence debt.
    score = task + 0.05 * coh + 0.25 * obs["omega"].mean()
    return {
        "candidate": candidate,
        "score": float(score.detach().cpu()),
        "task_mse": float(task.detach().cpu()),
        "coherence_energy": float(coh.detach().cpu()),
        "speed": float(speed.detach().cpu()),
        "info": float(info.detach().cpu()),
        "omega": float(obs["omega"].mean().detach().cpu()),
        "residual": float(obs["residual"].mean().detach().cpu()),
        "absorption": float(obs["absorption"].mean().detach().cpu()),
        "closure_score": float(obs["closure_score"].mean().detach().cpu()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Probe a trained CTNet checkpoint after strict zero-disk training.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--candidate", action="append", default=[], help="Candidate reply to rank. Can be passed multiple times.")
    p.add_argument("--ticks", type=int, default=1)
    p.add_argument("--chars", type=int, default=256)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--fp64", action="store_true")
    p.add_argument("--max-bytes", type=int, default=2048)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if args.fp64 else torch.float32

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    saved_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    model = make_model_from_args(saved_args, device=device, dtype=dtype)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    state = build_state(
        model,
        args.prompt,
        source="probe_prompt",
        regime="post_training_probe",
        device=device,
        dtype=dtype,
        max_bytes=args.max_bytes,
    )
    x0 = model.pack(state)

    with torch.no_grad():
        out = state
        for _ in range(max(1, args.ticks)):
            out = model.forward_state(out)
        xi = model.pack(out)
        recovered = model.inverse_state(out)
        rev_mae = (model.pack(recovered) - x0).abs().mean()
        obs = model.cubo_observation(out)
        coh, speed, info = model.core.coherence_energy(xi)
        approx = approx_decode_z(out.z, chars=args.chars)
        ranked = [score_candidate(model, out, c, prompt=args.prompt, max_bytes=args.max_bytes) for c in args.candidate]
        ranked.sort(key=lambda r: r["score"])

    report = {
        "checkpoint": str(Path(args.checkpoint)),
        "device": str(device),
        "prompt": args.prompt,
        "ticks": args.ticks,
        "approx_response_projection": approx,
        "metrics": {
            "reversibility_mae": float(rev_mae.detach().cpu()),
            "coherence_energy": float(coh.detach().cpu()),
            "speed": float(speed.detach().cpu()),
            "info": float(info.detach().cpu()),
            "omega": float(obs["omega"].mean().detach().cpu()),
            "residual": float(obs["residual"].mean().detach().cpu()),
            "absorption": float(obs["absorption"].mean().detach().cpu()),
            "closure_score": float(obs["closure_score"].mean().detach().cpu()),
        },
        "ranked_candidates": ranked,
        "note": "approx_response_projection is a structural byte projection, not a true language decoder.",
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("=== CTNet post-training probe ===")
        print(f"checkpoint: {report['checkpoint']}")
        print(f"device: {report['device']}")
        print(f"prompt: {args.prompt}")
        print("metrics:")
        for k, v in report["metrics"].items():
            print(f"  {k}: {v:.8e}")
        print("approx_response_projection:")
        print(approx)
        if ranked:
            print("ranked_candidates lower_is_better:")
            for i, r in enumerate(ranked, 1):
                print(
                    f"  {i}. score={r['score']:.8e} task={r['task_mse']:.8e} "
                    f"omega={r['omega']:.8e} closure={r['closure_score']:.8e} :: {r['candidate']}"
                )
        print("note: approximate projection is not a true decoder; use candidate ranking for structural response testing.")


if __name__ == "__main__":
    main()

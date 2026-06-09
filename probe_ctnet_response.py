#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definitive CTNet response probe.

This file does not pretend that CTNet is an autoregressive language model.
It probes conversation in the way CTNet is currently trained:

    prompt -> contextual mass -> candidate response -> closure score

A response is not accepted because it is the next token sequence emitted by a
decoder. It is accepted if, when reinscribed into the CTNet state together with
the prompt, it preserves closure:

    u = p

across the exposed scales and perspectives, while keeping coherence, omega,
reversibility and Cubo 6D closure healthy.

The probe therefore ranks candidate replies by CTNet compatibility.
Generation of free text belongs to a later linguistic observer/readout, not to
this structural probe.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

from ctnet_omega_cubo6d_plegado_ctnet26 import (
    FoldLayout,
    FoldedCTNetOmegaCubo26,
    FoldedOmegaCuboState,
)


@dataclass
class CandidateScore:
    candidate: str
    score: float
    up_total: float
    up_z: float
    up_memory: float
    up_relations: float
    up_cubo: float
    up_xi: float
    up_delta: float
    coherence_energy: float
    omega: float
    residual: float
    absorption: float
    closure_score: float
    rev_mae: float
    anchor_mse: float


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


def _even_last_dim(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2 == 0:
        return x
    return F.pad(x, (0, 1))


def _up_mse_last_dim(x: torch.Tensor) -> torch.Tensor:
    x = _even_last_dim(x)
    d2 = x.shape[-1] // 2
    u = x[..., :d2]
    p = x[..., d2:]
    return F.mse_loss(u, p)


def _pool_tokens(x: torch.Tensor, scale: int) -> torch.Tensor:
    if x.ndim != 3 or x.shape[1] < scale:
        return x
    b, n, d = x.shape
    usable = (n // scale) * scale
    if usable <= 0:
        return x
    return x[:, :usable, :].reshape(b, usable // scale, scale, d).mean(dim=2)


def multiscale_up_loss(x: torch.Tensor, *, token_scales: Tuple[int, ...] = (2, 4, 8)) -> torch.Tensor:
    terms: List[torch.Tensor] = [_up_mse_last_dim(x)]

    for shift in (1, 2, 3):
        if x.shape[-1] > shift:
            terms.append(_up_mse_last_dim(torch.roll(x, shifts=shift, dims=-1)))

    if x.ndim == 3:
        for shift in (1, 2, 4):
            if x.shape[1] > shift:
                terms.append(_up_mse_last_dim(torch.roll(x, shifts=shift, dims=1)))
        for scale in token_scales:
            if x.shape[1] >= scale:
                pooled = _pool_tokens(x, scale)
                terms.append(_up_mse_last_dim(pooled))
                if pooled.shape[1] > 1:
                    terms.append(_up_mse_last_dim(torch.roll(pooled, shifts=1, dims=1)))

    return torch.stack(terms).mean()


def all_perspective_up_loss(
    model: FoldedCTNetOmegaCubo26,
    state: FoldedOmegaCuboState,
    out: FoldedOmegaCuboState,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    xi_in = model.pack(state)
    xi_out = model.pack(out)
    delta = xi_out - xi_in

    z_up = multiscale_up_loss(out.z)
    mem_up = multiscale_up_loss(out.memory)
    rel_up = multiscale_up_loss(out.relations)
    cubo_up = multiscale_up_loss(out.cubo)
    xi_up = multiscale_up_loss(xi_out)
    delta_up = multiscale_up_loss(delta)
    total = torch.stack([z_up, mem_up, rel_up, cubo_up, xi_up, delta_up]).mean()

    return total, {
        "up_total": float(total.detach().cpu()),
        "up_z": float(z_up.detach().cpu()),
        "up_memory": float(mem_up.detach().cpu()),
        "up_relations": float(rel_up.detach().cpu()),
        "up_cubo": float(cubo_up.detach().cpu()),
        "up_xi": float(xi_up.detach().cpu()),
        "up_delta": float(delta_up.detach().cpu()),
    }


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


def build_dialogue_state(
    model: FoldedCTNetOmegaCubo26,
    prompt: str,
    candidate: str,
    *,
    source: str,
    regime: str,
    device: torch.device,
    dtype: torch.dtype,
    max_bytes: int,
) -> Tuple[FoldedOmegaCuboState, torch.Tensor]:
    """Build a CTNet state for prompt plus candidate response.

    The candidate is not decoded from CTNet. It is treated as a possible visible
    reinscription and inserted into the CTNet chart so the system can judge its
    closure compatibility.
    """
    L = model.layout
    prompt_block = f"<regime>{regime}</regime>\n<prompt>\n{prompt}\n</prompt>"
    candidate_block = f"<candidate_response>\n{candidate}\n</candidate_response>"
    dialogue_block = f"{prompt_block}\n{candidate_block}"

    z = _text_tensor(dialogue_block, (L.z_tokens, L.z_dim), amp=1.0, max_bytes=max_bytes)
    memory = _text_tensor(
        f"<source>{source}</source>\n{prompt_block}\n{candidate_block}",
        (L.mem_slots, L.mem_dim),
        amp=0.01,
        max_bytes=max_bytes,
    )
    relations = _text_tensor(
        f"<relations>{regime}|{source}</relations>\n<prompt>{prompt[:1024]}</prompt>\n<response>{candidate[:1024]}</response>",
        (L.rel_edges, L.rel_dim),
        amp=0.01,
        max_bytes=max_bytes,
    )
    candidate_z = _text_tensor(candidate, (L.z_tokens, L.z_dim), amp=1.0, max_bytes=max_bytes)

    z = z.unsqueeze(0).to(device=device, dtype=dtype)
    memory = memory.unsqueeze(0).to(device=device, dtype=dtype)
    relations = relations.unsqueeze(0).to(device=device, dtype=dtype)
    candidate_z = candidate_z.unsqueeze(0).to(device=device, dtype=dtype)
    pad = _pad_anchor(1, L.pad_size, dtype=dtype, device=device)

    with torch.no_grad():
        cubo = model.cubo(z, memory, relations)["vector"].to(device=device, dtype=dtype)

    return FoldedOmegaCuboState(z=z, memory=memory, relations=relations, cubo=cubo, pad=pad), candidate_z


def score_candidate(
    model: FoldedCTNetOmegaCubo26,
    prompt: str,
    candidate: str,
    *,
    ticks: int,
    source: str,
    regime: str,
    device: torch.device,
    dtype: torch.dtype,
    max_bytes: int,
    weights: Dict[str, float],
) -> CandidateScore:
    state, candidate_z = build_dialogue_state(
        model,
        prompt,
        candidate,
        source=source,
        regime=regime,
        device=device,
        dtype=dtype,
        max_bytes=max_bytes,
    )
    x0 = model.pack(state)

    with torch.no_grad():
        out = state
        for _ in range(max(1, ticks)):
            out = model.forward_state(out)

        xi = model.pack(out)
        recovered = model.inverse_state(out)
        rev_mae = (model.pack(recovered) - x0).abs().mean()
        obs = model.cubo_observation(out)
        coh, _, _ = model.core.coherence_energy(xi)
        up_total, up_metrics = all_perspective_up_loss(model, state, out)
        anchor = F.mse_loss(out.z, candidate_z)

        omega = obs["omega"].mean()
        residual = obs["residual"].mean()
        absorption = obs["absorption"].mean()
        closure_score = obs["closure_score"].mean()

        score = (
            weights["up"] * up_total
            + weights["coh"] * coh
            + weights["omega"] * omega
            + weights["rev"] * rev_mae
            + weights["anchor"] * anchor
            + weights["residual"] * torch.relu(residual - absorption)
            - weights["closure"] * closure_score
        )

    return CandidateScore(
        candidate=candidate,
        score=float(score.detach().cpu()),
        up_total=up_metrics["up_total"],
        up_z=up_metrics["up_z"],
        up_memory=up_metrics["up_memory"],
        up_relations=up_metrics["up_relations"],
        up_cubo=up_metrics["up_cubo"],
        up_xi=up_metrics["up_xi"],
        up_delta=up_metrics["up_delta"],
        coherence_energy=float(coh.detach().cpu()),
        omega=float(omega.detach().cpu()),
        residual=float(residual.detach().cpu()),
        absorption=float(absorption.detach().cpu()),
        closure_score=float(closure_score.detach().cpu()),
        rev_mae=float(rev_mae.detach().cpu()),
        anchor_mse=float(anchor.detach().cpu()),
    )


def read_candidates(args: argparse.Namespace) -> List[str]:
    candidates: List[str] = []
    candidates.extend(args.candidate or [])
    for path in args.candidates_file or []:
        text = Path(path).read_text(encoding="utf-8")
        if args.jsonl:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                candidates.append(str(obj.get("candidate", obj.get("text", ""))))
        else:
            # Blank-line separated candidates. Single-line files also work.
            chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
            candidates.extend(chunks)
    return [c for c in candidates if c.strip()]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Rank candidate replies by CTNet u=p closure compatibility, not by autoregressive decoding."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--candidate", action="append", default=[], help="Candidate reply. Can be passed multiple times.")
    p.add_argument("--candidates-file", action="append", default=[], help="File with candidates separated by blank lines, or JSONL with --jsonl.")
    p.add_argument("--jsonl", action="store_true", help="Read candidate files as JSONL containing candidate/text fields.")
    p.add_argument("--ticks", type=int, default=1)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--fp64", action="store_true")
    p.add_argument("--max-bytes", type=int, default=2048)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--json", action="store_true")

    p.add_argument("--w-up", type=float, default=1.0)
    p.add_argument("--w-coh", type=float, default=0.05)
    p.add_argument("--w-omega", type=float, default=0.25)
    p.add_argument("--w-rev", type=float, default=0.10)
    p.add_argument("--w-anchor", type=float, default=0.05)
    p.add_argument("--w-residual", type=float, default=0.25)
    p.add_argument("--w-closure", type=float, default=0.10)

    args = p.parse_args()
    candidates = read_candidates(args)
    if not candidates:
        raise SystemExit("No candidates supplied. Use --candidate or --candidates-file. CTNet does not free-decode text in this probe.")

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if args.fp64 else torch.float32

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    saved_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    model = make_model_from_args(saved_args, device=device, dtype=dtype)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    weights = {
        "up": args.w_up,
        "coh": args.w_coh,
        "omega": args.w_omega,
        "rev": args.w_rev,
        "anchor": args.w_anchor,
        "residual": args.w_residual,
        "closure": args.w_closure,
    }

    ranked = [
        score_candidate(
            model,
            args.prompt,
            candidate,
            ticks=args.ticks,
            source="probe_dialogue",
            regime="ctnet_response_compatibility_probe",
            device=device,
            dtype=dtype,
            max_bytes=args.max_bytes,
            weights=weights,
        )
        for candidate in candidates
    ]
    ranked.sort(key=lambda x: x.score)

    report = {
        "mode": "ctnet_response_compatibility_probe",
        "interpretation": "Lower score means the candidate preserves CTNet closure better under u=p/coherence/omega/reversibility.",
        "prompt": args.prompt,
        "checkpoint": args.checkpoint,
        "ticks": args.ticks,
        "weights": weights,
        "best": ranked[0].__dict__,
        "ranked": [r.__dict__ for r in ranked[: max(1, args.top_k)]],
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("=== CTNet response compatibility probe ===")
        print("mode: candidate response ranking by u=p closure, not free decoding")
        print(f"checkpoint: {args.checkpoint}")
        print(f"ticks: {args.ticks}")
        print(f"prompt: {args.prompt}")
        print()
        for i, r in enumerate(ranked[: max(1, args.top_k)], start=1):
            print(f"#{i} score={r.score:.6e} up={r.up_total:.3e} coh={r.coherence_energy:.3e} omega={r.omega:.1e} rev={r.rev_mae:.1e} closure={r.closure_score:.3e}")
            print(r.candidate)
            print()


if __name__ == "__main__":
    main()

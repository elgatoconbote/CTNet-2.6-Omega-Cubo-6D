#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTNet response-state probe.

This probe does not fabricate replies.
It does not rank prefabricated candidate sentences.
It does not decode free text with a fake tokenizer.

CTNet responds because the question deforms its contextual mass and the system
moves toward the most coherent response state. In the current formulation, the
criterion of that coherence is:

    u = p

across all exposed scales and perspectives.

Therefore this probe measures whether a prompt produces a coherent
question-to-response transition inside CTNet:

    prompt
    -> contextual mass
    -> question state
    -> response-intention state
    -> u=p closure diagnostics

Until a trained linguistic observer/readout exists, this file intentionally does
not print a natural-language answer. It reports whether CTNet has formed a
coherent response state. The textual observer is a later visible reinscription
layer; it must not be faked here.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from ctnet_omega_cubo6d_plegado_ctnet26 import (
    FoldLayout,
    FoldedCTNetOmegaCubo26,
    FoldedOmegaCuboState,
)


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
    return torch.tanh(v + 0.015 * torch.sin(phase) + 0.0075 * torch.cos(2.0 * phase))


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


def perspective_up(model: FoldedCTNetOmegaCubo26, state: FoldedOmegaCuboState, *, reference: FoldedOmegaCuboState | None = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    xi = model.pack(state)
    if reference is None:
        delta = xi
    else:
        delta = xi - model.pack(reference)

    z_up = multiscale_up_loss(state.z)
    mem_up = multiscale_up_loss(state.memory)
    rel_up = multiscale_up_loss(state.relations)
    cubo_up = multiscale_up_loss(state.cubo)
    xi_up = multiscale_up_loss(xi)
    delta_up = multiscale_up_loss(delta)
    total = torch.stack([z_up, mem_up, rel_up, cubo_up, xi_up, delta_up]).mean()

    return total, {
        "up_total": total,
        "up_z": z_up,
        "up_memory": mem_up,
        "up_relations": rel_up,
        "up_cubo": cubo_up,
        "up_xi": xi_up,
        "up_delta": delta_up,
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


def build_question_state(
    model: FoldedCTNetOmegaCubo26,
    prompt: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_bytes: int,
) -> FoldedOmegaCuboState:
    L = model.layout
    regime = "question_to_response"
    source = "probe_prompt"
    framed = f"<regime>{regime}</regime>\n<question>\n{prompt}\n</question>\n<intent>respond_coherently</intent>"

    z = _text_tensor(framed, (L.z_tokens, L.z_dim), amp=1.0, max_bytes=max_bytes)
    memory = _text_tensor(
        f"<source>{source}</source>\n{framed}",
        (L.mem_slots, L.mem_dim),
        amp=0.01,
        max_bytes=max_bytes,
    )
    relations = _text_tensor(
        f"<relations>{regime}|{source}|respond_coherently</relations>\n{prompt[:1024]}",
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


def state_metrics(model: FoldedCTNetOmegaCubo26, state: FoldedOmegaCuboState, *, reference: FoldedOmegaCuboState | None = None) -> Dict[str, float]:
    xi = model.pack(state)
    coh, speed, info = model.core.coherence_energy(xi)
    obs = model.cubo_observation(state)
    up, parts = perspective_up(model, state, reference=reference)

    if reference is None:
        rev_mae = torch.zeros((), device=xi.device, dtype=xi.dtype)
    else:
        recovered = model.inverse_state(state)
        rev_mae = (model.pack(recovered) - model.pack(reference)).abs().mean()

    metrics = {
        "up": up,
        "coh": coh,
        "speed": speed,
        "info": info,
        "omega": obs["omega"].mean(),
        "residual": obs["residual"].mean(),
        "absorption": obs["absorption"].mean(),
        "closure_score": obs["closure_score"].mean(),
        "rev_mae": rev_mae,
        **parts,
    }
    return {k: float(v.detach().cpu()) for k, v in metrics.items()}


def response_state_probe(model: FoldedCTNetOmegaCubo26, prompt: str, *, ticks: int, device: torch.device, dtype: torch.dtype, max_bytes: int) -> Dict:
    question = build_question_state(model, prompt, device=device, dtype=dtype, max_bytes=max_bytes)

    with torch.no_grad():
        understood = question
        for _ in range(max(1, ticks)):
            understood = model.forward_state(understood)

        response = understood
        for _ in range(max(1, ticks)):
            response = model.forward_state(response)

    q_metrics = state_metrics(model, question)
    understood_metrics = state_metrics(model, understood, reference=question)
    response_metrics = state_metrics(model, response, reference=understood)

    # Readiness is not a text score. It is a structural diagnostic: low u/p debt,
    # low omega, low reversibility debt, and high closure indicate that CTNet has
    # moved into a coherent response state.
    response_debt = (
        response_metrics["up"]
        + 0.05 * response_metrics["coh"]
        + 0.25 * response_metrics["omega"]
        + 0.10 * response_metrics["rev_mae"]
        - 0.10 * response_metrics["closure_score"]
    )

    return {
        "mode": "ctnet_response_state_probe",
        "principle": "question deforms contextual mass; response is coherent when u=p closes across perspectives",
        "warning": "No natural-language answer is printed because no trained linguistic observer/readout is present yet.",
        "prompt": prompt,
        "ticks": ticks,
        "question_state": q_metrics,
        "understood_state": understood_metrics,
        "response_state": response_metrics,
        "response_debt": response_debt,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Probe whether CTNet forms a coherent u=p response state for a question.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--ticks", type=int, default=1)
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

    report = response_state_probe(model, args.prompt, ticks=args.ticks, device=device, dtype=dtype, max_bytes=args.max_bytes)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("=== CTNet response-state probe ===")
        print("principle: question -> contextual mass -> coherent response state by u=p")
        print("warning: no fabricated natural-language answer; linguistic observer not trained yet")
        print(f"checkpoint: {args.checkpoint}")
        print(f"prompt: {args.prompt}")
        print(f"response_debt: {report['response_debt']:.6e}")
        print("\nresponse_state:")
        for k, v in report["response_state"].items():
            print(f"  {k}: {v:.6e}")


if __name__ == "__main__":
    main()

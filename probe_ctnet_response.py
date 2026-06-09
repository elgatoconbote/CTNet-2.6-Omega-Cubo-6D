#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTNet output-closure probe.

CTNet does not answer by selecting prefabricated text.
A question deforms contextual mass. The answer is the closing form of that
deformation in the available output chart.

The invariant is:

    u = p

This probe does not fabricate natural language. It measures whether a prompt
forms a coherent output state under u=p. A later visible chart can reinscribe
that state as text, symbols or another output format.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from ctnet_omega_cubo6d_plegado_ctnet26 import FoldLayout, FoldedCTNetOmegaCubo26, FoldedOmegaCuboState


def _byte_signal(text: str, size: int, *, max_bytes: int = 2048) -> torch.Tensor:
    raw = (text or "").encode("utf-8", errors="ignore")[:max_bytes] or b"<empty>"
    v = torch.zeros(size, dtype=torch.float32)
    for i, b in enumerate(raw):
        v[i % size] += ((float(b) / 127.5) - 1.0) / math.sqrt(1.0 + (i // size))
    phase = torch.linspace(0, 2.0 * math.pi, size, dtype=torch.float32)
    return torch.tanh(v + 0.015 * torch.sin(phase) + 0.0075 * torch.cos(2.0 * phase))


def _text_tensor(text: str, shape: Tuple[int, ...], *, amp: float = 1.0, max_bytes: int = 2048) -> torch.Tensor:
    n = 1
    for s in shape:
        n *= int(s)
    return (amp * _byte_signal(text, n, max_bytes=max_bytes)).reshape(*shape)


def _pad(batch: int, size: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if size <= 0:
        return torch.zeros(batch, 0, dtype=dtype, device=device)
    phase = torch.linspace(0, 2.0 * math.pi, size, dtype=dtype, device=device)
    return (0.01 * (torch.sin(phase) + 0.5 * torch.cos(2.0 * phase))).unsqueeze(0).repeat(batch, 1)


def _even(x: torch.Tensor) -> torch.Tensor:
    return x if x.shape[-1] % 2 == 0 else F.pad(x, (0, 1))


def _up(x: torch.Tensor) -> torch.Tensor:
    x = _even(x)
    h = x.shape[-1] // 2
    return F.mse_loss(x[..., :h], x[..., h:])


def _pool(x: torch.Tensor, scale: int) -> torch.Tensor:
    if x.ndim != 3 or x.shape[1] < scale:
        return x
    b, n, d = x.shape
    usable = (n // scale) * scale
    return x[:, :usable, :].reshape(b, usable // scale, scale, d).mean(dim=2)


def multiscale_up(x: torch.Tensor) -> torch.Tensor:
    terms: List[torch.Tensor] = [_up(x)]
    for shift in (1, 2, 3):
        if x.shape[-1] > shift:
            terms.append(_up(torch.roll(x, shifts=shift, dims=-1)))
    if x.ndim == 3:
        for shift in (1, 2, 4):
            if x.shape[1] > shift:
                terms.append(_up(torch.roll(x, shifts=shift, dims=1)))
        for scale in (2, 4, 8):
            if x.shape[1] >= scale:
                p = _pool(x, scale)
                terms.append(_up(p))
                if p.shape[1] > 1:
                    terms.append(_up(torch.roll(p, shifts=1, dims=1)))
    return torch.stack(terms).mean()


def make_model(args_dict: Dict, *, device: torch.device, dtype: torch.dtype) -> FoldedCTNetOmegaCubo26:
    layout = FoldLayout(
        N=int(args_dict.get("N", 64)),
        d=int(args_dict.get("d", 16)),
        z_tokens=int(args_dict.get("z_tokens", args_dict.get("z-tokens", 32))),
        z_dim=int(args_dict.get("z_dim", args_dict.get("z-dim", 16))),
        mem_slots=int(args_dict.get("mem_slots", args_dict.get("mem-slots", 8))),
        mem_dim=int(args_dict.get("mem_dim", args_dict.get("mem-dim", 16))),
        rel_edges=int(args_dict.get("rel_edges", args_dict.get("rel-edges", 8))),
        rel_dim=int(args_dict.get("rel_dim", args_dict.get("rel-dim", 16))),
    )
    layout.validate()
    return FoldedCTNetOmegaCubo26(
        layout=layout,
        fractal_steps=int(args_dict.get("fractal_steps", args_dict.get("fractal-steps", 4))),
        latent_steps=int(args_dict.get("latent_steps", args_dict.get("latent-steps", 2))),
        cubo_shear=float(args_dict.get("cubo_shear", args_dict.get("cubo-shear", 0.05))),
    ).to(device=device, dtype=dtype)


def build_state(model: FoldedCTNetOmegaCubo26, prompt: str, *, device: torch.device, dtype: torch.dtype, max_bytes: int) -> FoldedOmegaCuboState:
    L = model.layout
    framed = f"<question>{prompt}</question>\n<closure>u=p</closure>\n<output>coherent_closing_form</output>"
    z = _text_tensor(framed, (L.z_tokens, L.z_dim), amp=1.0, max_bytes=max_bytes).unsqueeze(0).to(device=device, dtype=dtype)
    mem = _text_tensor(f"<memory>{framed}</memory>", (L.mem_slots, L.mem_dim), amp=0.01, max_bytes=max_bytes).unsqueeze(0).to(device=device, dtype=dtype)
    rel = _text_tensor(f"<relations>question|output|u=p</relations>\n{prompt[:1024]}", (L.rel_edges, L.rel_dim), amp=0.01, max_bytes=max_bytes).unsqueeze(0).to(device=device, dtype=dtype)
    pad = _pad(1, L.pad_size, dtype=dtype, device=device)
    with torch.no_grad():
        cubo = model.cubo(z, mem, rel)["vector"].to(device=device, dtype=dtype)
    return FoldedOmegaCuboState(z=z, memory=mem, relations=rel, cubo=cubo, pad=pad)


def state_metrics(model: FoldedCTNetOmegaCubo26, state: FoldedOmegaCuboState, reference: FoldedOmegaCuboState | None = None) -> Dict[str, float]:
    xi = model.pack(state)
    delta = xi if reference is None else xi - model.pack(reference)
    coh, speed, info = model.core.coherence_energy(xi)
    obs = model.cubo_observation(state)
    vals = {
        "up_z": multiscale_up(state.z),
        "up_memory": multiscale_up(state.memory),
        "up_relations": multiscale_up(state.relations),
        "up_cubo": multiscale_up(state.cubo),
        "up_xi": multiscale_up(xi),
        "up_delta": multiscale_up(delta),
    }
    up_total = torch.stack(list(vals.values())).mean()
    if reference is not None:
        recovered = model.inverse_state(state)
        rev = (model.pack(recovered) - model.pack(reference)).abs().mean()
    else:
        rev = torch.zeros((), device=xi.device, dtype=xi.dtype)
    vals.update({
        "up": up_total,
        "coh": coh,
        "speed": speed,
        "info": info,
        "omega": obs["omega"].mean(),
        "residual": obs["residual"].mean(),
        "absorption": obs["absorption"].mean(),
        "closure_score": obs["closure_score"].mean(),
        "rev_mae": rev,
    })
    return {k: float(v.detach().cpu()) for k, v in vals.items()}


def probe(model: FoldedCTNetOmegaCubo26, prompt: str, *, ticks: int, device: torch.device, dtype: torch.dtype, max_bytes: int) -> Dict:
    question_state = build_state(model, prompt, device=device, dtype=dtype, max_bytes=max_bytes)
    with torch.no_grad():
        output_state = question_state
        for _ in range(max(1, ticks)):
            output_state = model.forward_state(output_state)
    q = state_metrics(model, question_state)
    o = state_metrics(model, output_state, question_state)
    debt = o["up"] + 0.05 * o["coh"] + 0.25 * o["omega"] + 0.10 * o["rev_mae"] - 0.10 * o["closure_score"]
    return {
        "mode": "ctnet_output_closure_probe",
        "principle": "question deforms contextual mass; answer is the coherent closing form under u=p",
        "warning": "No text is fabricated. A visible output chart must be trained to reinscribe this state.",
        "prompt": prompt,
        "ticks": ticks,
        "question_state": q,
        "output_state": o,
        "output_debt": debt,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Probe CTNet output closure under u=p.")
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
    saved = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    model = make_model(saved, device=device, dtype=dtype)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    report = probe(model, args.prompt, ticks=args.ticks, device=device, dtype=dtype, max_bytes=args.max_bytes)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("=== CTNet output-closure probe ===")
        print("principle: answer is coherent closing form under u=p")
        print("warning: no fabricated text; visible chart must be trained")
        print(f"checkpoint: {args.checkpoint}")
        print(f"prompt: {args.prompt}")
        print(f"output_debt: {report['output_debt']:.6e}")
        for k, v in report["output_state"].items():
            print(f"  {k}: {v:.6e}")


if __name__ == "__main__":
    main()

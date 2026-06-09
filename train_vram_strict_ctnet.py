#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTNet 2.6 Omega Cubo 6D strict online trainer.

Design target
-------------
- No Hugging Face datasets.
- No huggingface_hub.
- No pyarrow.
- No Parquet/Xet cache.
- No checkpoint or metrics file unless explicitly requested.
- CPU RAM only holds the current tiny text batch.
- CTNet tensors/gradients live on GPU when --cuda is used.

This is the correct mode for the structural requirement:
    sample -> fixed CTNet state -> GPU step -> discard sample

The default source URLs are public-domain text streams used only as a zero-disk
smoke test. Pass your own --url values for the real online corpus.
"""

from __future__ import annotations

import argparse
import math
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple

import torch
import torch.nn.functional as F

from ctnet_omega_cubo6d_plegado_ctnet26 import (
    FoldLayout,
    FoldedCTNetOmegaCubo26,
    FoldedOmegaCuboState,
    count_params,
)


DEFAULT_URLS = [
    "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
]


@dataclass
class OnlineSample:
    x: str
    y: str
    source: str
    regime: str = "zero_disk_online_text"


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


def _http_lines(url: str, *, timeout: float) -> Iterator[str]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "CTNetZeroDisk/1.0 (+https://github.com/elgatoconbote/CTNet-2.6-Omega-Cubo-6D)",
            "Accept": "text/plain,text/*,*/*;q=0.5",
            "Cache-Control": "no-store",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            yield raw.decode("utf-8", errors="ignore")


def online_blocks(urls: List[str], *, block_bytes: int, timeout: float) -> Iterator[OnlineSample]:
    if not urls:
        urls = DEFAULT_URLS[:]

    idx = 0
    failures = 0
    while True:
        url = urls[idx % len(urls)]
        idx += 1
        try:
            buf: List[str] = []
            n = 0
            for line in _http_lines(url, timeout=timeout):
                if not line.strip():
                    continue
                buf.append(line)
                n += len(line.encode("utf-8", errors="ignore"))
                if n >= block_bytes:
                    text = "".join(buf)[:block_bytes]
                    target = text[1:] + " "
                    yield OnlineSample(x=text, y=target, source=url)
                    buf = []
                    n = 0
            if buf:
                text = "".join(buf)[:block_bytes]
                target = text[1:] + " "
                yield OnlineSample(x=text, y=target, source=url)
            failures = 0
        except Exception as e:
            failures += 1
            print(f"source_error url={url!r} error={type(e).__name__}: {e}", flush=True)
            if failures >= max(3, len(urls)):
                raise RuntimeError("all online sources failed repeatedly; pass working --url values") from e


def batch_to_state(
    model: FoldedCTNetOmegaCubo26,
    samples: List[OnlineSample],
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
        z_rows.append(_text_tensor(f"<regime>{ex.regime}</regime>\n{ex.x}", (L.z_tokens, L.z_dim), amp=1.0, max_bytes=max_bytes))
        mem_rows.append(_text_tensor(f"<source>{ex.source}</source>\n<regime>{ex.regime}</regime>\n{ex.x}", (L.mem_slots, L.mem_dim), amp=0.01, max_bytes=max_bytes))
        rel_rows.append(_text_tensor(f"<relations>{ex.regime}|{ex.source}</relations>\n{ex.x[:1024]}", (L.rel_edges, L.rel_dim), amp=0.01, max_bytes=max_bytes))
        target_z_rows.append(_text_tensor(ex.y, (L.z_tokens, L.z_dim), amp=1.0, max_bytes=max_bytes))
        regimes.append(ex.regime)

    z = torch.stack(z_rows, dim=0).to(device=device, dtype=dtype, non_blocking=True)
    memory = torch.stack(mem_rows, dim=0).to(device=device, dtype=dtype, non_blocking=True)
    relations = torch.stack(rel_rows, dim=0).to(device=device, dtype=dtype, non_blocking=True)
    target_z = torch.stack(target_z_rows, dim=0).to(device=device, dtype=dtype, non_blocking=True)
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
    stream = online_blocks(args.url, block_bytes=args.block_bytes, timeout=args.timeout)

    print("=== CTNet strict zero-disk online training ===", flush=True)
    print("loader=no datasets/no huggingface_hub/no pyarrow/no xet", flush=True)
    print(f"device={device} dtype={dtype} params={count_params(model)}", flush=True)
    print(f"layout capacity={layout.capacity} semantic_size={layout.semantic_size} pad_size={layout.pad_size}", flush=True)
    print(f"fixed memory M=[B,{layout.mem_slots},{layout.mem_dim}] relations R=[B,{layout.rel_edges},{layout.rel_dim}]", flush=True)
    print(f"urls={args.url or DEFAULT_URLS}", flush=True)
    print("writes=none unless --save-final is passed", flush=True)

    t0 = time.time()
    last: Dict = {}

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

        if args.empty_cache_every > 0 and step % args.empty_cache_every == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            last = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "loss_task": float(loss_task.detach().cpu()),
                "loss_coh": float(loss_coh.detach().cpu()),
                "loss_omega": float(loss_omega.detach().cpu()),
                "loss_rev": float(loss_rev.detach().cpu()),
                "speed": float(speed.detach().cpu()),
                "info": float(info.detach().cpu()),
                "mem_slot_var": float(mem_var.detach().cpu()),
                "rel_slot_var": float(rel_var.detach().cpu()),
                "elapsed_sec": elapsed,
                "source": samples[0].source,
                "regimes": regimes,
            }
            print(
                f"step {step:6d} | loss={last['loss']:.6e} task={last['loss_task']:.6e} "
                f"omega={last['loss_omega']:.6e} coh={last['loss_coh']:.6e} "
                f"rev={last['loss_rev']:.2e} speed={last['speed']:.3f} time={elapsed:.1f}s",
                flush=True,
            )

    if args.save_final:
        torch.save({"model_state_dict": model.state_dict(), "last": last, "args": vars(args)}, args.save_final)
        print(f"saved_final={args.save_final}", flush=True)

    return last


def main() -> None:
    p = argparse.ArgumentParser(description="Strict zero-disk CTNet trainer: online text -> fixed state -> GPU -> discard.")
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

    p.add_argument("--lambda-task", type=float, default=1.0)
    p.add_argument("--lambda-coh", type=float, default=0.05)
    p.add_argument("--lambda-omega", type=float, default=0.25)
    p.add_argument("--lambda-cubo", type=float, default=0.05)
    p.add_argument("--lambda-structure", type=float, default=0.10)
    p.add_argument("--lambda-rev", type=float, default=0.10)
    p.add_argument("--reversibility-loss-every", type=int, default=10)
    p.add_argument("--min-slot-var", type=float, default=1e-8)

    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--empty-cache-every", type=int, default=25)
    p.add_argument("--save-final", default="", help="Optional path. By default nothing is written.")

    args = p.parse_args()
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()

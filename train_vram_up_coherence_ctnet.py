#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTNet 2.6 Omega Cubo 6D trainer with u=p coherence as the primary signal.

Every observer is contextual reality. External reality and CTNet's own internal
processes enter through the same observer path:

    observation -> Observador -> batch_to_state -> contextual mass -> CTNet -> u=p

There is no external observer module. CTNet observes while it runs, and what it
observes becomes contextual mass through the same chart used by the external
dataset.

The surface training loop can look close to a standard online trainer: stream text,
encode a fixed state, run CTNet, compute a loss, update weights.

The difference is what the loss means.

In this trainer the central criterion is not merely next-byte prediction and not
candidate ranking. The central criterion is CTNet coherence:

    u = p

at every scale and from every perspective that the fixed CTNet state exposes.

Perspectives measured:
- Z state,
- fixed memory M,
- fixed relation bank R,
- Cubo 6D coordinates C6,
- full packed state Xi,
- deformation DeltaXi = Xi_out - Xi_in.

Scales / perspectives measured:
- raw split on the last dimension,
- token pooling at scales 2, 4, 8 when a token axis exists,
- token rolls when a token axis exists,
- channel rolls along the feature axis.

The byte/text target remains only as an anchoring task. It keeps the chart tied to
language, but the main signal is modal closure u/p plus the CTNet coherence tensor.

No corpus cache, no Hugging Face datasets, no pyarrow, no vector store, no KV-cache.
The final CTNet deformation is saved to disk by default.
"""

from __future__ import annotations

import argparse
import math
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

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
class Observador:
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


def flujo_observadores(urls: List[str], *, block_bytes: int, timeout: float) -> Iterator[Observador]:
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
                    yield Observador(x=text, y=target, source=url)
                    buf = []
                    n = 0
            if buf:
                text = "".join(buf)[:block_bytes]
                target = text[1:] + " "
                yield Observador(x=text, y=target, source=url)
            failures = 0
        except Exception as e:
            failures += 1
            print(f"source_error url={url!r} error={type(e).__name__}: {e}", flush=True)
            if failures >= max(3, len(urls)):
                raise RuntimeError("all online sources failed repeatedly; pass working --url values") from e


def batch_to_state(
    model: FoldedCTNetOmegaCubo26,
    samples: List[Observador],
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
    """Pool token axis of [B,N,D] by averaging groups."""
    if x.ndim != 3 or x.shape[1] < scale:
        return x
    b, n, d = x.shape
    usable = (n // scale) * scale
    if usable <= 0:
        return x
    y = x[:, :usable, :].reshape(b, usable // scale, scale, d).mean(dim=2)
    return y


def multiscale_up_loss(x: torch.Tensor, *, token_scales: Tuple[int, ...] = (2, 4, 8)) -> torch.Tensor:
    """u=p loss across scales and perspectives for one tensor."""
    terms: List[torch.Tensor] = []

    terms.append(_up_mse_last_dim(x))

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
    """Compute u=p across every exposed CTNet perspective."""
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
    metrics = {
        "up_total": float(total.detach().cpu()),
        "up_z": float(z_up.detach().cpu()),
        "up_memory": float(mem_up.detach().cpu()),
        "up_relations": float(rel_up.detach().cpu()),
        "up_cubo": float(cubo_up.detach().cpu()),
        "up_xi": float(xi_up.detach().cpu()),
        "up_delta": float(delta_up.detach().cpu()),
    }
    return total, metrics



def _scalar(x: torch.Tensor) -> float:
    return float(x.detach().to(torch.float32).mean().cpu())


def _tensor_reality_line(name: str, x: torch.Tensor) -> str:
    y = x.detach().to(torch.float32)
    return (
        f"{name}.mean={float(y.mean().cpu()):.6e} "
        f"{name}.rms={float(y.pow(2).mean().sqrt().cpu()):.6e} "
        f"{name}.abs={float(y.abs().mean().cpu()):.6e}"
    )


def observador_interno(
    model: FoldedCTNetOmegaCubo26,
    state: FoldedOmegaCuboState,
    out: FoldedOmegaCuboState,
    *,
    step: int,
    obs: Dict[str, torch.Tensor],
    up_metrics: Dict[str, float],
) -> Observador:
    """
    CTNet observes its own running process exactly as it observes any dataset item.

    This is not an external observer and not a diagnostic module. It serializes the
    current internal transition as observed reality, then feeds it back through the
    same Observador -> batch_to_state -> contextual mass route used by external
    text.

    Everything observed is dataset.
    Everything becomes contextual mass.
    Everything is closed by u=p.
    """
    xi_in = model.pack(state)
    xi_out = model.pack(out)
    dxi = xi_out - xi_in

    text = "\n".join(
        [
            "<reality_observation>",
            "<source>ctnet_internal_process</source>",
            "<principle>u=p</principle>",
            "<meaning>question_or_input_deforms_state_response_is_coherent_closure</meaning>",
            f"step={step}",
            _tensor_reality_line("z_in", state.z),
            _tensor_reality_line("z_out", out.z),
            _tensor_reality_line("memory_in", state.memory),
            _tensor_reality_line("memory_out", out.memory),
            _tensor_reality_line("relations_in", state.relations),
            _tensor_reality_line("relations_out", out.relations),
            _tensor_reality_line("cubo_in", state.cubo),
            _tensor_reality_line("cubo_out", out.cubo),
            _tensor_reality_line("xi_in", xi_in),
            _tensor_reality_line("xi_out", xi_out),
            _tensor_reality_line("delta_xi", dxi),
            f"omega={_scalar(obs['omega']):.6e}",
            f"residual={_scalar(obs['residual']):.6e}",
            f"absorption={_scalar(obs['absorption']):.6e}",
            f"closure_score={_scalar(obs['closure_score']):.6e}",
            " ".join(f"{k}={v:.6e}" for k, v in sorted(up_metrics.items())),
            "</reality_observation>",
        ]
    )

    return Observador(
        x=text,
        y=text[1:] + " ",
        source="ctnet://internal_process",
        regime="ctnet_internal_reality",
    )


def loss_observador(
    model: FoldedCTNetOmegaCubo26,
    sample: Observador,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_bytes: int,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Trains an observed internal process through the same chart as the dataset.

    observed process -> Observador -> batch_to_state -> contextual mass -> CTNet -> u=p
    """
    obs_state, obs_target_z, _ = batch_to_state(
        model,
        [sample],
        device=device,
        dtype=dtype,
        max_bytes=max_bytes,
    )
    obs_out = model.forward_state(obs_state)
    obs_xi_out = model.pack(obs_out)

    loss_obs_up, obs_up_metrics = all_perspective_up_loss(model, obs_state, obs_out)
    loss_obs_anchor = F.mse_loss(obs_out.z, obs_target_z)
    loss_obs_coh, _, _ = model.core.coherence_energy(obs_xi_out)
    obs_cubo = model.cubo_observation(obs_out)
    loss_obs_omega = obs_cubo["omega"].mean()

    loss_obs_stream = (
        loss_obs_up
        + args.lambda_anchor * loss_obs_anchor
        + args.lambda_coh * loss_obs_coh
        + args.lambda_omega * loss_obs_omega
    )

    metrics = {
        "loss_internal_stream": float(loss_obs_stream.detach().cpu()),
        "loss_internal_up": float(loss_obs_up.detach().cpu()),
        "loss_internal_anchor": float(loss_obs_anchor.detach().cpu()),
        "loss_internal_coh": float(loss_obs_coh.detach().cpu()),
        "loss_internal_omega": float(loss_obs_omega.detach().cpu()),
        **{f"internal_{k}": v for k, v in obs_up_metrics.items()},
    }
    return loss_obs_stream, metrics




def salida_visible_desde_estado(out: FoldedOmegaCuboState, *, max_chars: int = 512) -> str:
    """
    Convierte la voluntad de cierre del estado CTNet en acción textual visible.

    No es next-token autoregresivo. Es una carta textual de efector:
    estado deformado -> campo de cierre -> símbolos visibles.

    El producto visible se reobserva después como Observador.
    """
    z = out.z.detach().to(torch.float32).flatten()
    if z.numel() == 0:
        return ""

    # Normalización local: convierte la deformación en una trayectoria de símbolos.
    z = torch.tanh(z)
    vals = ((z + 1.0) * 0.5 * 94.0 + 32.0).clamp(32, 126).to(torch.int64)

    chars = []
    last = None
    repeat = 0
    for v in vals[: max_chars * 4]:
        c = chr(int(v.item()))

        # Evita una salida totalmente bloqueada por repeticiones largas.
        if c == last:
            repeat += 1
            if repeat > 3:
                continue
        else:
            repeat = 0
            last = c

        chars.append(c)
        if len(chars) >= max_chars:
            break

    text = "".join(chars).strip()

    # Si la carta sale casi vacía, emite al menos una acción visible mínima.
    if not text:
        text = "u=p"

    return text


def efector_textual(
    model: FoldedCTNetOmegaCubo26,
    state: FoldedOmegaCuboState,
    out: FoldedOmegaCuboState,
    *,
    step: int,
    obs: Dict[str, torch.Tensor],
    up_metrics: Dict[str, float],
) -> Observador:
    """
    Efector textual de CTNet.

    No es un decoder autoregresivo. Es una carta de acción visible:
    el estado deformado genera una voluntad de cierre y esa voluntad se
    reinscribe como producto textual observable.

    pregunta/entrada -> deformación -> voluntad de cierre -> efector -> producto
    producto -> Observador -> batch_to_state -> masa contextual -> u=p
    """
    xi_in = model.pack(state)
    xi_out = model.pack(out)
    dxi = xi_out - xi_in
    salida_visible = salida_visible_desde_estado(out)

    producto = "\n".join(
        [
            "<producto_efector>",
            "<tipo>textual</tipo>",
            "<principio>u=p</principio>",
            "<respuesta>voluntad_de_cierre</respuesta>",
            "<sentido>CTNet_responde_porque_la_pregunta_deforma_estado</sentido>",
            "<criterio>la_accion_correcta_es_el_cierre_mas_coherente</criterio>",
            f"step={step}",
            _tensor_reality_line("estado_deformado_z", out.z),
            _tensor_reality_line("estado_deformado_memory", out.memory),
            _tensor_reality_line("estado_deformado_relations", out.relations),
            _tensor_reality_line("estado_deformado_cubo", out.cubo),
            _tensor_reality_line("xi_in", xi_in),
            _tensor_reality_line("xi_out", xi_out),
            _tensor_reality_line("delta_xi", dxi),
            f"omega={_scalar(obs['omega']):.6e}",
            f"residual={_scalar(obs['residual']):.6e}",
            f"absorption={_scalar(obs['absorption']):.6e}",
            f"closure_score={_scalar(obs['closure_score']):.6e}",
            " ".join(f"{k}={v:.6e}" for k, v in sorted(up_metrics.items())),
            "<accion_visible>escribir_simbolos_para_cerrar_deformacion_contextual</accion_visible>",
            "<salida_visible>",
            salida_visible,
            "</salida_visible>",
            "</producto_efector>",
        ]
    )

    return Observador(
        x=producto,
        y=producto[1:] + " ",
        source="ctnet://efector/textual",
        regime="producto_efector_textual",
    )


def loss_efector(
    model: FoldedCTNetOmegaCubo26,
    producto: Observador,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_bytes: int,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    El producto del efector vuelve a entrar como observación.

    producto visible -> Observador -> batch_to_state -> masa contextual -> CTNet -> u=p
    """
    loss, metrics = loss_observador(
        model,
        producto,
        device=device,
        dtype=dtype,
        max_bytes=max_bytes,
        args=args,
    )

    renamed = {}
    for k, v in metrics.items():
        renamed[k.replace("internal", "efector")] = v

    if "loss_efector_stream" not in renamed:
        renamed["loss_efector_stream"] = float(loss.detach().cpu())

    return loss, renamed


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
    stream = flujo_observadores(args.url, block_bytes=args.block_bytes, timeout=args.timeout)

    print("=== CTNet u=p multiscale coherence training ===", flush=True)
    print("objective=u=p at all exposed scales/perspectives + CT coherence tensor", flush=True)
    print("self_observation=internal processes are fed through the same observer chart", flush=True)
    print("efector=textual product is observed back through the same observer chart", flush=True)
    print("loader=no datasets/no huggingface_hub/no pyarrow/no xet", flush=True)
    print(f"device={device} dtype={dtype} params={count_params(model)}", flush=True)
    print(f"layout capacity={layout.capacity} semantic_size={layout.semantic_size} pad_size={layout.pad_size}", flush=True)
    print(f"fixed memory M=[B,{layout.mem_slots},{layout.mem_dim}] relations R=[B,{layout.rel_edges},{layout.rel_dim}]", flush=True)
    print(f"urls={args.url or DEFAULT_URLS}", flush=True)
    print(f"save_final={args.save_final}", flush=True)

    t0 = time.time()
    last: Dict = {}

    for step in range(1, args.steps + 1):
        samples: List[Observador] = [next(stream) for _ in range(args.batch)]
        state, target_z, regimes = batch_to_state(model, samples, device=device, dtype=dtype, max_bytes=args.max_bytes)

        optimizer.zero_grad(set_to_none=True)
        out = model.forward_state(state)
        xi_out = model.pack(out)

        loss_up, up_metrics = all_perspective_up_loss(model, state, out)
        loss_anchor = F.mse_loss(out.z, target_z)
        loss_coh, speed, info = model.core.coherence_energy(xi_out)
        obs = model.cubo_observation(out)
        loss_omega = obs["omega"].mean()
        loss_cubo_track = F.mse_loss(out.cubo, obs["vector"].detach())

        if args.self_observation_every > 0 and step % args.self_observation_every == 0:
            internal_sample = observador_interno(
                model,
                state,
                out,
                step=step,
                obs=obs,
                up_metrics=up_metrics,
            )
            loss_internal_stream, internal_metrics = loss_observador(
                model,
                internal_sample,
                device=device,
                dtype=dtype,
                max_bytes=args.max_bytes,
                args=args,
            )
        else:
            loss_internal_stream = torch.zeros((), device=device, dtype=dtype)
            internal_metrics = {
                "loss_internal_stream": 0.0,
                "loss_internal_up": 0.0,
                "loss_internal_anchor": 0.0,
                "loss_internal_coh": 0.0,
                "loss_internal_omega": 0.0,
            }

        if args.efector_every > 0 and step % args.efector_every == 0:
            producto_efector = efector_textual(
                model,
                state,
                out,
                step=step,
                obs=obs,
                up_metrics=up_metrics,
            )
            loss_efector_stream, efector_metrics = loss_efector(
                model,
                producto_efector,
                device=device,
                dtype=dtype,
                max_bytes=args.max_bytes,
                args=args,
            )
        else:
            loss_efector_stream = torch.zeros((), device=device, dtype=dtype)
            efector_metrics = {
                "loss_efector_stream": 0.0,
                "loss_efector_up": 0.0,
                "loss_efector_anchor": 0.0,
                "loss_efector_coh": 0.0,
                "loss_efector_omega": 0.0,
            }

        mem_var = slot_variance(out.memory)
        rel_var = slot_variance(out.relations)
        loss_structure = F.relu(args.min_slot_var - mem_var) + F.relu(args.min_slot_var - rel_var)

        if args.reversibility_loss_every > 0 and step % args.reversibility_loss_every == 0:
            recovered = model.inverse_state(out)
            loss_rev = F.mse_loss(model.pack(recovered), model.pack(state))
        else:
            loss_rev = torch.zeros((), device=device, dtype=dtype)

        loss = (
            args.lambda_up * loss_up
            + args.lambda_anchor * loss_anchor
            + args.lambda_coh * loss_coh
            + args.lambda_omega * loss_omega
            + args.lambda_cubo * loss_cubo_track
            + args.lambda_structure * loss_structure
            + args.lambda_rev * loss_rev
            + args.lambda_self_observation * loss_internal_stream
            + args.lambda_efector * loss_efector_stream
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
                "loss_up": float(loss_up.detach().cpu()),
                "loss_anchor": float(loss_anchor.detach().cpu()),
                "loss_coh": float(loss_coh.detach().cpu()),
                "loss_omega": float(loss_omega.detach().cpu()),
                "loss_rev": float(loss_rev.detach().cpu()),
                "loss_internal_stream": float(loss_internal_stream.detach().cpu()),
                "loss_efector_stream": float(loss_efector_stream.detach().cpu()),
                **internal_metrics,
                **efector_metrics,
                "speed": float(speed.detach().cpu()),
                "info": float(info.detach().cpu()),
                "omega": float(obs["omega"].mean().detach().cpu()),
                "residual": float(obs["residual"].mean().detach().cpu()),
                "absorption": float(obs["absorption"].mean().detach().cpu()),
                "closure_score": float(obs["closure_score"].mean().detach().cpu()),
                "mem_slot_var": float(mem_var.detach().cpu()),
                "rel_slot_var": float(rel_var.detach().cpu()),
                "elapsed_sec": elapsed,
                "source": samples[0].source,
                "regimes": regimes,
                **up_metrics,
            }
            print(
                f"step {step:6d} | loss={last['loss']:.6e} up={last['loss_up']:.3e} "
                f"z={last['up_z']:.2e} m={last['up_memory']:.2e} r={last['up_relations']:.2e} "
                f"c6={last['up_cubo']:.2e} xi={last['up_xi']:.2e} dxi={last['up_delta']:.2e} "
                f"anchor={last['loss_anchor']:.3e} self={last['loss_internal_stream']:.3e} "
                f"efector={last['loss_efector_stream']:.3e} omega={last['omega']:.1e} "
                f"coh={last['loss_coh']:.3e} "
                f"rev={last['loss_rev']:.1e} time={elapsed:.1f}s",
                flush=True,
            )

    if args.save_final:
        path = Path(args.save_final)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "last": last, "args": vars(args)}, path)
        print(f"saved_final={path}", flush=True)

    return last


def main() -> None:
    p = argparse.ArgumentParser(description="CTNet trainer where u=p multiscale coherence is primary.")
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

    p.add_argument("--lambda-up", type=float, default=1.0)
    p.add_argument("--lambda-anchor", type=float, default=0.10)
    p.add_argument("--lambda-coh", type=float, default=0.05)
    p.add_argument("--lambda-omega", type=float, default=0.25)
    p.add_argument("--lambda-cubo", type=float, default=0.05)
    p.add_argument("--lambda-structure", type=float, default=0.10)
    p.add_argument("--lambda-rev", type=float, default=0.10)
    p.add_argument("--lambda-self-observation", type=float, default=0.25)
    p.add_argument("--self-observation-every", type=int, default=1)
    p.add_argument("--lambda-efector", type=float, default=0.25)
    p.add_argument("--efector-every", type=int, default=1)
    p.add_argument("--reversibility-loss-every", type=int, default=10)
    p.add_argument("--min-slot-var", type=float, default=1e-8)

    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--empty-cache-every", type=int, default=25)
    p.add_argument("--save-final", default="checkpoints/ctnet_up_state_final.pt")

    args = p.parse_args()
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTNet-Omega + InfiniteAtlas-MTHD integration.

Integra el atlas MTHD virtual con el estado plegado de CTNet-2.6-Omega-Cubo-6D.

    Omega = (Z, M, R, C6, pad)

Cada recibo abre una coordenada virtual de manifold y se pliega dentro de M y R
mediante un shear reversible:

    M' = M + alpha * Drive_M(receipt)
    R' = R + alpha * Drive_R(receipt)

La inversa es:

    M = M' - alpha * Drive_M(receipt)
    R = R' - alpha * Drive_R(receipt)

No hay slots MTHD, no hay capacity MTHD, no hay route allocation y no hay route
exhausted. El payload exacto se recupera por la capsula reversible del recibo y
la trayectoria CTNet queda afectada por el pliegue en M/R antes del readout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from ctnet_infinite_atlas_mthd import InfiniteAtlasMTHD, Receipt, receipt_to_json
from ctnet_omega_cubo6d_plegado_ctnet26 import (
    FoldLayout,
    FoldedCTNetOmegaCubo26,
    FoldedOmegaCuboState,
    count_params,
)


def _canonical_receipt_bytes(receipt: Receipt) -> bytes:
    return json.dumps(receipt_to_json(receipt), sort_keys=True, separators=(",", ":")).encode("utf-8")


def _stream(seed: bytes, n: int) -> bytes:
    out = bytearray()
    i = 0
    while len(out) < n:
        h = hashlib.sha256()
        h.update(len(seed).to_bytes(8, "big"))
        h.update(seed)
        h.update(i.to_bytes(8, "big"))
        out.extend(h.digest())
        i += 1
    return bytes(out[:n])


def _drive_tensor(seed: bytes, shape: Tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    n = 1
    for s in shape:
        n *= int(s)
    raw = _stream(seed, n * 4)
    vals = []
    for i in range(n):
        u = int.from_bytes(raw[4 * i : 4 * i + 4], "big")
        vals.append((u / 4294967295.0) * 2.0 - 1.0)
    x = torch.tensor(vals, device=device, dtype=dtype).reshape(shape)
    eps = torch.finfo(dtype).eps if dtype.is_floating_point else 1e-12
    return x * torch.rsqrt(x.pow(2).mean().clamp_min(eps))


class CTNetOmegaMTHD26(nn.Module):
    """Wrapper CTNet-Omega-Cubo6D con MTHD plegada sobre M y R."""

    def __init__(
        self,
        layout: FoldLayout | None = None,
        *,
        fractal_steps: int = 4,
        latent_steps: int = 2,
        cubo_shear: float = 0.05,
        mthd_seed: str = "ctnet-omega-mthd",
        mthd_omega_words: int = 256,
        mthd_shear: float = 0.0125,
    ):
        super().__init__()
        self.base = FoldedCTNetOmegaCubo26(
            layout=layout or FoldLayout(),
            fractal_steps=fractal_steps,
            latent_steps=latent_steps,
            cubo_shear=cubo_shear,
        )
        self.atlas = InfiniteAtlasMTHD(seed=mthd_seed, omega_words=mthd_omega_words)
        self.mthd_shear = nn.Parameter(torch.tensor(float(mthd_shear), dtype=torch.float32))

    @property
    def layout(self) -> FoldLayout:
        return self.base.layout

    def random_state(self, *args: Any, **kwargs: Any) -> FoldedOmegaCuboState:
        return self.base.random_state(*args, **kwargs)

    def pack(self, state: FoldedOmegaCuboState) -> torch.Tensor:
        return self.base.pack(state)

    def unpack(self, tensor: torch.Tensor) -> FoldedOmegaCuboState:
        return self.base.unpack(tensor)

    def forward_state(self, state: FoldedOmegaCuboState) -> FoldedOmegaCuboState:
        return self.base.forward_state(state)

    def inverse_state(self, state: FoldedOmegaCuboState) -> FoldedOmegaCuboState:
        return self.base.inverse_state(state)

    def _mthd_drives(self, receipt: Receipt, state: FoldedOmegaCuboState) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = _canonical_receipt_bytes(receipt)
        d_m = hashlib.sha256(b"ctnet-mthd-memory" + raw).digest()
        d_r = hashlib.sha256(b"ctnet-mthd-relations" + raw).digest()
        mem_drive = _drive_tensor(d_m, tuple(state.memory.shape), device=state.memory.device, dtype=state.memory.dtype)
        rel_drive = _drive_tensor(d_r, tuple(state.relations.shape), device=state.relations.device, dtype=state.relations.dtype)
        return mem_drive, rel_drive

    def fold_receipt_state(self, state: FoldedOmegaCuboState, receipt: Receipt, *, sign: float = 1.0) -> FoldedOmegaCuboState:
        """Plega/despliega un recibo MTHD dentro de M y R por shear reversible."""
        mem_drive, rel_drive = self._mthd_drives(receipt, state)
        alpha_m = self.mthd_shear.to(device=state.memory.device, dtype=state.memory.dtype)
        alpha_r = self.mthd_shear.to(device=state.relations.device, dtype=state.relations.dtype)
        return FoldedOmegaCuboState(
            z=state.z,
            memory=state.memory + float(sign) * alpha_m * mem_drive,
            relations=state.relations + float(sign) * alpha_r * rel_drive,
            cubo=state.cubo,
            pad=state.pad,
        )

    def put_state(self, state: FoldedOmegaCuboState, key: str, data: bytes) -> Tuple[FoldedOmegaCuboState, Receipt]:
        """Crea recibo MTHD y lo pliega dentro del Omega tensorial."""
        receipt = self.atlas.put(key, data, fold=True)
        folded = self.fold_receipt_state(state, receipt, sign=+1.0)
        return folded, receipt

    def get(self, key: str, receipt: Receipt) -> bytes:
        return self.atlas.get(key, receipt)

    def unfold_state(self, state: FoldedOmegaCuboState, receipt: Receipt) -> FoldedOmegaCuboState:
        return self.fold_receipt_state(state, receipt, sign=-1.0)

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        return self.base.forward(xi)

    def inverse(self, yi: torch.Tensor) -> torch.Tensor:
        return self.base.inverse(yi)

    @torch.no_grad()
    def audit_mthd(
        self,
        *,
        batch: int = 2,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        seed: int = 0,
        steps: int = 1,
    ) -> Dict[str, Any]:
        device = device or next(self.parameters()).device
        self.to(device=device, dtype=dtype)
        state0 = self.random_state(batch=batch, device=device, dtype=dtype, seed=seed)
        folded, receipt = self.put_state(state0, "ctnet/mthd/integrated", b"indice plegado en manifold")
        out = self.get("ctnet/mthd/integrated", receipt)
        un = self.unfold_state(folded, receipt)

        mem_mae = (state0.memory - un.memory).abs().mean()
        rel_mae = (state0.relations - un.relations).abs().mean()

        st = folded
        for _ in range(max(1, int(steps))):
            st = self.forward_state(st)
        rec = st
        for _ in range(max(1, int(steps))):
            rec = self.inverse_state(rec)
        xi0 = self.pack(folded)
        xir = self.pack(rec)
        err = xi0 - xir

        return {
            "mthd_shape": list(self.atlas.shape),
            "has_capacity": False,
            "has_slots": False,
            "has_route_exhaustion": False,
            "mthd_direct_read_ok": out == b"indice plegado en manifold",
            "mthd_fold_memory_mae": float(mem_mae.detach().cpu()),
            "mthd_fold_relations_mae": float(rel_mae.detach().cpu()),
            "mthd_fold_reversible_ok": bool(mem_mae <= 1e-6 and rel_mae <= 1e-6),
            "state_memory_shape_ok": folded.memory.shape == state0.memory.shape,
            "state_relations_shape_ok": folded.relations.shape == state0.relations.shape,
            "ctnet_inverse_steps": int(max(1, steps)),
            "ctnet_inverse_packed_mae": float(err.abs().mean().detach().cpu()),
            "ctnet_inverse_packed_max": float(err.abs().max().detach().cpu()),
            "atlas_audit": self.atlas.audit(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="CTNet-Omega-Cubo6D + InfiniteAtlas-MTHD integrated audit")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--z-tokens", type=int, default=32)
    parser.add_argument("--z-dim", type=int, default=16)
    parser.add_argument("--mem-slots", type=int, default=8)
    parser.add_argument("--mem-dim", type=int, default=16)
    parser.add_argument("--rel-edges", type=int, default=8)
    parser.add_argument("--rel-dim", type=int, default=16)
    parser.add_argument("--fractal-steps", type=int, default=4)
    parser.add_argument("--latent-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mthd-seed", default="ctnet-omega-mthd")
    parser.add_argument("--mthd-omega-words", type=int, default=256)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--fp64", action="store_true")
    args = parser.parse_args()

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
    model = CTNetOmegaMTHD26(
        layout=layout,
        fractal_steps=args.fractal_steps,
        latent_steps=args.latent_steps,
        mthd_seed=args.mthd_seed,
        mthd_omega_words=args.mthd_omega_words,
    ).to(device)
    print("=== CTNet-Omega-Cubo6D + InfiniteAtlas-MTHD ===")
    print(f"device={device} dtype={dtype} params={count_params(model)}")
    audit = model.audit_mthd(batch=args.batch, dtype=dtype, device=device, seed=args.seed, steps=args.steps)
    print(json.dumps(audit, indent=2, default=str))


if __name__ == "__main__":
    main()

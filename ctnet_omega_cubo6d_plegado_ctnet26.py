#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTNet Omega + Cubo 6D plegado dentro de un estado CTNet 2.6.

Idea central
------------
Toda la estructura Omega + Cubo 6D se empaqueta en un único tensor cardinal
CTNet 2.6 de soporte fijo:

    Xi = pack(Z, M, R, C6, pad)  ->  [B, N, d]

Donde:
    Z   : estado cardinal reversible
    M   : memoria topológica interna fija
    R   : banco relacional fijo
    C6  : coordenadas plegadas del Cubo 6D
          [s6, theta15, residual, absorption, omega, closure_score, gates4]
    pad : reserva estructural fija, necesaria para no perder los grados de libertad
          que CTNet 2.6 mezcla fuera de los segmentos semánticos explícitos.

El paso reversible completo es:

    1) shear de Cubo 6D: C6 <- C6 + gamma * CuboObserver(Z,M,R)
       Esta operación es reversible porque Z,M,R no cambian dentro del shear.

    2) CTNet 2.6 reversible sobre Xi completo:
       Xi' = CTNet26(Xi)

La inversa exacta es:

    1) Xi_shear = CTNet26^{-1}(Xi')
    2) C6 <- C6 - gamma * CuboObserver(Z,M,R)

Esto mantiene reversibilidad a nivel máquina en FP32/FP64, dentro del error numérico
normal de PyTorch, sin memoria creciente, sin KV-cache, sin vector-store y sin
proyección destructiva dentro de la ruta inversa.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utilidades CTNet 2.6
# ============================================================


def rmsnorm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class PermutationRouter(nn.Module):
    """Router por permutación. Invertible exactamente salvo error numérico nulo."""

    def __init__(self, d: int, seed: int = 0):
        super().__init__()
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(d, generator=g)
        self.register_buffer("perm", perm.long())
        self.register_buffer("inv_perm", torch.argsort(perm).long())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(-1, self.perm)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(-1, self.inv_perm)


class HyperKernelND(nn.Module):
    """Kernel hiperradial 1D multi-rango usado por CTNet 2.6."""

    def __init__(self, channels: int, radii: Tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.channels = channels
        self.radii = tuple(radii)
        self.ctrl_logits = nn.Parameter(torch.zeros(len(self.radii), 3))
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        assert z.shape[-1] == self.channels
        ctrl = F.softmax(self.ctrl_logits, dim=-1)
        y = torch.zeros_like(z)
        for i, radius in enumerate(self.radii):
            y = y + ctrl[i, 0] * torch.roll(z, -radius, 1)
            y = y + ctrl[i, 1] * z
            y = y + ctrl[i, 2] * torch.roll(z, radius, 1)
        return self.scale.to(dtype=z.dtype) * gelu(rmsnorm(y))


class RevBlockNDBase(nn.Module):
    """Bloque reversible por acoplamiento aditivo.

    F y G no necesitan ser invertibles; el bloque sí lo es por triangularidad.
    """

    def __init__(self, d: int, radii: Tuple[int, ...] = (1, 2, 4), gamma: float = 0.5, seed: int = 777):
        super().__init__()
        if d % 2 != 0:
            raise ValueError("d must be even")
        self.d2 = d // 2
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        self.fH = HyperKernelND(self.d2, radii)
        self.gH = HyperKernelND(self.d2, radii)
        self.router = PermutationRouter(d, seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.router(x)
        x1, x2 = x[..., : self.d2], x[..., self.d2 :]
        g = self.gamma.to(dtype=x.dtype)
        y1 = x1 + g * self.fH(x2)
        y2 = x2 + g * self.gH(y1)
        return torch.cat([y1, y2], dim=-1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y1, y2 = y[..., : self.d2], y[..., self.d2 :]
        g = self.gamma.to(dtype=y.dtype)
        x2 = y2 - g * self.gH(y1)
        x1 = y1 - g * self.fH(x2)
        return self.router.inverse(torch.cat([x1, x2], dim=-1))


class SmallCTNetLatent(nn.Module):
    """Subred CTNet reversible usada como latente fractal."""

    def __init__(self, d_half: int, steps: int = 2):
        super().__init__()
        if d_half % 2 != 0:
            raise ValueError("d_half must be even")
        self.steps = int(steps)
        self.base = RevBlockNDBase(d_half, radii=(1, 2), seed=123)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.steps):
            x = self.base(x)
        return x

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        for _ in range(self.steps):
            y = self.base.inverse(y)
        return y


class CoherenceTensor(nn.Module):
    """Tensor de coherencia diagonal + bajo rango de CTNet 2.6."""

    def __init__(self, d: int, rank: int = 4, beta: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.d = d
        self.rank = rank
        self.beta = beta
        self.eps = eps
        self.metric_diag = nn.Parameter(torch.ones(d))
        self.low_rank = nn.Parameter(torch.randn(d, rank) * 0.1)

    def forward(self, x: torch.Tensor, base_coh: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.shape[-1] != self.d:
            raise ValueError("unexpected coherence dimension")
        xm = x - x.mean(dim=(0, 1), keepdim=True)
        var = xm.pow(2).mean(dim=(0, 1))
        diag = F.softplus(self.metric_diag).to(dtype=x.dtype) + self.eps
        I_diag = (var * diag).sum()
        L = self.low_rank.to(dtype=x.dtype)
        I_low = torch.einsum("bnd,dr->bnr", xm, L).pow(2).mean(dim=(0, 1)).sum()
        I = I_diag + I_low
        speed = torch.exp(self.beta * torch.clamp(I / float(self.d), -5.0, 5.0))
        return speed * base_coh, speed, I


class FractalCTBlock(nn.Module):
    """Bloque fractal CTNet 2.6."""

    def __init__(self, d: int, latent: SmallCTNetLatent, radii: Tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        if d % 4 != 0:
            raise ValueError("d must be a multiple of 4")
        self.d = d
        self.d2 = d // 2
        self.router = PermutationRouter(d, seed=999)
        self.latent = latent
        self.gamma_lat = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))
        self.gamma_main = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.fH = HyperKernelND(self.d2, radii)
        self.gH = HyperKernelND(self.d2, radii)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.router(x)
        u, p = x[..., : self.d2], x[..., self.d2 :]
        u = u + self.gamma_lat.to(dtype=x.dtype) * self.latent(p)
        x = torch.cat([u, p], dim=-1)
        x1, x2 = x[..., : self.d2], x[..., self.d2 :]
        g = self.gamma_main.to(dtype=x.dtype)
        y1 = x1 + g * self.fH(x2)
        y2 = x2 + g * self.gH(y1)
        return torch.cat([y1, y2], dim=-1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y1, y2 = y[..., : self.d2], y[..., self.d2 :]
        g = self.gamma_main.to(dtype=y.dtype)
        p = y2 - g * self.gH(y1)
        u_mix = y1 - g * self.fH(p)
        u = u_mix - self.gamma_lat.to(dtype=y.dtype) * self.latent(p)
        return self.router.inverse(torch.cat([u, p], dim=-1))


class CTNetFractal(nn.Module):
    """CTNet 2.6 fractal reversible."""

    def __init__(self, d: int = 16, N: int = 64, fractal_steps: int = 4, latent_steps: int = 2):
        super().__init__()
        if d % 4 != 0:
            raise ValueError("d must be a multiple of 4")
        self.d = d
        self.N = N
        self.fractal_steps = int(fractal_steps)
        self.latent = SmallCTNetLatent(d // 2, steps=latent_steps)
        self.block = FractalCTBlock(d, self.latent)
        self.coh_tensor = CoherenceTensor(d=d, rank=4, beta=3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        for _ in range(self.fractal_steps):
            z = self.block(z)
        return z

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        z = y
        for _ in range(self.fractal_steps):
            z = self.block.inverse(z)
        return z

    def coherence_energy(self, up: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u, p = up[..., : self.d // 2], up[..., self.d // 2 :]
        u_hat = self.latent(p)
        p_hat = self.latent.inverse(u)
        base_coh = (u - u_hat).pow(2).mean() + (p - p_hat).pow(2).mean()
        return self.coh_tensor(up, base_coh)


class CTNet26(nn.Module):
    """Envoltura CTNet 2.6."""

    def __init__(self, d: int = 16, N: int = 64, fractal_steps: int = 4, latent_steps: int = 2):
        super().__init__()
        self.core = CTNetFractal(d=d, N=N, fractal_steps=fractal_steps, latent_steps=latent_steps)
        self.d = d
        self.N = N

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return self.core.inverse(y)

    def coherence_energy(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.core.coherence_energy(x)


# ============================================================
# Cubo 6D plegable
# ============================================================


AXES_6D: Tuple[str, ...] = ("L", "P", "E", "O", "M", "T")
PLANES_6D: Tuple[Tuple[int, int], ...] = tuple((i, j) for i in range(6) for j in range(i + 1, 6))
CUBO_VECTOR_DIM = 6 + 15 + 4 + 4  # s6 + theta15 + residual/absorption/omega/score + gates4 = 29


def all_tetravalent_states() -> torch.Tensor:
    rows = []
    for l in range(4):
        for p in range(4):
            for e in range(4):
                for o in range(4):
                    for m in range(4):
                        for t in range(4):
                            rows.append((l, p, e, o, m, t))
    return torch.tensor(rows, dtype=torch.float32)


class Cubo6DObserver(nn.Module):
    """Observador de cierre Cubo 6D sin memoria creciente.

    Devuelve un vector fijo de 29 coordenadas que puede sumarse por shear reversible
    al segmento C6 del estado plegado.
    """

    def __init__(self, absorption_strength: float = 0.72, eps: float = 1e-6):
        super().__init__()
        self.absorption_strength = float(absorption_strength)
        self.eps = eps
        self.theta_bias = nn.Parameter(torch.zeros(15))
        self.theta_scale = nn.Parameter(torch.full((15,), 0.15))
        self.register_buffer("states6", all_tetravalent_states(), persistent=False)

    @staticmethod
    def _match_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
        if x.shape[-1] == dim:
            return x
        if x.shape[-1] < dim:
            return F.pad(x, (0, dim - x.shape[-1]))
        return x[..., :dim]

    def modal_features(self, z: torch.Tensor, memory: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] % 2 != 0:
            raise ValueError("z feature dimension must be even")
        u, p = z[..., : z.shape[-1] // 2], z[..., z.shape[-1] // 2 :]
        z_mean = z.mean(dim=1)
        mem_mean = self._match_dim(memory.mean(dim=1), z.shape[-1])
        rel_mean = self._match_dim(relations.mean(dim=1), z.shape[-1])

        limit = z.pow(2).mean(dim=(1, 2)).sqrt()
        partition = (u - p).pow(2).mean(dim=(1, 2)).sqrt()
        presence = memory.pow(2).mean(dim=(1, 2)).sqrt()
        orientation = relations.pow(2).mean(dim=(1, 2)).sqrt()
        mediation = F.cosine_similarity(z_mean, mem_mean + rel_mean, dim=-1).abs()
        incoherence = torch.sigmoid(partition + 0.25 * presence + 0.25 * orientation - mediation)
        closure = 1.0 / (1.0 + incoherence)

        raw = torch.stack([limit, partition, presence, orientation, mediation, closure], dim=-1)
        raw = torch.nan_to_num(raw, nan=0.0, posinf=10.0, neginf=0.0)
        return torch.sigmoid(torch.log1p(raw.clamp_min(0.0)))

    def theta(self, features: torch.Tensor) -> torch.Tensor:
        driver = torch.stack([features[:, i] - features[:, j] for i, j in PLANES_6D], dim=-1)
        return math.pi * torch.tanh(
            self.theta_bias.to(device=features.device, dtype=features.dtype).unsqueeze(0)
            + self.theta_scale.to(device=features.device, dtype=features.dtype).unsqueeze(0) * driver
        )

    def rotate_states(self, theta: torch.Tensor) -> torch.Tensor:
        batch = theta.shape[0]
        states = (self.states6.to(device=theta.device, dtype=theta.dtype) / 3.0) - 0.5
        x = states.unsqueeze(0).expand(batch, -1, -1).clone()
        for k, (i, j) in enumerate(PLANES_6D):
            c = torch.cos(theta[:, k]).unsqueeze(-1)
            s = torch.sin(theta[:, k]).unsqueeze(-1)
            xi = x[..., i].clone()
            xj = x[..., j].clone()
            x[..., i] = xi * c - xj * s
            x[..., j] = xi * s + xj * c
        return x

    def forward(self, z: torch.Tensor, memory: torch.Tensor, relations: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.modal_features(z, memory, relations)
        theta = self.theta(features)
        rotated = self.rotate_states(theta)
        target = features - 0.5
        residuals = (rotated - target.unsqueeze(1)).pow(2).mean(dim=-1).sqrt()
        residual, idx = residuals.min(dim=1)
        state6 = self.states6.to(device=z.device, dtype=z.dtype).index_select(0, idx)
        closure_degree = state6.sum(dim=-1) / 18.0

        # Absorción fija y diferenciable: no enumera memoria ni relaciones.
        absorption = self.absorption_strength * (
            0.22
            + 0.48 * closure_degree
            + 0.15 * (1.0 - features[:, 1].clamp(0.0, 1.0))
            + 0.10 * features[:, 4].clamp(0.0, 1.0)
            + 0.05 * features[:, 5].clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)

        omega = torch.relu(residual - absorption)
        closure_score = torch.exp(-omega)
        gate_base = torch.tanh(closure_score - omega)
        gates = torch.stack(
            [
                (1.0 + 0.18 * gate_base).clamp(0.55, 1.45),
                (1.0 + 0.24 * torch.tanh(absorption - omega)).clamp(0.55, 1.45),
                (1.0 + 0.24 * torch.tanh(closure_degree - omega)).clamp(0.55, 1.45),
                (1.0 + 0.12 * torch.tanh(closure_score - 0.5)).clamp(0.70, 1.30),
            ],
            dim=-1,
        )

        # Vector normalizado para plegado estable dentro del estado CTNet.
        vector = torch.cat(
            [
                state6 / 3.0,
                theta / math.pi,
                residual.unsqueeze(-1),
                absorption.unsqueeze(-1),
                omega.unsqueeze(-1),
                closure_score.unsqueeze(-1),
                gates,
            ],
            dim=-1,
        )
        assert vector.shape[-1] == CUBO_VECTOR_DIM
        return {
            "vector": vector,
            "state6": state6,
            "theta15": theta,
            "residual": residual,
            "absorption": absorption,
            "omega": omega,
            "closure_score": closure_score,
            "gates": gates,
        }


# ============================================================
# Estado plegado y layout fijo
# ============================================================


@dataclass
class FoldedOmegaCuboState:
    z: torch.Tensor          # [B, z_tokens, z_dim]
    memory: torch.Tensor     # [B, mem_slots, mem_dim]
    relations: torch.Tensor  # [B, rel_edges, rel_dim]
    cubo: torch.Tensor       # [B, 29]
    pad: torch.Tensor        # [B, pad_size]


@dataclass(frozen=True)
class FoldLayout:
    N: int = 64
    d: int = 16
    z_tokens: int = 32
    z_dim: int = 16
    mem_slots: int = 8
    mem_dim: int = 16
    rel_edges: int = 8
    rel_dim: int = 16

    @property
    def capacity(self) -> int:
        return self.N * self.d

    @property
    def z_size(self) -> int:
        return self.z_tokens * self.z_dim

    @property
    def memory_size(self) -> int:
        return self.mem_slots * self.mem_dim

    @property
    def relations_size(self) -> int:
        return self.rel_edges * self.rel_dim

    @property
    def cubo_size(self) -> int:
        return CUBO_VECTOR_DIM

    @property
    def semantic_size(self) -> int:
        return self.z_size + self.memory_size + self.relations_size + self.cubo_size

    @property
    def pad_size(self) -> int:
        return self.capacity - self.semantic_size

    def validate(self) -> None:
        if self.d % 4 != 0:
            raise ValueError("CTNet 2.6 folded d must be a multiple of 4")
        if self.z_dim % 2 != 0:
            raise ValueError("z_dim must be even for u/p modal split")
        if self.pad_size < 0:
            raise ValueError(
                f"Folded state does not fit: semantic_size={self.semantic_size}, capacity={self.capacity}. "
                "Increase N or d, or reduce component sizes."
            )

    def pack(self, state: FoldedOmegaCuboState) -> torch.Tensor:
        self.validate()
        batch = state.z.shape[0]
        parts = [
            state.z.reshape(batch, -1),
            state.memory.reshape(batch, -1),
            state.relations.reshape(batch, -1),
            state.cubo.reshape(batch, -1),
            state.pad.reshape(batch, -1),
        ]
        flat = torch.cat(parts, dim=-1)
        if flat.shape[-1] != self.capacity:
            raise ValueError(f"packed size {flat.shape[-1]} != capacity {self.capacity}")
        return flat.reshape(batch, self.N, self.d)

    def unpack(self, tensor: torch.Tensor) -> FoldedOmegaCuboState:
        self.validate()
        if tensor.ndim != 3 or tensor.shape[-2:] != (self.N, self.d):
            raise ValueError(f"expected tensor shape [B,{self.N},{self.d}]")
        batch = tensor.shape[0]
        flat = tensor.reshape(batch, -1)
        i = 0

        z = flat[:, i : i + self.z_size].reshape(batch, self.z_tokens, self.z_dim)
        i += self.z_size

        memory = flat[:, i : i + self.memory_size].reshape(batch, self.mem_slots, self.mem_dim)
        i += self.memory_size

        relations = flat[:, i : i + self.relations_size].reshape(batch, self.rel_edges, self.rel_dim)
        i += self.relations_size

        cubo = flat[:, i : i + self.cubo_size]
        i += self.cubo_size

        pad = flat[:, i : i + self.pad_size]
        i += self.pad_size

        if i != self.capacity:
            raise RuntimeError("layout slicing bug")
        return FoldedOmegaCuboState(z=z, memory=memory, relations=relations, cubo=cubo, pad=pad)


# ============================================================
# CTNet Omega + Cubo 6D plegado dentro de CTNet 2.6
# ============================================================


class FoldedCTNetOmegaCubo26(nn.Module):
    """CTNet Omega + Cubo 6D plegado como un único estado CTNet 2.6.

    La ruta reversible completa preserva todas las coordenadas, incluido el padding
    estructural. No se descartan dimensiones internas tras el forward.
    """

    def __init__(
        self,
        layout: FoldLayout | None = None,
        *,
        fractal_steps: int = 4,
        latent_steps: int = 2,
        cubo_shear: float = 0.05,
    ):
        super().__init__()
        self.layout = layout or FoldLayout()
        self.layout.validate()
        self.core = CTNet26(d=self.layout.d, N=self.layout.N, fractal_steps=fractal_steps, latent_steps=latent_steps)
        self.cubo = Cubo6DObserver()
        self.cubo_shear = nn.Parameter(torch.tensor(float(cubo_shear), dtype=torch.float32))

    def random_state(
        self,
        batch: int = 2,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        seed: int | None = None,
        initialize_cubo_from_observer: bool = True,
    ) -> FoldedOmegaCuboState:
        if seed is not None:
            torch.manual_seed(seed)
        device = device or next(self.parameters()).device
        L = self.layout
        z = torch.randn(batch, L.z_tokens, L.z_dim, device=device, dtype=dtype)
        memory = 0.01 * torch.randn(batch, L.mem_slots, L.mem_dim, device=device, dtype=dtype)
        relations = 0.01 * torch.randn(batch, L.rel_edges, L.rel_dim, device=device, dtype=dtype)
        # La reserva estructural no debe inicializarse como cero puro.
        # Un subespacio muerto puede amplificar error de redondeo en FP32 tras varias
        # capas fractales. Usamos una fibra fija no degenerada de baja amplitud.
        if L.pad_size > 0:
            phase = torch.linspace(0, 2 * math.pi, L.pad_size, device=device, dtype=dtype)
            pad_anchor = 0.01 * (torch.sin(phase) + 0.5 * torch.cos(2.0 * phase))
            pad = pad_anchor.unsqueeze(0).repeat(batch, 1)
        else:
            pad = torch.zeros(batch, 0, device=device, dtype=dtype)
        cubo_vec = torch.zeros(batch, L.cubo_size, device=device, dtype=dtype)
        state = FoldedOmegaCuboState(z=z, memory=memory, relations=relations, cubo=cubo_vec, pad=pad)
        if initialize_cubo_from_observer:
            obs = self.cubo(z, memory, relations)["vector"]
            state = FoldedOmegaCuboState(z=z, memory=memory, relations=relations, cubo=obs, pad=pad)
        return state

    def pack(self, state: FoldedOmegaCuboState) -> torch.Tensor:
        return self.layout.pack(state)

    def unpack(self, tensor: torch.Tensor) -> FoldedOmegaCuboState:
        return self.layout.unpack(tensor)

    def cubo_observation(self, state: FoldedOmegaCuboState) -> Dict[str, torch.Tensor]:
        return self.cubo(state.z, state.memory, state.relations)

    def closure_shear(self, state: FoldedOmegaCuboState, *, sign: float = 1.0) -> FoldedOmegaCuboState:
        """Shear reversible sobre el segmento Cubo.

        C6' = C6 + sign * alpha * CuboObserver(Z,M,R)

        Como Z, M y R no cambian dentro de este shear, la inversa se obtiene con
        sign=-1 y el mismo observador.
        """
        obs = self.cubo_observation(state)["vector"]
        alpha = self.cubo_shear.to(device=state.cubo.device, dtype=state.cubo.dtype)
        return FoldedOmegaCuboState(
            z=state.z,
            memory=state.memory,
            relations=state.relations,
            cubo=state.cubo + float(sign) * alpha * obs,
            pad=state.pad,
        )

    def forward_state(self, state: FoldedOmegaCuboState) -> FoldedOmegaCuboState:
        sheared = self.closure_shear(state, sign=+1.0)
        xi = self.pack(sheared)
        yi = self.core(xi)
        return self.unpack(yi)

    def inverse_state(self, state: FoldedOmegaCuboState) -> FoldedOmegaCuboState:
        yi = self.pack(state)
        pre_shear_xi = self.core.inverse(yi)
        pre_shear = self.unpack(pre_shear_xi)
        return self.closure_shear(pre_shear, sign=-1.0)

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        """Forward tensorial puro: [B,N,d] -> [B,N,d]."""
        state = self.unpack(xi)
        return self.pack(self.forward_state(state))

    def inverse(self, yi: torch.Tensor) -> torch.Tensor:
        """Inversa tensorial pura: [B,N,d] -> [B,N,d]."""
        state = self.unpack(yi)
        return self.pack(self.inverse_state(state))

    @staticmethod
    def state_errors(a: FoldedOmegaCuboState, b: FoldedOmegaCuboState) -> Dict[str, torch.Tensor]:
        return {
            "z_mae": (a.z - b.z).abs().mean(),
            "memory_mae": (a.memory - b.memory).abs().mean(),
            "relations_mae": (a.relations - b.relations).abs().mean(),
            "cubo_mae": (a.cubo - b.cubo).abs().mean(),
            "pad_mae": (a.pad - b.pad).abs().mean() if a.pad.numel() else torch.zeros((), device=a.z.device, dtype=a.z.dtype),
        }

    @torch.no_grad()
    def audit(
        self,
        *,
        batch: int = 2,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        steps: int = 1,
        seed: int = 0,
    ) -> Dict[str, float]:
        device = device or next(self.parameters()).device
        self.to(device=device, dtype=dtype)
        state0 = self.random_state(batch=batch, device=device, dtype=dtype, seed=seed)
        xi0 = self.pack(state0)

        state = state0
        for _ in range(max(1, steps)):
            state = self.forward_state(state)

        recovered = state
        for _ in range(max(1, steps)):
            recovered = self.inverse_state(recovered)

        xir = self.pack(recovered)
        err = xi0 - xir
        component_errors = self.state_errors(state0, recovered)
        obs = self.cubo_observation(state0)

        return {
            "steps": float(max(1, steps)),
            "packed_mae": float(err.abs().mean().detach().cpu()),
            "packed_max": float(err.abs().max().detach().cpu()),
            "packed_rel": float((err.norm() / (xi0.norm() + 1e-12)).detach().cpu()),
            "z_mae": float(component_errors["z_mae"].detach().cpu()),
            "memory_mae": float(component_errors["memory_mae"].detach().cpu()),
            "relations_mae": float(component_errors["relations_mae"].detach().cpu()),
            "cubo_mae": float(component_errors["cubo_mae"].detach().cpu()),
            "pad_mae": float(component_errors["pad_mae"].detach().cpu()),
            "z_shape_ok": float(recovered.z.shape == state0.z.shape),
            "memory_shape_ok": float(recovered.memory.shape == state0.memory.shape),
            "relations_shape_ok": float(recovered.relations.shape == state0.relations.shape),
            "cubo_shape_ok": float(recovered.cubo.shape == state0.cubo.shape),
            "pad_shape_ok": float(recovered.pad.shape == state0.pad.shape),
            "capacity": float(self.layout.capacity),
            "semantic_size": float(self.layout.semantic_size),
            "pad_size": float(self.layout.pad_size),
            "cubo_omega_initial": float(obs["omega"].mean().detach().cpu()),
            "cubo_closure_score_initial": float(obs["closure_score"].mean().detach().cpu()),
        }


# ============================================================
# CLI de auditoría
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="CTNet Omega + Cubo 6D plegado en estado CTNet 2.6")
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
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--fp64", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
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
    model = FoldedCTNetOmegaCubo26(layout=layout, fractal_steps=args.fractal_steps, latent_steps=args.latent_steps).to(device)

    print("=== CTNet Omega + Cubo 6D plegado en CTNet 2.6 ===")
    print(f"device={device} params={count_params(model)} capacity={layout.capacity} semantic_size={layout.semantic_size} pad_size={layout.pad_size}")

    audit32 = model.audit(batch=args.batch, dtype=torch.float32, device=device, steps=args.steps, seed=args.seed)
    print("\n[FP32]")
    for k, v in audit32.items():
        print(f"{k}: {v:.12g}")

    if args.fp64:
        model64 = FoldedCTNetOmegaCubo26(layout=layout, fractal_steps=args.fractal_steps, latent_steps=args.latent_steps).double().to(device)
        audit64 = model64.audit(batch=args.batch, dtype=torch.float64, device=device, steps=args.steps, seed=args.seed)
        print("\n[FP64]")
        for k, v in audit64.items():
            print(f"{k}: {v:.12g}")


if __name__ == "__main__":
    main()

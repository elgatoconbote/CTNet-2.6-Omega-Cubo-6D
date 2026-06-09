#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTNet process observer.

A CTNet system must be able to observe every process involved in its own
response loop. This module provides neutral observation utilities for:

- folded state components: Z, memory, relations, Cubo 6D, pad, Xi,
- transitions between states,
- u=p debt at multiple scales,
- coherence tensor energy,
- Cubo 6D closure values,
- reversibility debt,
- process deltas.

It does not generate text and does not rank candidate text. It observes the
system so that later training can close the full loop:

    input -> internal process -> output chart -> product observation -> correction

The invariant used by every observer is u=p closure.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ctnet_omega_cubo6d_plegado_ctnet26 import FoldedCTNetOmegaCubo26, FoldedOmegaCuboState


@dataclass
class TensorObservation:
    name: str
    mean: float
    std: float
    rms: float
    abs_mean: float
    up: float


@dataclass
class StateObservation:
    z: TensorObservation
    memory: TensorObservation
    relations: TensorObservation
    cubo: TensorObservation
    pad: TensorObservation
    xi: TensorObservation
    coherence_energy: float
    coherence_speed: float
    coherence_info: float
    cubo_residual: float
    cubo_absorption: float
    cubo_omega: float
    cubo_closure_score: float


@dataclass
class TransitionObservation:
    before: StateObservation
    after: StateObservation
    delta_xi: TensorObservation
    delta_z: TensorObservation
    delta_memory: TensorObservation
    delta_relations: TensorObservation
    delta_cubo: TensorObservation
    reversibility_mae: float
    transition_debt: float


def _as_float(x: torch.Tensor) -> float:
    return float(x.detach().to(torch.float32).mean().cpu())


def _even_last_dim(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2 == 0:
        return x
    return F.pad(x, (0, 1))


def up_loss(x: torch.Tensor) -> torch.Tensor:
    x = _even_last_dim(x)
    h = x.shape[-1] // 2
    return F.mse_loss(x[..., :h], x[..., h:])


def _pool_tokens(x: torch.Tensor, scale: int) -> torch.Tensor:
    if x.ndim != 3 or x.shape[1] < scale:
        return x
    b, n, d = x.shape
    usable = (n // scale) * scale
    if usable <= 0:
        return x
    return x[:, :usable, :].reshape(b, usable // scale, scale, d).mean(dim=2)


def multiscale_up_loss(x: torch.Tensor, *, token_scales: Tuple[int, ...] = (2, 4, 8)) -> torch.Tensor:
    terms = [up_loss(x)]
    for shift in (1, 2, 3):
        if x.shape[-1] > shift:
            terms.append(up_loss(torch.roll(x, shifts=shift, dims=-1)))
    if x.ndim == 3:
        for shift in (1, 2, 4):
            if x.shape[1] > shift:
                terms.append(up_loss(torch.roll(x, shifts=shift, dims=1)))
        for scale in token_scales:
            if x.shape[1] >= scale:
                pooled = _pool_tokens(x, scale)
                terms.append(up_loss(pooled))
                if pooled.shape[1] > 1:
                    terms.append(up_loss(torch.roll(pooled, shifts=1, dims=1)))
    return torch.stack(terms).mean()


def observe_tensor(name: str, x: torch.Tensor) -> TensorObservation:
    xf = x.detach().to(torch.float32)
    return TensorObservation(
        name=name,
        mean=_as_float(xf),
        std=_as_float(xf.std(unbiased=False)),
        rms=_as_float(xf.pow(2).mean().sqrt()),
        abs_mean=_as_float(xf.abs().mean()),
        up=_as_float(multiscale_up_loss(x)),
    )


def observe_state(model: FoldedCTNetOmegaCubo26, state: FoldedOmegaCuboState) -> StateObservation:
    xi = model.pack(state)
    coherence_energy, speed, info = model.core.coherence_energy(xi)
    cubo_obs = model.cubo_observation(state)
    return StateObservation(
        z=observe_tensor("z", state.z),
        memory=observe_tensor("memory", state.memory),
        relations=observe_tensor("relations", state.relations),
        cubo=observe_tensor("cubo", state.cubo),
        pad=observe_tensor("pad", state.pad),
        xi=observe_tensor("xi", xi),
        coherence_energy=_as_float(coherence_energy),
        coherence_speed=_as_float(speed),
        coherence_info=_as_float(info),
        cubo_residual=_as_float(cubo_obs["residual"]),
        cubo_absorption=_as_float(cubo_obs["absorption"]),
        cubo_omega=_as_float(cubo_obs["omega"]),
        cubo_closure_score=_as_float(cubo_obs["closure_score"]),
    )


def observe_transition(
    model: FoldedCTNetOmegaCubo26,
    before: FoldedOmegaCuboState,
    after: Optional[FoldedOmegaCuboState] = None,
) -> TransitionObservation:
    if after is None:
        after = model.forward_state(before)

    before_obs = observe_state(model, before)
    after_obs = observe_state(model, after)

    xi_before = model.pack(before)
    xi_after = model.pack(after)
    recovered = model.inverse_state(after)
    reversibility_mae = (model.pack(recovered) - xi_before).abs().mean()

    delta_xi = xi_after - xi_before
    delta_z = after.z - before.z
    delta_memory = after.memory - before.memory
    delta_relations = after.relations - before.relations
    delta_cubo = after.cubo - before.cubo

    transition_debt = (
        multiscale_up_loss(delta_xi)
        + multiscale_up_loss(after.z)
        + multiscale_up_loss(after.memory)
        + multiscale_up_loss(after.relations)
        + multiscale_up_loss(after.cubo)
        + 0.05 * model.core.coherence_energy(xi_after)[0]
        + 0.25 * model.cubo_observation(after)["omega"].mean()
        + 0.10 * reversibility_mae
        - 0.10 * model.cubo_observation(after)["closure_score"].mean()
    )

    return TransitionObservation(
        before=before_obs,
        after=after_obs,
        delta_xi=observe_tensor("delta_xi", delta_xi),
        delta_z=observe_tensor("delta_z", delta_z),
        delta_memory=observe_tensor("delta_memory", delta_memory),
        delta_relations=observe_tensor("delta_relations", delta_relations),
        delta_cubo=observe_tensor("delta_cubo", delta_cubo),
        reversibility_mae=_as_float(reversibility_mae),
        transition_debt=_as_float(transition_debt),
    )


def to_dict(obs) -> Dict:
    return asdict(obs)

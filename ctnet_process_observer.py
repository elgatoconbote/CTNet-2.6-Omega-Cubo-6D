#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTNet contextual process observation.

Observation is not an external report. In CTNet every observation must become
contextual mass again.

This module observes an internal process as a new FoldedOmegaCuboState:

    process:  state_before -> state_after
    observe:  pack(before, after, delta)
    fold:     observation mass [Z, M, R, C6, pad]
    close:    CT coherence tensor + u=p

So the system can learn to observe its own internal processes in the same form
as any other observation.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ctnet_omega_cubo6d_plegado_ctnet26 import FoldedCTNetOmegaCubo26, FoldedOmegaCuboState


def _even_last_dim(x: torch.Tensor) -> torch.Tensor:
    return x if x.shape[-1] % 2 == 0 else F.pad(x, (0, 1))


def _up_mse_last_dim(x: torch.Tensor) -> torch.Tensor:
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
    terms = [_up_mse_last_dim(x)]
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


def _fold_signal(signal: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    """Fold a [B,K] process signal into [B,*shape] without learned projection.

    The fold is deterministic, differentiable, fixed-size and zero-cache. It is a
    chart operation: if the signal is short, repeat it; if long, crop it.
    """
    batch = signal.shape[0]
    need = 1
    for s in shape:
        need *= int(s)
    if signal.shape[-1] < need:
        reps = (need + signal.shape[-1] - 1) // signal.shape[-1]
        signal = signal.repeat(1, reps)
    return torch.tanh(signal[:, :need]).reshape(batch, *shape)


def observe_process_as_mass(
    model: FoldedCTNetOmegaCubo26,
    before: FoldedOmegaCuboState,
    after: FoldedOmegaCuboState,
) -> FoldedOmegaCuboState:
    """Turn an observed internal process into contextual mass.

    This is the key point: the observation is not scalar diagnostics. It becomes
    another CTNet state, so it can be processed by the same tensor of coherence
    and the same u=p criterion.
    """
    L = model.layout
    xi_before = model.pack(before)
    xi_after = model.pack(after)
    delta = xi_after - xi_before

    process_signal = torch.cat(
        [
            xi_before.reshape(xi_before.shape[0], -1),
            xi_after.reshape(xi_after.shape[0], -1),
            delta.reshape(delta.shape[0], -1),
            before.cubo.reshape(before.cubo.shape[0], -1),
            after.cubo.reshape(after.cubo.shape[0], -1),
            (after.cubo - before.cubo).reshape(after.cubo.shape[0], -1),
        ],
        dim=-1,
    )

    z = _fold_signal(process_signal, (L.z_tokens, L.z_dim))
    memory = 0.01 * _fold_signal(torch.roll(process_signal, shifts=17, dims=-1), (L.mem_slots, L.mem_dim))
    relations = 0.01 * _fold_signal(torch.roll(process_signal, shifts=43, dims=-1), (L.rel_edges, L.rel_dim))

    if L.pad_size > 0:
        pad = 0.01 * _fold_signal(torch.roll(process_signal, shifts=71, dims=-1), (L.pad_size,))
    else:
        pad = process_signal.new_zeros(process_signal.shape[0], 0)

    cubo = model.cubo(z, memory, relations)["vector"].to(device=z.device, dtype=z.dtype)
    return FoldedOmegaCuboState(z=z, memory=memory, relations=relations, cubo=cubo, pad=pad)


def observation_mass_loss(
    model: FoldedCTNetOmegaCubo26,
    before: FoldedOmegaCuboState,
    after: FoldedOmegaCuboState,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Differentiable closure loss for process observation-as-mass."""
    obs_state = observe_process_as_mass(model, before, after)
    obs_out = model.forward_state(obs_state)
    xi_obs = model.pack(obs_out)
    coh, _, _ = model.core.coherence_energy(xi_obs)
    cubo_obs = model.cubo_observation(obs_out)
    delta = xi_obs - model.pack(obs_state)

    z_up = multiscale_up_loss(obs_out.z)
    mem_up = multiscale_up_loss(obs_out.memory)
    rel_up = multiscale_up_loss(obs_out.relations)
    cubo_up = multiscale_up_loss(obs_out.cubo)
    xi_up = multiscale_up_loss(xi_obs)
    delta_up = multiscale_up_loss(delta)
    up = torch.stack([z_up, mem_up, rel_up, cubo_up, xi_up, delta_up]).mean()

    omega = cubo_obs["omega"].mean()
    closure = cubo_obs["closure_score"].mean()
    loss = up + 0.05 * coh + 0.25 * omega - 0.10 * closure
    metrics = {
        "obs_mass_loss": float(loss.detach().cpu()),
        "obs_mass_up": float(up.detach().cpu()),
        "obs_mass_coh": float(coh.detach().cpu()),
        "obs_mass_omega": float(omega.detach().cpu()),
        "obs_mass_closure": float(closure.detach().cpu()),
        "obs_mass_z": float(z_up.detach().cpu()),
        "obs_mass_memory": float(mem_up.detach().cpu()),
        "obs_mass_relations": float(rel_up.detach().cpu()),
        "obs_mass_cubo": float(cubo_up.detach().cpu()),
        "obs_mass_xi": float(xi_up.detach().cpu()),
        "obs_mass_delta": float(delta_up.detach().cpu()),
    }
    return loss, metrics

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTNet runtime loop v0.5.

Ciclo operativo persistente para CTNet-Omega-Cubo6D + MTHD:
observacion -> pliegue -> medicion -> simulacion de ciclo completo -> accion
-> reobservacion -> memoria.

v0.2 corrigio la seleccion de acciones: cada candidato se simula hasta su
reobservacion, y no solo como pliegue aislado.

v0.3 introdujo dos reguladores necesarios:
- inhibit: accion nula real. No escribe accion ni reobservacion cuando todos los
  candidatos abren deuda.
- stabilize: accion interna reversible que aplica pasos inversos CTNet tras el
  pliegue de accion antes de reobservar.

v0.4 fija el principio arquitectonico central: la deuda que gobierna accion,
simulacion e inhibicion nace del tensor de coherencia + cierre u/p. Las señales
del Cubo6D quedan como diagnostico, no como ley de decision primaria.

v0.4.1 resuelve de forma robusta el nucleo fractal real: en algunos layouts esta
en base.core, y en otros en base.core.core.

v0.5 introduce consolidate_u_p: un efector interno que convierte la trayectoria
reciente de cierre en una regla MTHD corta. La regla solo gana si su simulacion
reduce closure_debt = coherence_debt + up_debt; si no, gana inhibit.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

from ctnet_infinite_atlas_mthd import InfiniteAtlasMTHD, receipt_to_json
from ctnet_omega_cubo6d_plegado_ctnet26 import FoldedOmegaCuboState
from ctnet_omega_mthd_integrated import CTNetOmegaMTHD26


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_bytes(obj: Any) -> bytes:
    return canonical_json(obj).encode("utf-8")


def digest_obj(obj: Any, n: int = 16) -> str:
    return hashlib.sha256(json_bytes(obj)).hexdigest()[:n]


def tensor_digest(x: torch.Tensor, n: int = 16) -> str:
    cpu = x.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()[:n]


def mean_float(x: torch.Tensor) -> float:
    return float(torch.nan_to_num(x.detach()).mean().cpu())


def safe_float(x: Any) -> float:
    try:
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return 0.0
        return y
    except Exception:
        return 0.0


def log_debt(x: float, scale: float = 10.0) -> float:
    return math.log1p(max(0.0, safe_float(x))) / scale


def clone_state(state: FoldedOmegaCuboState) -> FoldedOmegaCuboState:
    return FoldedOmegaCuboState(
        z=state.z.detach().clone(),
        memory=state.memory.detach().clone(),
        relations=state.relations.detach().clone(),
        cubo=state.cubo.detach().clone(),
        pad=state.pad.detach().clone(),
    )


@dataclass
class Observador:
    x: str
    y: str = ""
    source: str = "external"
    regime: str = "observation"
    ts: float = field(default_factory=time.time)
    meta: Dict[str, Any] = field(default_factory=dict)

    def record(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Probe:
    tick: int

    # Ley de gobierno v0.4+: tensor de coherencia + cierre u/p.
    debt: float
    closure_debt: float
    coherence_debt: float
    up_debt: float
    coherence: float
    up_error: float
    up_forward_mse: float
    up_inverse_mse: float
    speed: float
    info_energy: float

    # Diagnostico Cubo6D: no manda la decision primaria.
    omega: float
    closure_score: float
    absorption: float
    residual: float

    z_digest: str
    memory_digest: str
    relations_digest: str
    cubo_digest: str
    ts: float = field(default_factory=time.time)

    def record(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Action:
    kind: str
    payload: str
    reason: str
    simulated_debt: float = 0.0
    simulated_delta: float = 0.0
    simulated_probe: Optional[Dict[str, Any]] = None

    def record(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class RuntimeConfig:
    root: str = ".ctnet_runtime"
    seed: int = 0
    dtype: str = "float32"
    device: str = "cpu"
    mthd_seed: str = "ctnet-runtime"
    mthd_omega_words: int = 256
    closure_steps: int = 1
    stabilizer_steps: int = 1
    consolidation_every: int = 16
    consolidation_window: int = 8
    max_records: int = 2048
    text_max: int = 480

    def torch_dtype(self) -> torch.dtype:
        return torch.float64 if self.dtype == "float64" else torch.float32

    def torch_device(self) -> torch.device:
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


class CTNetRuntimeLoop:
    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self.root = Path(cfg.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.device = cfg.torch_device()
        self.dtype = cfg.torch_dtype()
        torch.manual_seed(int(cfg.seed))
        self.model = CTNetOmegaMTHD26(mthd_seed=cfg.mthd_seed, mthd_omega_words=cfg.mthd_omega_words).to(device=self.device, dtype=self.dtype)
        self.model.eval()
        self.tick = 0
        self.state: Optional[FoldedOmegaCuboState] = None
        self.records: List[Dict[str, Any]] = []

    @property
    def state_path(self) -> Path:
        return self.root / "omega_state.pt"

    @property
    def atlas_path(self) -> Path:
        return self.root / "mthd_atlas.json"

    @property
    def log_path(self) -> Path:
        return self.root / "runtime.jsonl"

    @property
    def config_path(self) -> Path:
        return self.root / "runtime_config.json"

    def init_or_load(self) -> None:
        self.config_path.write_text(json.dumps(dataclasses.asdict(self.cfg), indent=2, ensure_ascii=False), encoding="utf-8")
        if self.state_path.exists() and self.atlas_path.exists():
            self.load()
        else:
            self.state = self.model.random_state(batch=1, device=self.device, dtype=self.dtype, seed=self.cfg.seed)
            self.save()

    def save(self) -> None:
        if self.state is None:
            raise RuntimeError("state is not initialized")
        torch.save(
            {
                "tick": self.tick,
                "state": {
                    "z": self.state.z.detach().cpu(),
                    "memory": self.state.memory.detach().cpu(),
                    "relations": self.state.relations.detach().cpu(),
                    "cubo": self.state.cubo.detach().cpu(),
                    "pad": self.state.pad.detach().cpu(),
                },
            },
            self.state_path,
        )
        self.model.atlas.save(self.atlas_path)

    def load(self) -> None:
        payload = torch.load(self.state_path, map_location=self.device)
        st = payload["state"]
        self.tick = int(payload.get("tick", 0))
        self.state = FoldedOmegaCuboState(
            z=st["z"].to(device=self.device, dtype=self.dtype),
            memory=st["memory"].to(device=self.device, dtype=self.dtype),
            relations=st["relations"].to(device=self.device, dtype=self.dtype),
            cubo=st["cubo"].to(device=self.device, dtype=self.dtype),
            pad=st["pad"].to(device=self.device, dtype=self.dtype),
        )
        self.model.atlas = InfiniteAtlasMTHD.load(self.atlas_path)
        self.records = self._tail_jsonl(self.log_path, self.cfg.max_records)

    @staticmethod
    def _tail_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        out: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return out

    def _append_jsonl(self, record: Dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def fractal_core(self) -> Any:
        """Devuelve el modulo que contiene latent y coherence_energy."""
        candidate = self.model.base.core
        if hasattr(candidate, "latent") and hasattr(candidate, "coherence_energy"):
            return candidate
        inner = getattr(candidate, "core", None)
        if inner is not None and hasattr(inner, "latent") and hasattr(inner, "coherence_energy"):
            return inner
        raise AttributeError("No se encontro nucleo CTNet fractal con latent y coherence_energy")

    @torch.no_grad()
    def up_metrics(self, xi: torch.Tensor) -> Dict[str, float]:
        """Mide el cierre u/p explicito: u debe reconstruirse desde p y p desde u."""
        core = self.fractal_core()
        d2 = int(core.d) // 2
        u, p = xi[..., :d2], xi[..., d2:]
        u_hat = core.latent(p)
        p_hat = core.latent.inverse(u)
        up_forward_mse = mean_float((u - u_hat).pow(2))
        up_inverse_mse = mean_float((p - p_hat).pow(2))
        up_error = up_forward_mse + up_inverse_mse
        return {
            "up_error": float(up_error),
            "up_forward_mse": float(up_forward_mse),
            "up_inverse_mse": float(up_inverse_mse),
        }

    @torch.no_grad()
    def probe(self, state: Optional[FoldedOmegaCuboState] = None) -> Probe:
        if state is None:
            if self.state is None:
                raise RuntimeError("state is not initialized")
            state = self.state

        # Diagnostico geometrico Cubo6D.
        obs = self.model.base.cubo_observation(state)

        # Ley de gobierno: tensor de coherencia + u/p sobre Xi.
        xi = self.model.pack(state)
        core = self.fractal_core()
        coh, speed, info_energy = core.coherence_energy(xi)
        up = self.up_metrics(xi)

        coherence = safe_float(coh.detach().cpu())
        up_error = up["up_error"]
        coherence_debt = log_debt(coherence)
        up_debt = log_debt(up_error)
        closure_debt = coherence_debt + up_debt

        return Probe(
            tick=self.tick,
            debt=float(closure_debt),
            closure_debt=float(closure_debt),
            coherence_debt=float(coherence_debt),
            up_debt=float(up_debt),
            coherence=float(coherence),
            up_error=float(up_error),
            up_forward_mse=float(up["up_forward_mse"]),
            up_inverse_mse=float(up["up_inverse_mse"]),
            speed=safe_float(speed.detach().mean().cpu()),
            info_energy=safe_float(info_energy.detach().cpu()),
            omega=mean_float(obs["omega"]),
            closure_score=mean_float(obs["closure_score"]),
            absorption=mean_float(obs["absorption"]),
            residual=mean_float(obs["residual"]),
            z_digest=tensor_digest(state.z),
            memory_digest=tensor_digest(state.memory),
            relations_digest=tensor_digest(state.relations),
            cubo_digest=tensor_digest(state.cubo),
        )

    @staticmethod
    def compact_probe(probe: Probe) -> Dict[str, float]:
        return {
            "closure_debt": probe.closure_debt,
            "coherence_debt": probe.coherence_debt,
            "up_debt": probe.up_debt,
            "coherence": probe.coherence,
            "up_error": probe.up_error,
            "up_forward_mse": probe.up_forward_mse,
            "up_inverse_mse": probe.up_inverse_mse,
        }

    @staticmethod
    def probe_debt(record: Dict[str, Any], name: str) -> float:
        probe = record.get(name, {}) or {}
        return safe_float(probe.get("closure_debt", probe.get("debt", 0.0)))

    def recent_closure_trace(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        window = int(limit if limit is not None else self.cfg.consolidation_window)
        out: List[Dict[str, Any]] = []
        for rec in self.records[-max(1, window) :]:
            initial = self.probe_debt(rec, "initial_probe")
            final = self.probe_debt(rec, "final_probe")
            chosen = (rec.get("chosen_action") or {}).get("kind", "unknown")
            out.append(
                {
                    "tick": rec.get("tick"),
                    "chosen": chosen,
                    "initial_closure_debt": initial,
                    "final_closure_debt": final,
                    "delta_debt": safe_float(rec.get("delta_debt", final - initial)),
                    "closed": bool(rec.get("closed_debt", final <= initial)),
                    "governance": rec.get("governance", "legacy"),
                }
            )
        return out

    def consolidation_rule(self, observation: Observador, probe: Probe) -> Dict[str, Any]:
        trace = self.recent_closure_trace(self.cfg.consolidation_window)
        by_action: Dict[str, Dict[str, float]] = {}
        for item in trace:
            kind = str(item.get("chosen", "unknown"))
            slot = by_action.setdefault(kind, {"count": 0.0, "sum_delta": 0.0, "min_delta": float("inf")})
            delta = safe_float(item.get("delta_debt", 0.0))
            slot["count"] += 1.0
            slot["sum_delta"] += delta
            slot["min_delta"] = min(slot["min_delta"], delta)
        for slot in by_action.values():
            count = max(1.0, slot["count"])
            slot["mean_delta"] = slot["sum_delta"] / count
            if math.isinf(slot["min_delta"]):
                slot["min_delta"] = 0.0

        summary = {
            "schema": "ctnet.consolidate_u_p.v1",
            "governance": "coherence_tensor_plus_u_p",
            "tick": self.tick,
            "observation_digest": digest_obj(observation.record(), 24),
            "baseline": self.compact_probe(probe),
            "trace": trace,
            "by_action": by_action,
            "acceptance": "fold only if simulated closure_debt decreases relative to baseline; otherwise inhibit",
        }
        text = canonical_json(summary)
        return {"kind": "repeat", "text": text[:4096], "n": 1}

    @torch.no_grad()
    def fold_record(self, phase: str, record: Dict[str, Any]) -> Dict[str, Any]:
        if self.state is None:
            raise RuntimeError("state is not initialized")
        key = f"runtime/{self.tick:012d}/{phase}/{digest_obj(record, 20)}"
        self.state, receipt = self.model.put_state(self.state, key, json_bytes(record))
        for _ in range(max(1, int(self.cfg.closure_steps))):
            self.state = self.model.forward_state(self.state)
        return {"key": key, "receipt": receipt_to_json(receipt)}

    @torch.no_grad()
    def simulate_fold_record(self, state: FoldedOmegaCuboState, phase: str, record: Dict[str, Any]) -> FoldedOmegaCuboState:
        key = f"sim/{self.tick:012d}/{phase}/{digest_obj(record, 20)}"
        rule = {"kind": "repeat", "text": canonical_json(record)[:512], "n": 1}
        receipt = self.model.atlas.fold_rule(key, rule, fold=False)
        sim_state = self.model.fold_receipt_state(state, receipt, sign=+1.0)
        for _ in range(max(1, int(self.cfg.closure_steps))):
            sim_state = self.model.forward_state(sim_state)
        return sim_state

    @torch.no_grad()
    def simulate_consolidation(self, state: FoldedOmegaCuboState, action: Action) -> FoldedOmegaCuboState:
        payload = json.loads(action.payload)
        key = f"sim/{self.tick:012d}/consolidate_u_p/{payload['rule_digest']}"
        receipt = self.model.atlas.fold_rule(key, payload["rule"], fold=False)
        sim_state = self.model.fold_receipt_state(state, receipt, sign=+1.0)
        for _ in range(max(1, int(self.cfg.closure_steps))):
            sim_state = self.model.forward_state(sim_state)
        return sim_state

    @torch.no_grad()
    def fold_consolidation(self, action: Action) -> Dict[str, Any]:
        if self.state is None:
            raise RuntimeError("state is not initialized")
        payload = json.loads(action.payload)
        key = f"runtime/{self.tick:012d}/consolidate_u_p/{payload['rule_digest']}"
        receipt = self.model.atlas.fold_rule(key, payload["rule"], fold=True)
        self.state = self.model.fold_receipt_state(self.state, receipt, sign=+1.0)
        for _ in range(max(1, int(self.cfg.closure_steps))):
            self.state = self.model.forward_state(self.state)
        return {"key": key, "receipt": receipt_to_json(receipt), "rule_digest": payload["rule_digest"]}

    @torch.no_grad()
    def apply_internal_effect(self, state: FoldedOmegaCuboState, action: Action) -> FoldedOmegaCuboState:
        if action.kind != "stabilize":
            return state
        out = state
        for _ in range(max(1, int(self.cfg.stabilizer_steps))):
            out = self.model.inverse_state(out)
        return out

    def propose(self, observation: Observador, probe: Probe) -> List[Action]:
        text = observation.x.strip().replace("\n", " ")[: self.cfg.text_max]
        consolidation_rule = self.consolidation_rule(observation, probe)
        consolidation_payload = {
            "rule_digest": digest_obj(consolidation_rule, 24),
            "rule": consolidation_rule,
            "window": int(self.cfg.consolidation_window),
        }
        return [
            Action("text", f"closure_debt={probe.closure_debt:.6f} coh={probe.coherence:.6g} up={probe.up_error:.6g} obs={text}", "externalizar cierre desde tensor de coherencia + u/p"),
            Action("memory", f"guardar continuidad obs_hash={digest_obj(observation.record(), 24)} closure_debt={probe.closure_debt:.6f}", "reforzar autobiografia si mejora cierre u/p"),
            Action("self_probe", canonical_json({"closure_debt": probe.closure_debt, "coherence": probe.coherence, "up_error": probe.up_error, "up_forward_mse": probe.up_forward_mse, "up_inverse_mse": probe.up_inverse_mse})[: self.cfg.text_max], "observar tensor de coherencia y cierre u/p"),
            Action("consolidate_u_p", canonical_json(consolidation_payload), "plegar regla MTHD corta si reduce closure_debt"),
            Action("stabilize", f"aplicar {max(1, int(self.cfg.stabilizer_steps))} paso(s) inverso(s) CTNet antes de reobservacion", "regular deriva interna reversible medida por coherencia + u/p"),
            Action("inhibit", "sin accion ni reobservacion", "preservar cierre u/p cuando toda accion abre deuda"),
        ]

    def visible_from_action(self, action: Action) -> str:
        if action.kind == "text":
            return action.payload
        if action.kind == "memory":
            return "internal_memory_written:" + action.payload
        if action.kind == "self_probe":
            return "self_probe:" + action.payload
        if action.kind == "consolidate_u_p":
            payload = json.loads(action.payload)
            return "internal_consolidated_u_p:" + payload["rule_digest"]
        if action.kind == "stabilize":
            return "internal_stabilized:" + action.payload
        if action.kind == "inhibit":
            return "inhibited_action:" + action.payload
        return action.payload

    @torch.no_grad()
    def simulate_full_cycle(self, action: Action, baseline_debt: float) -> Action:
        if self.state is None:
            raise RuntimeError("state is not initialized")
        if action.kind == "inhibit":
            probe = self.probe(self.state)
            action.simulated_debt = probe.debt
            action.simulated_delta = probe.debt - baseline_debt
            action.simulated_probe = probe.record()
            return action
        if action.kind == "consolidate_u_p":
            sim_state = self.simulate_consolidation(clone_state(self.state), action)
            sim_probe = self.probe(sim_state)
            action.simulated_debt = sim_probe.debt
            action.simulated_delta = sim_probe.debt - baseline_debt
            action.simulated_probe = sim_probe.record()
            return action
        sim_state = clone_state(self.state)
        sim_state = self.simulate_fold_record(sim_state, "action", {"tick": self.tick, "phase": "action", "action": action.record()})
        sim_state = self.apply_internal_effect(sim_state, action)
        visible = self.visible_from_action(action)
        reobs = Observador(x=visible, source="ctnet_effector", regime="reobservation", meta={"action_kind": action.kind})
        sim_state = self.simulate_fold_record(sim_state, "reobserve", {"tick": self.tick, "phase": "reobserve", "observation": reobs.record()})
        sim_probe = self.probe(sim_state)
        action.simulated_debt = sim_probe.debt
        action.simulated_delta = sim_probe.debt - baseline_debt
        action.simulated_probe = sim_probe.record()
        return action

    def choose(self, actions: Iterable[Action], baseline_debt: float) -> Action:
        sims = [self.simulate_full_cycle(a, baseline_debt) for a in actions]
        # La eleccion se ordena por delta de closure_debt, que en v0.5 es CoherenceTensor + u/p.
        sims.sort(key=lambda a: (a.simulated_delta, a.simulated_debt, 1 if a.kind == "inhibit" else 0))
        return sims[0]

    @torch.no_grad()
    def step(self, text: str, source: str = "user", regime: str = "external") -> Dict[str, Any]:
        if self.state is None:
            self.init_or_load()
        self.tick += 1
        observation = Observador(x=text, source=source, regime=regime)
        obs_record = {"tick": self.tick, "phase": "observe", "observation": observation.record()}
        obs_receipt = self.fold_record("observe", obs_record)
        probe0 = self.probe()
        actions = self.propose(observation, probe0)
        chosen = self.choose(actions, baseline_debt=probe0.debt)

        action_receipt = None
        reobs_receipt = None
        visible = self.visible_from_action(chosen)
        if chosen.kind == "consolidate_u_p":
            action_receipt = self.fold_consolidation(chosen)
        elif chosen.kind != "inhibit":
            action_receipt = self.fold_record("action", {"tick": self.tick, "phase": "action", "action": chosen.record()})
            self.state = self.apply_internal_effect(self.state, chosen)
            reobs = Observador(x=visible, source="ctnet_effector", regime="reobservation", meta={"action_kind": chosen.kind})
            reobs_receipt = self.fold_record("reobserve", {"tick": self.tick, "phase": "reobserve", "observation": reobs.record()})

        probe1 = self.probe()
        final_delta = probe1.debt - probe0.debt
        event = {
            "tick": self.tick,
            "observation": observation.record(),
            "observe_receipt": obs_receipt,
            "initial_probe": probe0.record(),
            "actions": [a.record() for a in actions],
            "chosen_action": chosen.record(),
            "action_receipt": action_receipt,
            "visible": visible,
            "reobserve_receipt": reobs_receipt,
            "final_probe": probe1.record(),
            "delta_debt": final_delta,
            "closed_debt": final_delta <= 0.0,
            "governance": "coherence_tensor_plus_u_p",
            "consolidation_window": int(self.cfg.consolidation_window),
        }
        self.records.append(event)
        self.records = self.records[-self.cfg.max_records :]
        self._append_jsonl(event)
        self.save()
        return event

    def status(self) -> Dict[str, Any]:
        if self.state is None:
            self.init_or_load()
        probe = self.probe()
        return {
            "root": str(self.root),
            "tick": self.tick,
            "atlas_shape": list(self.model.atlas.shape),
            "records_loaded": len(self.records),
            "governance": "coherence_tensor_plus_u_p",
            "probe": probe.record(),
            "recent_closure_trace": self.recent_closure_trace(self.cfg.consolidation_window),
            "paths": {"state": str(self.state_path), "atlas": str(self.atlas_path), "log": str(self.log_path)},
        }


def make_config(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        root=args.root,
        seed=args.seed,
        dtype="float64" if args.fp64 else "float32",
        device="cuda" if args.cuda else "cpu",
        closure_steps=args.closure_steps,
        stabilizer_steps=args.stabilizer_steps,
        consolidation_every=args.consolidation_every,
        consolidation_window=args.consolidation_window,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="CTNet runtime loop")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q: argparse.ArgumentParser) -> None:
        q.add_argument("--root", default=".ctnet_runtime")
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--cuda", action="store_true")
        q.add_argument("--fp64", action="store_true")
        q.add_argument("--closure-steps", type=int, default=1)
        q.add_argument("--stabilizer-steps", type=int, default=1)
        q.add_argument("--consolidation-every", type=int, default=16)
        q.add_argument("--consolidation-window", type=int, default=8)

    q = sub.add_parser("init")
    common(q)
    q = sub.add_parser("status")
    common(q)
    q = sub.add_parser("step")
    common(q)
    q.add_argument("--observe", required=True)
    q.add_argument("--source", default="user")
    q.add_argument("--regime", default="external")

    args = p.parse_args()
    rt = CTNetRuntimeLoop(make_config(args))
    rt.init_or_load()
    if args.cmd == "init":
        print(json.dumps(rt.status(), indent=2, ensure_ascii=False))
    elif args.cmd == "status":
        print(json.dumps(rt.status(), indent=2, ensure_ascii=False))
    elif args.cmd == "step":
        print(json.dumps(rt.step(args.observe, source=args.source, regime=args.regime), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

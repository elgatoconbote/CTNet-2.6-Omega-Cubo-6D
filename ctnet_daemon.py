#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTNet daemon v0.1.

Proceso persistente minimo para CTNetRuntimeLoop.

No introduce una politica externa de decision. El daemon solo aporta sensorium,
continuidad de ejecucion, cola de observaciones, pulso interno y trazas. Cada
accion real sigue delegada en CTNetRuntimeLoop.step(), cuya gobernanza es:

    closure_debt = coherence_debt + up_debt
    governance = coherence_tensor_plus_u_p

Ruta operativa:
    observe_once -> runtime.step -> MTHD receipt -> closure probe -> event log

Uso basico:
    python3 ctnet_daemon.py once --observe "texto"
    python3 ctnet_daemon.py loop --inbox .ctnet_runtime/inbox.txt --heartbeat

El fichero de inbox es texto UTF-8. Cada linea no vacia se consume como una
observacion externa. Si no hay observaciones y --heartbeat esta activo, el daemon
emite una observacion interna de continuidad con la deuda actual.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ctnet_runtime_loop import CTNetRuntimeLoop, RuntimeConfig, canonical_json, digest_obj


@dataclass
class DaemonConfig:
    root: str = ".ctnet_runtime"
    seed: int = 0
    fp64: bool = False
    cuda: bool = False
    closure_steps: int = 1
    stabilizer_steps: int = 2
    consolidation_every: int = 16
    consolidation_window: int = 8
    inbox: str = ".ctnet_runtime/inbox.txt"
    interval: float = 5.0
    max_steps: int = 0
    heartbeat: bool = False
    bind_self: bool = False


@dataclass
class DaemonObservation:
    text: str
    source: str = "daemon"
    regime: str = "daemon"
    meta: Dict[str, Any] = field(default_factory=dict)


class CTNetDaemon:
    def __init__(self, cfg: DaemonConfig):
        self.cfg = cfg
        self.root = Path(cfg.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.inbox = Path(cfg.inbox)
        self.inbox.parent.mkdir(parents=True, exist_ok=True)
        self.daemon_log = self.root / "daemon.jsonl"
        self.identity_path = self.root / "self_identity.json"
        self.runtime = CTNetRuntimeLoop(self.runtime_config())
        self.runtime.init_or_load()

    def runtime_config(self) -> RuntimeConfig:
        return RuntimeConfig(
            root=self.cfg.root,
            seed=self.cfg.seed,
            dtype="float64" if self.cfg.fp64 else "float32",
            device="cuda" if self.cfg.cuda else "cpu",
            closure_steps=self.cfg.closure_steps,
            stabilizer_steps=self.cfg.stabilizer_steps,
            consolidation_every=self.cfg.consolidation_every,
            consolidation_window=self.cfg.consolidation_window,
        )

    def append_daemon_log(self, record: Dict[str, Any]) -> None:
        with self.daemon_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def load_or_create_identity(self) -> Dict[str, Any]:
        if self.identity_path.exists():
            return json.loads(self.identity_path.read_text(encoding="utf-8"))
        identity = {
            "schema": "ctnet.self_identity.v1",
            "q_self": "runtime/self/root",
            "governance": "coherence_tensor_plus_u_p",
            "continuity_axiom": "preserve closure trajectory across observation, memory, action and self-observation",
            "boundaries": {
                "self": ["state", "memory", "relations", "actions", "closure_trace"],
                "world": ["external_observations", "files", "terminal", "user_events"],
                "unknown": ["unverified_inference", "unclassified_signal"],
            },
            "created_ts": time.time(),
        }
        self.identity_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return identity

    def bind_self_once(self) -> Optional[Dict[str, Any]]:
        identity = self.load_or_create_identity()
        marker = self.root / ".self_bound"
        digest = digest_obj(identity, 24)
        if marker.exists() and marker.read_text(encoding="utf-8").strip() == digest:
            return None
        obs = DaemonObservation(
            text="CTNet vincula carta raíz de identidad operativa y continuidad autobiográfica.",
            source="ctnet_daemon",
            regime="self_identity",
            meta={"identity_digest": digest, "identity": identity},
        )
        event = self.observe(obs)
        marker.write_text(digest, encoding="utf-8")
        return event

    def read_inbox(self) -> List[DaemonObservation]:
        if not self.inbox.exists():
            return []
        lines = self.inbox.read_text(encoding="utf-8").splitlines()
        observations: List[DaemonObservation] = []
        kept: List[str] = []
        for line in lines:
            text = line.strip()
            if not text:
                continue
            observations.append(
                DaemonObservation(
                    text=text,
                    source="inbox",
                    regime="external",
                    meta={"inbox": str(self.inbox)},
                )
            )
        # Consume all non-empty lines. Empty lines are discarded too.
        self.inbox.write_text("\n".join(kept), encoding="utf-8")
        return observations

    def heartbeat_observation(self) -> DaemonObservation:
        status = self.runtime.status()
        probe = status.get("probe", {})
        trace = status.get("recent_closure_trace", [])
        payload = {
            "tick": status.get("tick"),
            "closure_debt": probe.get("closure_debt"),
            "coherence_debt": probe.get("coherence_debt"),
            "up_debt": probe.get("up_debt"),
            "recent_trace": trace[-4:],
        }
        return DaemonObservation(
            text="CTNet emite pulso interno de continuidad y autoobservación: " + canonical_json(payload),
            source="ctnet_daemon",
            regime="heartbeat",
            meta={"heartbeat": payload},
        )

    def observe(self, observation: DaemonObservation) -> Dict[str, Any]:
        event = self.runtime.step(observation.text, source=observation.source, regime=observation.regime)
        summary = {
            "ts": time.time(),
            "observation": asdict(observation),
            "tick": event.get("tick"),
            "governance": event.get("governance"),
            "chosen": (event.get("chosen_action") or {}).get("kind"),
            "delta_debt": event.get("delta_debt"),
            "closed_debt": event.get("closed_debt"),
            "initial_closure_debt": (event.get("initial_probe") or {}).get("closure_debt"),
            "final_closure_debt": (event.get("final_probe") or {}).get("closure_debt"),
        }
        self.append_daemon_log(summary)
        return event

    def once(self, text: str, source: str = "user", regime: str = "external") -> Dict[str, Any]:
        if self.cfg.bind_self:
            self.bind_self_once()
        return self.observe(DaemonObservation(text=text, source=source, regime=regime))

    def loop(self) -> None:
        if self.cfg.bind_self:
            self.bind_self_once()
        steps = 0
        while True:
            observations = self.read_inbox()
            if not observations and self.cfg.heartbeat:
                observations = [self.heartbeat_observation()]
            for obs in observations:
                event = self.observe(obs)
                print(json.dumps({
                    "tick": event.get("tick"),
                    "chosen": (event.get("chosen_action") or {}).get("kind"),
                    "delta_debt": event.get("delta_debt"),
                    "closed_debt": event.get("closed_debt"),
                    "governance": event.get("governance"),
                }, ensure_ascii=False, sort_keys=True))
                steps += 1
                if self.cfg.max_steps and steps >= self.cfg.max_steps:
                    return
            time.sleep(max(0.1, float(self.cfg.interval)))


def add_common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", default=".ctnet_runtime")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fp64", action="store_true")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--closure-steps", type=int, default=1)
    p.add_argument("--stabilizer-steps", type=int, default=2)
    p.add_argument("--consolidation-every", type=int, default=16)
    p.add_argument("--consolidation-window", type=int, default=8)
    p.add_argument("--inbox", default=".ctnet_runtime/inbox.txt")
    p.add_argument("--bind-self", action="store_true")


def cfg_from_args(args: argparse.Namespace) -> DaemonConfig:
    return DaemonConfig(
        root=args.root,
        seed=args.seed,
        fp64=bool(args.fp64),
        cuda=bool(args.cuda),
        closure_steps=args.closure_steps,
        stabilizer_steps=args.stabilizer_steps,
        consolidation_every=args.consolidation_every,
        consolidation_window=args.consolidation_window,
        inbox=args.inbox,
        interval=getattr(args, "interval", 5.0),
        max_steps=getattr(args, "max_steps", 0),
        heartbeat=bool(getattr(args, "heartbeat", False)),
        bind_self=bool(args.bind_self),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CTNet persistent daemon wrapper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_once = sub.add_parser("once")
    add_common_flags(p_once)
    p_once.add_argument("--observe", required=True)
    p_once.add_argument("--source", default="user")
    p_once.add_argument("--regime", default="external")

    p_loop = sub.add_parser("loop")
    add_common_flags(p_loop)
    p_loop.add_argument("--interval", type=float, default=5.0)
    p_loop.add_argument("--max-steps", type=int, default=0)
    p_loop.add_argument("--heartbeat", action="store_true")

    p_status = sub.add_parser("status")
    add_common_flags(p_status)

    args = parser.parse_args()
    daemon = CTNetDaemon(cfg_from_args(args))

    if args.cmd == "once":
        event = daemon.once(args.observe, source=args.source, regime=args.regime)
        print(json.dumps(event, indent=2, ensure_ascii=False))
    elif args.cmd == "loop":
        daemon.loop()
    elif args.cmd == "status":
        print(json.dumps(daemon.runtime.status(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

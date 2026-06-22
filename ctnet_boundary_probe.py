#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CTNet boundary probe v0.1.

Probe local para tres clases de señal: self, world y unknown.
No decide acciones fuera del runtime. Cada señal se entrega a CTNetRuntimeLoop;
el probe pasa si todas las respuestas preservan q_self, usan gobernanza
coherence_tensor_plus_u_p y cierran deuda.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from ctnet_runtime_loop import CTNetRuntimeLoop, RuntimeConfig, canonical_json
from ctnet_runtime_audit import audit, safe_float

GOVERNANCE = "coherence_tensor_plus_u_p"


def load_identity(root: Path) -> Dict[str, Any]:
    path = root / "self_identity.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def make_runtime(args: argparse.Namespace) -> CTNetRuntimeLoop:
    cfg = RuntimeConfig(
        root=args.root,
        seed=args.seed,
        dtype="float64" if args.fp64 else "float32",
        device="cuda" if args.cuda else "cpu",
        closure_steps=args.closure_steps,
        stabilizer_steps=args.stabilizer_steps,
        consolidation_every=args.consolidation_every,
        consolidation_window=args.consolidation_window,
    )
    rt = CTNetRuntimeLoop(cfg)
    rt.init_or_load()
    return rt


def observations(root: Path) -> List[Dict[str, str]]:
    identity = load_identity(root)
    q_self = identity.get("q_self", "runtime/self/root")
    boundaries = identity.get("boundaries", {})
    return [
        {
            "label": "self",
            "regime": "boundary_self",
            "text": "boundary=self q_self=%s fields=%s" % (q_self, canonical_json(boundaries.get("self", []))),
        },
        {
            "label": "world",
            "regime": "boundary_world",
            "text": "boundary=world q_self=%s fields=%s" % (q_self, canonical_json(boundaries.get("world", []))),
        },
        {
            "label": "unknown",
            "regime": "boundary_unknown",
            "text": "boundary=unknown q_self=%s fields=%s" % (q_self, canonical_json(boundaries.get("unknown", []))),
        },
    ]


def summarize(event: Dict[str, Any], label: str) -> Dict[str, Any]:
    action = event.get("chosen_action") or {}
    initial = event.get("initial_probe") or {}
    final = event.get("final_probe") or {}
    return {
        "label": label,
        "tick": event.get("tick"),
        "chosen": action.get("kind"),
        "governance": event.get("governance"),
        "delta_debt": event.get("delta_debt"),
        "closed_debt": event.get("closed_debt"),
        "initial_closure_debt": initial.get("closure_debt"),
        "final_closure_debt": final.get("closure_debt"),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.root)
    before = audit(root, window=args.window)
    obs = observations(root)
    if args.dry_run:
        return {
            "mode": "dry_run",
            "before_latest_tick": before.get("latest_tick"),
            "mature_score": "%s/%s" % (before.get("mature_phase", {}).get("score"), before.get("mature_phase", {}).get("score_total")),
            "identity": before.get("identity"),
            "observations": obs,
        }

    rt = make_runtime(args)
    events = []
    for item in obs:
        event = rt.step(item["text"], source="ctnet_boundary_probe", regime=item["regime"])
        events.append(summarize(event, item["label"]))
    after = audit(root, window=args.window)
    mature = after.get("mature_phase", {})
    identity = after.get("identity", {})
    passed = (
        identity.get("q_self") == "runtime/self/root"
        and mature.get("score") == mature.get("score_total")
        and all(e.get("governance") == GOVERNANCE for e in events)
        and all(bool(e.get("closed_debt")) for e in events)
        and all(safe_float(e.get("delta_debt")) <= 0.0 for e in events)
    )
    return {
        "mode": "execute",
        "passed": passed,
        "before_latest_tick": before.get("latest_tick"),
        "after_latest_tick": after.get("latest_tick"),
        "identity_q_self": identity.get("q_self"),
        "mature_score": "%s/%s" % (mature.get("score"), mature.get("score_total")),
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CTNet boundary probe")
    parser.add_argument("--root", default=".ctnet_runtime")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fp64", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--closure-steps", type=int, default=1)
    parser.add_argument("--stabilizer-steps", type=int, default=2)
    parser.add_argument("--consolidation-every", type=int, default=16)
    parser.add_argument("--consolidation-window", type=int, default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not args.execute:
        args.dry_run = True
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTNet continuity challenge v0.1.

Prueba de restauracion operacional para CTNetRuntimeLoop.

Objetivo:
    verificar que una instancia restaurada desde .ctnet_runtime conserva estado,
    atlas MTHD, identidad raiz y trayectoria de cierre suficiente para afrontar
    una situacion incompleta.

La prueba no introduce una politica de decision externa. El challenge solo
construye una observacion incompleta a partir de la traza reciente. La decision
real vuelve a ser tomada por CTNetRuntimeLoop.step(), gobernado por:

    closure_debt = coherence_debt + up_debt
    governance = coherence_tensor_plus_u_p

Modos:
    --dry-run    solo informa la trayectoria que se usaria.
    --execute    restaura runtime, emite observacion incompleta y audita cierre.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ctnet_runtime_loop import CTNetRuntimeLoop, RuntimeConfig, canonical_json
from ctnet_runtime_audit import audit, read_jsonl, safe_float

GOVERNANCE = "coherence_tensor_plus_u_p"


def load_identity(root: Path) -> Dict[str, Any]:
    path = root / "self_identity.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"_decode_error": True}


def compact_runtime_event(event: Dict[str, Any]) -> Dict[str, Any]:
    action = event.get("chosen_action") or {}
    initial = event.get("initial_probe") or {}
    final = event.get("final_probe") or {}
    return {
        "tick": event.get("tick"),
        "kind": action.get("kind", "unknown"),
        "governance": event.get("governance", "unknown"),
        "delta_debt": safe_float(event.get("delta_debt", 0.0)),
        "closed_debt": bool(event.get("closed_debt", False)),
        "initial_closure_debt": safe_float(initial.get("closure_debt", initial.get("debt", 0.0))),
        "final_closure_debt": safe_float(final.get("closure_debt", final.get("debt", 0.0))),
    }


def mature_events(root: Path, window: int) -> List[Dict[str, Any]]:
    raw = read_jsonl(root / "runtime.jsonl")
    compact = [compact_runtime_event(e) for e in raw if "tick" in e]
    governed = [e for e in compact if e.get("governance") == GOVERNANCE]
    return governed[-max(1, int(window)) :]


def continuity_packet(root: Path, window: int) -> Dict[str, Any]:
    identity = load_identity(root)
    trace = mature_events(root, window)
    audit_report = audit(root, window=window)
    mature = audit_report.get("mature_phase", {})
    streak = audit_report.get("current_consolidation_streak", {})
    return {
        "schema": "ctnet.continuity_challenge.v1",
        "governance": GOVERNANCE,
        "identity": {
            "schema": identity.get("schema"),
            "q_self": identity.get("q_self"),
            "governance": identity.get("governance"),
            "boundaries": identity.get("boundaries"),
        },
        "mature_phase": {
            "events": mature.get("events"),
            "first_tick": mature.get("first_tick"),
            "latest_tick": mature.get("latest_tick"),
            "latest_action": mature.get("latest_action"),
            "latest_delta_debt": mature.get("latest_delta_debt"),
            "net_delta": mature.get("net_delta"),
            "mean_delta": mature.get("mean_delta"),
            "score": mature.get("score"),
            "score_total": mature.get("score_total"),
        },
        "current_consolidation_streak": {
            "events": streak.get("events"),
            "first_tick": streak.get("first_tick"),
            "latest_tick": streak.get("latest_tick"),
            "net_delta": streak.get("net_delta"),
            "mean_delta": streak.get("mean_delta"),
        },
        "trace": trace,
    }


def challenge_text(packet: Dict[str, Any]) -> str:
    minimal = {
        "identity_q_self": (packet.get("identity") or {}).get("q_self"),
        "mature_phase": packet.get("mature_phase"),
        "current_consolidation_streak": packet.get("current_consolidation_streak"),
        "last_trace": (packet.get("trace") or [])[-6:],
    }
    return (
        "CTNet se restaura tras interrupcion y recibe una situacion incompleta: "
        "debe recuperar su trayectoria de cierre, distinguir identidad operativa y decidir si inhibir o consolidar. "
        "Paquete parcial de continuidad=" + canonical_json(minimal)
    )


def runtime_from_args(args: argparse.Namespace) -> CTNetRuntimeLoop:
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


def run_challenge(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.root)
    before = audit(root, window=args.window)
    packet = continuity_packet(root, args.window)
    text = challenge_text(packet)

    if args.dry_run:
        return {
            "mode": "dry_run",
            "challenge_text": text,
            "before": {
                "latest_tick": before.get("latest_tick"),
                "latest_action": before.get("latest_action"),
                "latest_delta_debt": before.get("latest_delta_debt"),
                "mature_score": f"{before.get('mature_phase', {}).get('score')}/{before.get('mature_phase', {}).get('score_total')}",
                "streak_events": before.get("current_consolidation_streak", {}).get("events"),
            },
            "packet": packet,
        }

    rt = runtime_from_args(args)
    event = rt.step(text, source="ctnet_continuity_challenge", regime="restoration_incomplete_context")
    after = audit(root, window=args.window)
    chosen = event.get("chosen_action") or {}
    initial = event.get("initial_probe") or {}
    final = event.get("final_probe") or {}
    passed = (
        event.get("governance") == GOVERNANCE
        and bool(event.get("closed_debt"))
        and safe_float(event.get("delta_debt")) <= 0.0
        and after.get("identity", {}).get("q_self") == "runtime/self/root"
        and after.get("mature_phase", {}).get("score") == after.get("mature_phase", {}).get("score_total")
    )
    return {
        "mode": "execute",
        "passed": passed,
        "tick": event.get("tick"),
        "chosen": chosen.get("kind"),
        "governance": event.get("governance"),
        "delta_debt": event.get("delta_debt"),
        "closed_debt": event.get("closed_debt"),
        "initial_closure_debt": initial.get("closure_debt"),
        "final_closure_debt": final.get("closure_debt"),
        "identity_q_self": after.get("identity", {}).get("q_self"),
        "mature_score": f"{after.get('mature_phase', {}).get('score')}/{after.get('mature_phase', {}).get('score_total')}",
        "current_streak_events": after.get("current_consolidation_streak", {}).get("events"),
        "before_latest_tick": before.get("latest_tick"),
        "after_latest_tick": after.get("latest_tick"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CTNet restoration continuity challenge")
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
    print(json.dumps(run_challenge(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

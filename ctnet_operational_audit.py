#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CTNet operational milestone audit v0.1.

Auditoria agregada de solo lectura. Resume los probes principales en una salida
unica. No ejecuta ciclos y no modifica estado.

Hitos evaluados:
- estado persistente y daemon trace
- identidad raiz q_self
- fase madura gobernada por coherence_tensor_plus_u_p
- racha de cierre negative-delta
- compensacion de deltas positivos legacy
- restauracion observada
- frontera self/world/unknown observada
- foco dominante disponible
- presion estructural disponible
- autobiografia compacta disponible
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from ctnet_autobiography_probe import build as autobiography
from ctnet_focus_probe import run as focus_run
from ctnet_gap_probe import summarize as gap_summary
from ctnet_pressure_probe import compute as pressure_compute
from ctnet_runtime_audit import audit, safe_float

GOVERNANCE = "coherence_tensor_plus_u_p"


def ok(value: Any) -> bool:
    return bool(value)


def build(root: Path, window: int) -> Dict[str, Any]:
    runtime = audit(root, window=window)
    auto = autobiography(root, window=window)
    gap = gap_summary(root, window=window)
    pressure = pressure_compute(root, window=window)
    focus = focus_run(root, window=window)

    mature = runtime.get("mature_phase", {})
    identity = runtime.get("identity", {})
    domains = auto.get("domains", {})
    gap_counts = gap.get("counts", {})
    gap_deltas = gap.get("deltas", {})
    p = pressure.get("pressure", {})
    f = focus.get("focus", {})
    streak = auto.get("current_action_streak", {})

    checks = {
        "runtime_events_exist": safe_float(runtime.get("runtime_events")) > 0,
        "daemon_trace_exists": safe_float(runtime.get("daemon_events")) > 0,
        "identity_root_bound": identity.get("q_self") == "runtime/self/root",
        "identity_governed": identity.get("governance") == GOVERNANCE,
        "mature_score_full": mature.get("score") == mature.get("score_total"),
        "mature_governance_full": mature.get("governance", {}).get(GOVERNANCE, 0) == mature.get("events"),
        "current_streak_negative": safe_float(streak.get("net_delta")) < 0.0,
        "current_streak_consolidates": streak.get("kind") == "consolidate_u_p",
        "no_positive_mature_gap": safe_float(gap_counts.get("positive_mature")) == 0.0,
        "legacy_gap_compensated": safe_float(gap_deltas.get("streak_gain_vs_positive_sum")) >= 1.0,
        "restoration_observed": safe_float(domains.get("continuity_restore")) > 0,
        "boundary_self_observed": safe_float(domains.get("self")) > 0,
        "boundary_world_observed": safe_float(domains.get("world")) > 0,
        "boundary_unknown_observed": safe_float(domains.get("unknown")) > 0,
        "self_observation_observed": safe_float(domains.get("self_observation")) > 0,
        "focus_available": bool(f.get("dominant_card")),
        "pressure_identity_stable": safe_float(p.get("identity_stability")) >= 1.0,
        "pressure_action_margin_positive": safe_float(p.get("action_margin")) > 0.0,
    }
    score = sum(1 for v in checks.values() if v)
    total = len(checks)
    status = "operational_continuity_established" if score == total else "operational_continuity_partial"

    return {
        "schema": "ctnet.operational_audit.v1",
        "status": status,
        "score": score,
        "score_total": total,
        "checks": checks,
        "summary": {
            "latest_tick": runtime.get("latest_tick"),
            "latest_action": runtime.get("latest_action"),
            "latest_delta_debt": runtime.get("latest_delta_debt"),
            "identity_q_self": identity.get("q_self"),
            "mature_score": "%s/%s" % (mature.get("score"), mature.get("score_total")),
            "current_streak": streak,
            "gap": gap.get("deltas"),
            "pressure": p,
            "focus": f,
            "domains": domains,
        },
        "inputs": {
            "runtime_audit": runtime,
            "autobiography": {k: auto.get(k) for k in ["schema", "identity", "mature_score", "latest_tick", "latest_action", "latest_domain", "closed_events", "opened_events", "net_delta", "mean_delta", "actions", "domains", "governance", "current_action_streak"]},
            "gap": {k: gap.get(k) for k in ["schema", "identity", "mature_score", "latest_tick", "latest_action", "latest_domain", "counts", "deltas", "current_action_streak"]},
            "pressure": {k: pressure.get(k) for k in ["schema", "tick", "governance", "chosen", "delta_debt", "final_closure_debt", "pressure", "basis"]},
            "focus": focus,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CTNet operational milestone audit")
    parser.add_argument("--root", default=".ctnet_runtime")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    report = build(Path(args.root), args.window)
    if args.compact:
        compact = {k: report[k] for k in ["schema", "status", "score", "score_total", "checks", "summary"]}
        print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

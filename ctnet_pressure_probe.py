#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CTNet closure pressure probe v0.1.

Diagnostico de solo lectura. Lee runtime.jsonl y calcula:
- deuda final reciente
- alivio de cierre del ultimo tick
- margen entre mejor y segunda accion simulada
- racha actual de consolidate_u_p
- estabilidad de identidad segun auditoria madura

No ejecuta ciclos y no modifica estado.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

from ctnet_runtime_audit import audit, read_jsonl, safe_float


def clamp01(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return max(0.0, min(1.0, x))


def latest_event(root: Path) -> Dict[str, Any]:
    events = [e for e in read_jsonl(root / "runtime.jsonl") if "tick" in e]
    if not events:
        raise SystemExit("No runtime events found")
    return events[-1]


def candidates(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for action in event.get("actions", []) or []:
        rows.append({
            "kind": action.get("kind"),
            "simulated_delta": safe_float(action.get("simulated_delta")),
            "simulated_debt": safe_float(action.get("simulated_debt")),
        })
    rows.sort(key=lambda x: (x["simulated_delta"], x["simulated_debt"]))
    return rows


def compute(root: Path, window: int) -> Dict[str, Any]:
    event = latest_event(root)
    report = audit(root, window=window)
    rows = candidates(event)
    initial = event.get("initial_probe") or {}
    final = event.get("final_probe") or {}
    initial_debt = safe_float(initial.get("closure_debt", initial.get("debt", 0.0)))
    final_debt = safe_float(final.get("closure_debt", final.get("debt", 0.0)))
    delta = safe_float(event.get("delta_debt"))
    best = rows[0] if rows else {}
    second = rows[1] if len(rows) > 1 else best
    best_delta = safe_float(best.get("simulated_delta"))
    second_delta = safe_float(second.get("simulated_delta"))
    margin = second_delta - best_delta
    positive = [r for r in rows if safe_float(r.get("simulated_delta")) > 0.0]
    mature = report.get("mature_phase", {})
    streak = report.get("current_consolidation_streak", {})
    identity = report.get("identity", {})
    score = safe_float(mature.get("score"))
    total = max(1.0, safe_float(mature.get("score_total"), 10.0))

    return {
        "schema": "ctnet.closure_pressure.v1",
        "tick": event.get("tick"),
        "governance": event.get("governance"),
        "law": "closure_debt = coherence_debt + up_debt",
        "chosen": (event.get("chosen_action") or {}).get("kind"),
        "initial_closure_debt": initial_debt,
        "final_closure_debt": final_debt,
        "delta_debt": delta,
        "pressure": {
            "debt_level": clamp01(final_debt / 10.0),
            "relief": clamp01(max(0.0, -delta) / max(1.0, initial_debt)),
            "action_margin": clamp01(max(0.0, margin) / max(1.0, abs(best_delta))),
            "identity_stability": clamp01(score / total) if identity.get("q_self") == "runtime/self/root" else 0.0,
            "opening_candidate_ratio": clamp01(len(positive) / max(1.0, len(rows))),
        },
        "basis": {
            "best_delta": best_delta,
            "second_delta": second_delta,
            "margin": margin,
            "mature_score": "%s/%s" % (mature.get("score"), mature.get("score_total")),
            "streak_events": streak.get("events"),
            "streak_mean_delta": streak.get("mean_delta"),
            "identity_q_self": identity.get("q_self"),
        },
        "ranked_candidates": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CTNet closure pressure diagnostics")
    parser.add_argument("--root", default=".ctnet_runtime")
    parser.add_argument("--window", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(compute(Path(args.root), args.window), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

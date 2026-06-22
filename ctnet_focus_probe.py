#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CTNet focus probe v0.1.

Diagnostico de solo lectura. Estima la carta dominante del ultimo tick usando
observacion, accion elegida, margen de simulacion, presion de cierre y riesgo.
No ejecuta ciclos y no modifica estado.

La atencion aqui no mezcla tokens: resume que carta operativa domina el cierre.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from ctnet_pressure_probe import compute as pressure_compute
from ctnet_runtime_audit import read_jsonl, safe_float


def latest_event(root: Path) -> Dict[str, Any]:
    events = [e for e in read_jsonl(root / "runtime.jsonl") if "tick" in e]
    if not events:
        raise SystemExit("No runtime events found")
    return events[-1]


def obs_card(event: Dict[str, Any]) -> Dict[str, Any]:
    obs = event.get("observation") or {}
    return {
        "source": obs.get("source"),
        "regime": obs.get("regime"),
        "text_preview": str(obs.get("x", ""))[:160],
    }


def classify_focus(event: Dict[str, Any], pressure: Dict[str, Any]) -> Dict[str, Any]:
    chosen = (event.get("chosen_action") or {}).get("kind")
    obs = obs_card(event)
    p = pressure.get("pressure", {})
    basis = pressure.get("basis", {})
    regime = str(obs.get("regime") or "")
    opening_ratio = safe_float(p.get("opening_candidate_ratio"))
    margin = safe_float(basis.get("margin"))
    debt = safe_float(pressure.get("final_closure_debt"))

    if "boundary_self" in regime:
        domain = "self"
    elif "boundary_world" in regime:
        domain = "world"
    elif "boundary_unknown" in regime:
        domain = "unknown"
    elif "restoration" in regime:
        domain = "continuity_restore"
    elif "heartbeat" in regime:
        domain = "self_observation"
    else:
        domain = regime or "external"

    if chosen == "consolidate_u_p":
        operation = "metabolic_consolidation"
    elif chosen == "inhibit":
        operation = "homeostatic_inhibition"
    elif chosen == "stabilize":
        operation = "reversible_stabilization"
    else:
        operation = str(chosen or "unknown")

    if opening_ratio >= 0.5:
        risk = "high_opening_candidate_ratio"
    elif margin <= 1.0e-9:
        risk = "low_action_margin"
    else:
        risk = "controlled"

    return {
        "dominant_card": "%s/%s" % (domain, operation),
        "domain": domain,
        "operation": operation,
        "risk_marker": risk,
        "why": {
            "chosen": chosen,
            "margin_to_second": margin,
            "final_closure_debt": debt,
            "opening_candidate_ratio": opening_ratio,
            "observation_regime": regime,
        },
    }


def run(root: Path, window: int) -> Dict[str, Any]:
    event = latest_event(root)
    pressure = pressure_compute(root, window)
    focus = classify_focus(event, pressure)
    return {
        "schema": "ctnet.focus_probe.v1",
        "tick": event.get("tick"),
        "governance": event.get("governance"),
        "law": "closure_debt = coherence_debt + up_debt",
        "observation": obs_card(event),
        "focus": focus,
        "pressure": pressure.get("pressure"),
        "basis": pressure.get("basis"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CTNet focus diagnostics")
    parser.add_argument("--root", default=".ctnet_runtime")
    parser.add_argument("--window", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(run(Path(args.root), args.window), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

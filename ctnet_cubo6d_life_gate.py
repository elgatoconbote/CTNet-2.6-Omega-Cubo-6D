#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CTNet Cubo6D life gate v0.1.

Compuerta de solo lectura para runtimes que ya usan la ley integrada:

    closure_debt = coherence_debt + up_debt + cubo6d_debt

No ejecuta ciclos y no modifica estado. Lee runtime.jsonl y exige que el ultimo
evento gobernado incluya Cubo6D dentro de los probes, con cierre no creciente.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

GOVERNANCE = "coherence_tensor_plus_u_p_plus_cubo6d"
LAW = "closure_debt = coherence_debt + up_debt + cubo6d_debt"


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def read_events(root: Path) -> List[Dict[str, Any]]:
    path = root / "runtime.jsonl"
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and "tick" in item:
            out.append(item)
    return out


def latest_runtime_event(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    candidates = [e for e in events if isinstance(e.get("initial_probe"), dict) and isinstance(e.get("final_probe"), dict)]
    if not candidates:
        return {}
    return candidates[-1]


def probe_has_cubo6d(probe: Dict[str, Any]) -> bool:
    required = [
        "coherence_debt",
        "up_debt",
        "cubo6d_debt",
        "cubo6d_residual_debt",
        "cubo6d_omega_debt",
        "cubo6d_nonclosure_debt",
        "cubo6d_absorption_debt",
        "closure_debt",
    ]
    return all(k in probe for k in required)


def law_holds(probe: Dict[str, Any], tolerance: float) -> bool:
    expected = safe_float(probe.get("coherence_debt")) + safe_float(probe.get("up_debt")) + safe_float(probe.get("cubo6d_debt"))
    observed = safe_float(probe.get("closure_debt"))
    return abs(expected - observed) <= tolerance


def gate(root: Path, tolerance: float) -> Dict[str, Any]:
    events = read_events(root)
    event = latest_runtime_event(events)
    initial = event.get("initial_probe", {}) or {}
    final = event.get("final_probe", {}) or {}
    checks = {
        "has_runtime_events": bool(events),
        "has_latest_event": bool(event),
        "governance_integrated": event.get("governance") == GOVERNANCE,
        "closure_law_integrated": event.get("closure_law") == LAW,
        "cubo6d_required": event.get("cubo6d_required") is True,
        "initial_probe_has_cubo6d": probe_has_cubo6d(initial),
        "final_probe_has_cubo6d": probe_has_cubo6d(final),
        "initial_law_holds": law_holds(initial, tolerance),
        "final_law_holds": law_holds(final, tolerance),
        "closed_integrated_debt": safe_float(event.get("delta_debt")) <= tolerance,
    }
    passed = all(checks.values())
    failed = [k for k, v in checks.items() if not v]
    return {
        "schema": "ctnet.cubo6d_life_gate.v1",
        "passed": passed,
        "score": sum(1 for v in checks.values() if v),
        "score_total": len(checks),
        "failed_checks": failed,
        "root": str(root),
        "governance": GOVERNANCE,
        "closure_law": LAW,
        "latest_tick": event.get("tick"),
        "latest_action": (event.get("chosen_action") or {}).get("kind"),
        "delta_debt": event.get("delta_debt"),
        "initial_closure_debt": initial.get("closure_debt"),
        "final_closure_debt": final.get("closure_debt"),
        "initial_cubo6d_debt": initial.get("cubo6d_debt"),
        "final_cubo6d_debt": final.get("cubo6d_debt"),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Cubo6D integrated CTNet runtime events")
    parser.add_argument("--root", default=".ctnet_runtime")
    parser.add_argument("--tolerance", type=float, default=1.0e-9)
    args = parser.parse_args()
    report = gate(Path(args.root), args.tolerance)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

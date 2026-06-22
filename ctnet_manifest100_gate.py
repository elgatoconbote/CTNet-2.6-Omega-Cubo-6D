#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from ctnet_cubo6d_life_gate import gate as cubo6d_gate

GOVERNANCE = "coherence_tensor_plus_u_p_plus_cubo6d"
LAW = "closure_debt = coherence_debt + up_debt + cubo6d_debt"

def sf(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
        return item if isinstance(item, dict) else {}
    except Exception:
        return {}

def read_events(root: Path) -> List[Dict[str, Any]]:
    path = root / "runtime.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and isinstance(item.get("initial_probe"), dict) and isinstance(item.get("final_probe"), dict):
            out.append(item)
    return out

def probe_ok(probe: Dict[str, Any]) -> bool:
    required = {
        "z_digest", "memory_digest", "relations_digest", "cubo_digest",
        "closure_debt", "coherence_debt", "up_debt", "cubo6d_debt",
        "omega", "closure_score", "absorption", "residual",
    }
    return required.issubset(set(probe.keys()))

def all_actions_simulated(event: Dict[str, Any]) -> bool:
    actions = [a for a in event.get("actions", []) if isinstance(a, dict)]
    return bool(actions) and all("simulated_delta" in a and isinstance(a.get("simulated_probe"), dict) for a in actions)

def chosen_ranked_first(event: Dict[str, Any]) -> bool:
    actions = [a for a in event.get("actions", []) if isinstance(a, dict)]
    chosen = event.get("chosen_action") or {}
    if not actions or not chosen.get("kind"):
        return False
    improvers = [a for a in actions if a.get("kind") != "inhibit" and sf(a.get("simulated_delta")) < 0.0]
    if improvers:
        best = sorted(improvers, key=lambda a: (sf(a.get("simulated_delta")), sf(a.get("simulated_debt"))))[0]
        return best.get("kind") == chosen.get("kind")
    return chosen.get("kind") == "inhibit"

def manifest_gate(root: Path, min_events: int, tolerance: float) -> Dict[str, Any]:
    events = read_events(root)
    event = events[-1] if events else {}
    initial = event.get("initial_probe") or {}
    final = event.get("final_probe") or {}
    actions = [a for a in event.get("actions", []) if isinstance(a, dict)]
    kinds = {a.get("kind") for a in actions}
    chosen = event.get("chosen_action") or {}
    identity = read_json(root / "self_identity.json")
    cubo = cubo6d_gate(root, tolerance)

    expected_effectors = {"text", "memory", "self_probe", "consolidate_u_p", "stabilize", "inhibit"}

    checks = {
        "cubo6d_integrated_gate_10_10": cubo.get("passed") is True and cubo.get("score") == cubo.get("score_total"),
        "runtime_artifacts_present": (root / "omega_state.pt").exists() and (root / "mthd_atlas.json").exists() and (root / "runtime.jsonl").exists(),
        "autobiographical_events_enough": len(events) >= min_events,
        "daemon_or_runtime_trace_present": (root / "daemon.jsonl").exists() or len(events) >= min_events,
        "latest_event_has_observe_receipt": isinstance(event.get("observe_receipt"), dict),
        "latest_event_has_action_receipt": isinstance(event.get("action_receipt"), dict),
        "governance_integrated": event.get("governance") == GOVERNANCE,
        "closure_law_integrated": event.get("closure_law") == LAW,
        "cubo6d_required": event.get("cubo6d_required") is True,
        "initial_probe_proprioceptive": probe_ok(initial),
        "final_probe_proprioceptive": probe_ok(final),
        "integrated_debt_closed": sf(event.get("delta_debt")) <= tolerance and sf(final.get("closure_debt")) <= sf(initial.get("closure_debt")) + tolerance,
        "candidate_effectors_present": expected_effectors.issubset(kinds),
        "actions_simulated_before_choice": all_actions_simulated(event),
        "dominant_action_selected_by_closure": chosen_ranked_first(event),
        "internal_consolidation_or_rest_present": chosen.get("kind") == "consolidate_u_p" or any((e.get("chosen_action") or {}).get("kind") == "consolidate_u_p" for e in events),
        "language_is_effector_not_center": "text" in kinds and chosen.get("kind") != "text",
        "identity_file_present": bool(identity),
        "identity_q_self_present": bool(identity.get("q_self") or identity.get("identity_q_self") or identity.get("root") or identity.get("key")),
        "mthd_receipts_present": any(isinstance(e.get("observe_receipt"), dict) for e in events) and any(isinstance(e.get("action_receipt"), dict) for e in events),
    }

    failed = [k for k, v in checks.items() if not v]
    score = sum(1 for v in checks.values() if v)
    total = len(checks)

    return {
        "schema": "ctnet.manifest100_gate.v1",
        "passed": score == total,
        "score": score,
        "score_total": total,
        "percent": round(100.0 * score / max(1, total), 2),
        "failed_checks": failed,
        "root": str(root),
        "latest_tick": event.get("tick"),
        "latest_action": chosen.get("kind"),
        "latest_delta_debt": event.get("delta_debt"),
        "governance": event.get("governance"),
        "closure_law": event.get("closure_law"),
        "initial_closure_debt": initial.get("closure_debt"),
        "final_closure_debt": final.get("closure_debt"),
        "initial_cubo6d_debt": initial.get("cubo6d_debt"),
        "final_cubo6d_debt": final.get("cubo6d_debt"),
        "candidate_effectors": sorted([str(x) for x in kinds]),
        "checks": checks,
        "cubo6d_life_gate": {
            "passed": cubo.get("passed"),
            "score": cubo.get("score"),
            "score_total": cubo.get("score_total"),
            "failed_checks": cubo.get("failed_checks"),
        },
    }

def main() -> None:
    p = argparse.ArgumentParser(description="CTNet manifest 100 operational gate")
    p.add_argument("--root", default=".ctnet_runtime")
    p.add_argument("--min-events", type=int, default=2)
    p.add_argument("--tolerance", type=float, default=1.0e-9)
    args = p.parse_args()
    report = manifest_gate(Path(args.root), args.min_events, args.tolerance)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)

if __name__ == "__main__":
    main()

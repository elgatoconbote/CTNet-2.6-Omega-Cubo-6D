#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTNet runtime continuity audit v0.2.

Audita una instancia local de CTNetRuntimeLoop/CTNetDaemon sin ejecutar nuevos
ciclos. Lee runtime.jsonl, daemon.jsonl y self_identity.json, y produce un
informe compacto sobre continuidad operativa.

Criterios auditados:
- continuidad de ticks
- gobernanza por coherence_tensor_plus_u_p
- tendencia de closure_debt
- acciones cerradas frente a acciones que abren deuda
- consolidaciones metabolicas con delta negativo
- inhibiciones homeostaticas con delta cero
- existencia de carta raiz de identidad
- supervivencia de la traza en ficheros persistentes

v0.2 separa dos lecturas:
- historical_window: ventana solicitada, puede contener eventos legacy.
- mature_phase: tramo desde el primer evento gobernado por coherence_tensor_plus_u_p.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

GOVERNANCE = "coherence_tensor_plus_u_p"
EPS = 1.0e-12


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        if y != y or y in (float("inf"), float("-inf")):
            return default
        return y
    except Exception:
        return default


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            out.append({"_decode_error": True, "raw": line[:240]})
    return out


def probe_debt(event: Dict[str, Any], name: str) -> Optional[float]:
    probe = event.get(name)
    if not isinstance(probe, dict):
        return None
    if "closure_debt" in probe:
        return safe_float(probe.get("closure_debt"))
    if "debt" in probe:
        return safe_float(probe.get("debt"))
    return None


def action_kind(event: Dict[str, Any]) -> str:
    action = event.get("chosen_action")
    if isinstance(action, dict):
        return str(action.get("kind", "unknown"))
    return str(event.get("chosen", "unknown"))


def event_delta(event: Dict[str, Any]) -> float:
    if "delta_debt" in event:
        return safe_float(event.get("delta_debt"))
    initial = probe_debt(event, "initial_probe")
    final = probe_debt(event, "final_probe")
    if initial is not None and final is not None:
        return final - initial
    return 0.0


def tick(event: Dict[str, Any]) -> Optional[int]:
    try:
        return int(event.get("tick"))
    except Exception:
        return None


def compact_event(event: Dict[str, Any]) -> Dict[str, Any]:
    initial = probe_debt(event, "initial_probe")
    final = probe_debt(event, "final_probe")
    return {
        "tick": tick(event),
        "kind": action_kind(event),
        "delta_debt": event_delta(event),
        "initial_closure_debt": initial,
        "final_closure_debt": final,
        "closed_debt": bool(event.get("closed_debt", event_delta(event) <= 0.0)),
        "governance": event.get("governance", "unknown"),
    }


def count_by(items: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in items:
        value = str(item.get(key, "unknown"))
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def monotonic_ticks(items: List[Dict[str, Any]]) -> bool:
    ticks = [x.get("tick") for x in items if isinstance(x.get("tick"), int)]
    return all(b > a for a, b in zip(ticks, ticks[1:]))


def current_streak(events: List[Dict[str, Any]], kind: Optional[str] = None, governance: Optional[str] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for event in reversed(events):
        if kind is not None and event.get("kind") != kind:
            break
        if governance is not None and event.get("governance") != governance:
            break
        out.append(event)
    return list(reversed(out))


def first_governed_phase(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for i, event in enumerate(events):
        if event.get("governance") == GOVERNANCE:
            return events[i:]
    return []


def phase_stats(events: List[Dict[str, Any]], *, label: str) -> Dict[str, Any]:
    deltas = [safe_float(e.get("delta_debt")) for e in events]
    negative = [d for d in deltas if d < -EPS]
    zero = [d for d in deltas if abs(d) <= EPS]
    positive = [d for d in deltas if d > EPS]
    first_debt = events[0].get("initial_closure_debt") if events else None
    last_debt = events[-1].get("final_closure_debt") if events else None
    net_delta = None
    if isinstance(first_debt, (int, float)) and isinstance(last_debt, (int, float)):
        net_delta = safe_float(last_debt) - safe_float(first_debt)

    return {
        "label": label,
        "events": len(events),
        "first_tick": events[0].get("tick") if events else None,
        "latest_tick": events[-1].get("tick") if events else None,
        "latest_action": events[-1].get("kind") if events else None,
        "latest_delta_debt": events[-1].get("delta_debt") if events else None,
        "net_delta": net_delta,
        "mean_delta": statistics.mean(deltas) if deltas else None,
        "min_delta": min(deltas) if deltas else None,
        "max_delta": max(deltas) if deltas else None,
        "negative_steps": len(negative),
        "zero_steps": len(zero),
        "positive_steps": len(positive),
        "actions": count_by(events, "kind"),
        "governance": count_by(events, "governance"),
        "tail": events,
    }


def score_phase(events: List[Dict[str, Any]], identity: Optional[Dict[str, Any]], daemon_events: List[Dict[str, Any]], state_path: Path, atlas_path: Path) -> Dict[str, Any]:
    consolidations = [e for e in events if e.get("kind") == "consolidate_u_p"]
    inhibitions = [e for e in events if e.get("kind") == "inhibit"]
    negative = [e for e in events if safe_float(e.get("delta_debt")) < -EPS]
    score_parts = {
        "has_runtime_log": bool(events),
        "ticks_monotonic": monotonic_ticks(events),
        "governed_phase": bool(events) and all(e.get("governance") == GOVERNANCE for e in events),
        "has_negative_closure": bool(negative),
        "has_homeostatic_inhibition": bool(inhibitions) and any(abs(safe_float(e.get("delta_debt"))) <= EPS for e in inhibitions),
        "has_metabolic_consolidation": bool(consolidations) and all(safe_float(e.get("delta_debt")) < -EPS for e in consolidations[-min(3, len(consolidations)) :]),
        "identity_bound": bool(identity and identity.get("schema") == "ctnet.self_identity.v1"),
        "state_persisted": state_path.exists(),
        "atlas_persisted": atlas_path.exists(),
        "daemon_trace_exists": bool(daemon_events),
    }
    return {
        "score": sum(1 for v in score_parts.values() if v),
        "score_total": len(score_parts),
        "score_parts": score_parts,
    }


def audit(root: Path, window: int = 16) -> Dict[str, Any]:
    runtime_log = root / "runtime.jsonl"
    daemon_log = root / "daemon.jsonl"
    identity_path = root / "self_identity.json"
    atlas_path = root / "mthd_atlas.json"
    state_path = root / "omega_state.pt"

    events_raw = read_jsonl(runtime_log)
    events = [compact_event(e) for e in events_raw if "tick" in e]
    historical_tail = events[-max(1, int(window)) :]
    mature_phase_all = first_governed_phase(events)
    mature_tail = mature_phase_all[-max(1, int(window)) :]
    consolidation_streak = current_streak(events, kind="consolidate_u_p", governance=GOVERNANCE)

    identity = None
    if identity_path.exists():
        try:
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            identity = {"_decode_error": True}

    daemon_events = read_jsonl(daemon_log)
    historical_score = score_phase(historical_tail, identity, daemon_events, state_path, atlas_path)
    mature_score = score_phase(mature_tail, identity, daemon_events, state_path, atlas_path)

    latest = events[-1] if events else {}
    return {
        "root": str(root),
        "runtime_events": len(events),
        "daemon_events": len(daemon_events),
        "latest_tick": latest.get("tick"),
        "latest_action": latest.get("kind"),
        "latest_delta_debt": latest.get("delta_debt"),
        "identity": {
            "exists": identity is not None,
            "schema": identity.get("schema") if isinstance(identity, dict) else None,
            "q_self": identity.get("q_self") if isinstance(identity, dict) else None,
            "governance": identity.get("governance") if isinstance(identity, dict) else None,
        },
        "historical_window": {
            **phase_stats(historical_tail, label="historical_window"),
            **historical_score,
        },
        "mature_phase": {
            **phase_stats(mature_tail, label="mature_phase"),
            **mature_score,
            "all_mature_events": len(mature_phase_all),
        },
        "current_consolidation_streak": phase_stats(consolidation_streak, label="current_consolidation_streak"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CTNet runtime continuity")
    parser.add_argument("--root", default=".ctnet_runtime")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--mature", action="store_true", help="compact output focused on the governed mature phase")
    args = parser.parse_args()

    report = audit(Path(args.root), window=args.window)
    if args.compact:
        phase = report["mature_phase"] if args.mature else report["historical_window"]
        compact = {
            "phase": phase["label"],
            "latest_tick": report["latest_tick"],
            "latest_action": report["latest_action"],
            "latest_delta_debt": report["latest_delta_debt"],
            "phase_net_delta": phase["net_delta"],
            "phase_mean_delta": phase["mean_delta"],
            "score": f"{phase['score']}/{phase['score_total']}",
            "actions": phase["actions"],
            "governance": phase["governance"],
            "identity": report["identity"],
            "current_consolidation_streak": {
                "events": report["current_consolidation_streak"]["events"],
                "first_tick": report["current_consolidation_streak"]["first_tick"],
                "latest_tick": report["current_consolidation_streak"]["latest_tick"],
                "net_delta": report["current_consolidation_streak"]["net_delta"],
                "mean_delta": report["current_consolidation_streak"]["mean_delta"],
            },
        }
        print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

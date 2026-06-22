#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTNet runtime continuity audit v0.1.

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
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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


def audit(root: Path, window: int = 16) -> Dict[str, Any]:
    runtime_log = root / "runtime.jsonl"
    daemon_log = root / "daemon.jsonl"
    identity_path = root / "self_identity.json"
    atlas_path = root / "mthd_atlas.json"
    state_path = root / "omega_state.pt"

    events_raw = read_jsonl(runtime_log)
    events = [compact_event(e) for e in events_raw if "tick" in e]
    tail = events[-max(1, int(window)) :]
    deltas = [safe_float(e.get("delta_debt")) for e in tail]
    negative = [d for d in deltas if d < 0]
    zero = [d for d in deltas if abs(d) <= 1.0e-12]
    positive = [d for d in deltas if d > 1.0e-12]
    consolidations = [e for e in tail if e.get("kind") == "consolidate_u_p"]
    inhibitions = [e for e in tail if e.get("kind") == "inhibit"]
    governed = [e for e in tail if e.get("governance") == "coherence_tensor_plus_u_p"]

    first_debt = tail[0].get("initial_closure_debt") if tail else None
    last_debt = tail[-1].get("final_closure_debt") if tail else None
    net_delta = None
    if isinstance(first_debt, (int, float)) and isinstance(last_debt, (int, float)):
        net_delta = safe_float(last_debt) - safe_float(first_debt)

    identity = None
    if identity_path.exists():
        try:
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            identity = {"_decode_error": True}

    daemon_events = read_jsonl(daemon_log)

    score_parts = {
        "has_runtime_log": bool(events),
        "ticks_monotonic": monotonic_ticks(events),
        "governed_tail": bool(tail) and len(governed) == len(tail),
        "has_negative_closure": bool(negative),
        "has_homeostatic_inhibition": bool(inhibitions) and any(abs(safe_float(e.get("delta_debt"))) <= 1.0e-12 for e in inhibitions),
        "has_metabolic_consolidation": bool(consolidations) and all(safe_float(e.get("delta_debt")) < 0 for e in consolidations[-min(3, len(consolidations)) :]),
        "identity_bound": bool(identity and identity.get("schema") == "ctnet.self_identity.v1"),
        "state_persisted": state_path.exists(),
        "atlas_persisted": atlas_path.exists(),
        "daemon_trace_exists": bool(daemon_events),
    }
    score = sum(1 for v in score_parts.values() if v)

    return {
        "root": str(root),
        "runtime_events": len(events),
        "daemon_events": len(daemon_events),
        "window": len(tail),
        "latest_tick": tail[-1].get("tick") if tail else None,
        "latest_action": tail[-1].get("kind") if tail else None,
        "latest_delta_debt": tail[-1].get("delta_debt") if tail else None,
        "net_window_delta": net_delta,
        "mean_window_delta": statistics.mean(deltas) if deltas else None,
        "min_window_delta": min(deltas) if deltas else None,
        "max_window_delta": max(deltas) if deltas else None,
        "negative_steps": len(negative),
        "zero_steps": len(zero),
        "positive_steps": len(positive),
        "actions": count_by(tail, "kind"),
        "governance": count_by(tail, "governance"),
        "score": score,
        "score_total": len(score_parts),
        "score_parts": score_parts,
        "identity": {
            "exists": identity is not None,
            "schema": identity.get("schema") if isinstance(identity, dict) else None,
            "q_self": identity.get("q_self") if isinstance(identity, dict) else None,
            "governance": identity.get("governance") if isinstance(identity, dict) else None,
        },
        "tail": tail,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CTNet runtime continuity")
    parser.add_argument("--root", default=".ctnet_runtime")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    report = audit(Path(args.root), window=args.window)
    if args.compact:
        compact = {
            "latest_tick": report["latest_tick"],
            "latest_action": report["latest_action"],
            "latest_delta_debt": report["latest_delta_debt"],
            "net_window_delta": report["net_window_delta"],
            "score": f"{report['score']}/{report['score_total']}",
            "actions": report["actions"],
            "governance": report["governance"],
            "identity": report["identity"],
        }
        print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

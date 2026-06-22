#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CTNet autobiography probe v0.1.

Diagnostico de solo lectura. Construye un indice autobiografico compacto a
partir de runtime.jsonl, daemon.jsonl y self_identity.json.

El indice no gobierna acciones. Resume trayectoria propia:
- que observo
- que accion eligio
- si cerro deuda
- que dominio estaba activo
- como evoluciono closure_debt
- que racha operativa esta viva
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ctnet_runtime_audit import audit, read_jsonl, safe_float

GOVERNANCE = "coherence_tensor_plus_u_p"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def action_kind(event: Dict[str, Any]) -> str:
    action = event.get("chosen_action") or {}
    return str(action.get("kind", event.get("chosen", "unknown")))


def probe_debt(event: Dict[str, Any], name: str) -> float:
    probe = event.get(name) or {}
    return safe_float(probe.get("closure_debt", probe.get("debt", 0.0)))


def observation_card(event: Dict[str, Any]) -> Dict[str, Any]:
    obs = event.get("observation") or {}
    text = str(obs.get("x", ""))
    return {
        "source": obs.get("source"),
        "regime": obs.get("regime"),
        "preview": text[:120],
    }


def domain_from_regime(regime: Any) -> str:
    r = str(regime or "")
    if "boundary_self" in r or "self" in r:
        return "self"
    if "boundary_world" in r or "world" in r:
        return "world"
    if "boundary_unknown" in r or "unknown" in r:
        return "unknown"
    if "heartbeat" in r:
        return "self_observation"
    if "restoration" in r:
        return "continuity_restore"
    if "external" in r:
        return "external"
    return r or "unspecified"


def episode(event: Dict[str, Any]) -> Dict[str, Any]:
    obs = observation_card(event)
    initial = probe_debt(event, "initial_probe")
    final = probe_debt(event, "final_probe")
    delta = safe_float(event.get("delta_debt", final - initial))
    return {
        "tick": event.get("tick"),
        "domain": domain_from_regime(obs.get("regime")),
        "source": obs.get("source"),
        "regime": obs.get("regime"),
        "chosen": action_kind(event),
        "governance": event.get("governance", "unknown"),
        "initial_closure_debt": initial,
        "final_closure_debt": final,
        "delta_debt": delta,
        "closed_debt": bool(event.get("closed_debt", delta <= 0.0)),
        "preview": obs.get("preview"),
    }


def count_by(items: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in items:
        k = str(item.get(key, "unknown"))
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def current_action_streak(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {"events": 0}
    kind = items[-1].get("chosen")
    out = []
    for item in reversed(items):
        if item.get("chosen") != kind:
            break
        out.append(item)
    out = list(reversed(out))
    return {
        "kind": kind,
        "events": len(out),
        "first_tick": out[0].get("tick"),
        "latest_tick": out[-1].get("tick"),
        "net_delta": sum(safe_float(e.get("delta_debt")) for e in out),
        "mean_delta": sum(safe_float(e.get("delta_debt")) for e in out) / max(1, len(out)),
    }


def build(root: Path, window: int) -> Dict[str, Any]:
    raw = [e for e in read_jsonl(root / "runtime.jsonl") if "tick" in e]
    episodes = [episode(e) for e in raw]
    tail = episodes[-max(1, int(window)) :]
    identity = load_json(root / "self_identity.json")
    report = audit(root, window=window)
    mature = report.get("mature_phase", {})
    deltas = [safe_float(e.get("delta_debt")) for e in tail]
    closed = [e for e in tail if bool(e.get("closed_debt"))]
    opened = [e for e in tail if not bool(e.get("closed_debt"))]
    return {
        "schema": "ctnet.autobiography_probe.v1",
        "root": str(root),
        "window": len(tail),
        "identity": {
            "schema": identity.get("schema"),
            "q_self": identity.get("q_self"),
            "governance": identity.get("governance"),
        },
        "mature_score": "%s/%s" % (mature.get("score"), mature.get("score_total")),
        "latest_tick": tail[-1].get("tick") if tail else None,
        "latest_action": tail[-1].get("chosen") if tail else None,
        "latest_domain": tail[-1].get("domain") if tail else None,
        "closed_events": len(closed),
        "opened_events": len(opened),
        "net_delta": sum(deltas),
        "mean_delta": sum(deltas) / max(1, len(deltas)),
        "actions": count_by(tail, "chosen"),
        "domains": count_by(tail, "domain"),
        "governance": count_by(tail, "governance"),
        "current_action_streak": current_action_streak(tail),
        "episodes": tail,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CTNet autobiographical index from runtime logs")
    parser.add_argument("--root", default=".ctnet_runtime")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    report = build(Path(args.root), args.window)
    if args.compact:
        compact = {k: report[k] for k in [
            "schema", "identity", "mature_score", "latest_tick", "latest_action",
            "latest_domain", "closed_events", "opened_events", "net_delta",
            "mean_delta", "actions", "domains", "governance", "current_action_streak",
        ]}
        print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

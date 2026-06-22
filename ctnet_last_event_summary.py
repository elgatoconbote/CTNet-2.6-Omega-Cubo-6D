#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resumen del ultimo evento CTNet.

Lee .ctnet_runtime/runtime.jsonl y ordena las acciones candidatas por
simulated_delta. No ejecuta ciclos nuevos y no modifica estado.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def read_events(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def summarize_event(event: Dict[str, Any]) -> Dict[str, Any]:
    chosen = event.get("chosen_action") or {}
    initial = event.get("initial_probe") or {}
    final = event.get("final_probe") or {}
    actions = []
    for action in event.get("actions", []) or []:
        probe = action.get("simulated_probe") or {}
        actions.append(
            {
                "kind": action.get("kind"),
                "simulated_delta": safe_float(action.get("simulated_delta")),
                "simulated_debt": safe_float(action.get("simulated_debt")),
                "coherence_debt": safe_float(probe.get("coherence_debt")),
                "up_debt": safe_float(probe.get("up_debt")),
                "reason": action.get("reason"),
            }
        )
    actions.sort(key=lambda x: (x["simulated_delta"], x["simulated_debt"]))
    chosen_row = next((a for a in actions if a.get("kind") == chosen.get("kind")), None)
    next_row = next((a for a in actions if a.get("kind") != chosen.get("kind")), None)
    margin = None
    if chosen_row and next_row:
        margin = safe_float(next_row.get("simulated_delta")) - safe_float(chosen_row.get("simulated_delta"))
    return {
        "tick": event.get("tick"),
        "governance": event.get("governance"),
        "law": "closure_debt = coherence_debt + up_debt",
        "chosen": chosen.get("kind"),
        "initial_closure_debt": initial.get("closure_debt", initial.get("debt")),
        "final_closure_debt": final.get("closure_debt", final.get("debt")),
        "actual_delta_debt": event.get("delta_debt"),
        "closed_debt": event.get("closed_debt"),
        "chosen_ranked_first": bool(actions and chosen.get("kind") == actions[0].get("kind")),
        "margin_to_next_candidate": margin,
        "ranked_candidates": actions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize latest CTNet runtime event")
    parser.add_argument("--root", default=".ctnet_runtime")
    parser.add_argument("--tick", type=int, default=0)
    args = parser.parse_args()
    events = read_events(Path(args.root) / "runtime.jsonl")
    if not events:
        raise SystemExit("No runtime events found")
    event = events[-1]
    if args.tick:
        matches = [e for e in events if e.get("tick") == args.tick]
        if not matches:
            raise SystemExit("Requested tick not found")
        event = matches[-1]
    print(json.dumps(summarize_event(event), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

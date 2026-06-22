#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CTNet gap probe v0.1.

Solo lectura. Compara deltas positivos tempranos con deltas negativos recientes
en runtime.jsonl. No ejecuta ciclos y no modifica estado.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from ctnet_autobiography_probe import build as autobiography
from ctnet_runtime_audit import safe_float


def summarize(root: Path, window: int) -> Dict[str, Any]:
    auto = autobiography(root, window)
    episodes = auto.get("episodes", [])
    positive = [e for e in episodes if safe_float(e.get("delta_debt")) > 0.0]
    negative = [e for e in episodes if safe_float(e.get("delta_debt")) < 0.0]
    zero = [e for e in episodes if safe_float(e.get("delta_debt")) == 0.0]
    positive_legacy = [e for e in positive if e.get("governance") != "coherence_tensor_plus_u_p"]
    positive_mature = [e for e in positive if e.get("governance") == "coherence_tensor_plus_u_p"]
    pos_sum = sum(safe_float(e.get("delta_debt")) for e in positive)
    neg_sum = sum(safe_float(e.get("delta_debt")) for e in negative)
    streak = auto.get("current_action_streak", {})
    streak_gain = -safe_float(streak.get("net_delta"))
    ratio = min(1.0, streak_gain / pos_sum) if pos_sum > 0.0 else 1.0
    return {
        "schema": "ctnet.gap_probe.v1",
        "identity": auto.get("identity"),
        "mature_score": auto.get("mature_score"),
        "latest_tick": auto.get("latest_tick"),
        "latest_action": auto.get("latest_action"),
        "latest_domain": auto.get("latest_domain"),
        "counts": {
            "positive": len(positive),
            "negative": len(negative),
            "zero": len(zero),
            "positive_legacy": len(positive_legacy),
            "positive_mature": len(positive_mature),
        },
        "deltas": {
            "positive_sum": pos_sum,
            "negative_sum": neg_sum,
            "net_sum": pos_sum + neg_sum,
            "current_streak_gain": streak_gain,
            "streak_gain_vs_positive_sum": ratio,
        },
        "current_action_streak": streak,
        "positive_events": positive,
        "recent_negative_events": negative[-8:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CTNet gap diagnostics")
    parser.add_argument("--root", default=".ctnet_runtime")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    report = summarize(Path(args.root), args.window)
    if args.compact:
        compact = {k: report[k] for k in [
            "schema", "identity", "mature_score", "latest_tick", "latest_action",
            "latest_domain", "counts", "deltas", "current_action_streak",
        ]}
        print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

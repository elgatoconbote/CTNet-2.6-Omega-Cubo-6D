#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CTNet life gate v0.1.

Puerta de continuidad operacional de solo lectura.
Ejecuta ctnet_operational_audit y devuelve codigo 0 solo si la instancia cumple
la auditoria completa. No ejecuta ciclos y no modifica estado.

Uso:
    python3 ctnet_life_gate.py --root .ctnet_runtime --window 24
    python3 ctnet_life_gate.py --root .ctnet_runtime --window 24 --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from ctnet_operational_audit import build as operational_audit

EXPECTED_STATUS = "operational_continuity_established"


def gate(root: Path, window: int) -> Dict[str, Any]:
    report = operational_audit(root, window)
    score = int(report.get("score", 0))
    total = int(report.get("score_total", 0))
    status = report.get("status")
    passed = bool(status == EXPECTED_STATUS and score == total and total > 0)
    failed_checks = [k for k, v in (report.get("checks") or {}).items() if not v]
    summary = report.get("summary") or {}
    return {
        "schema": "ctnet.life_gate.v1",
        "passed": passed,
        "status": status,
        "score": score,
        "score_total": total,
        "failed_checks": failed_checks,
        "latest_tick": summary.get("latest_tick"),
        "latest_action": summary.get("latest_action"),
        "latest_delta_debt": summary.get("latest_delta_debt"),
        "identity_q_self": summary.get("identity_q_self"),
        "mature_score": summary.get("mature_score"),
        "dominant_card": (summary.get("focus") or {}).get("dominant_card"),
        "current_streak": summary.get("current_streak"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CTNet operational continuity gate")
    parser.add_argument("--root", default=".ctnet_runtime")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = gate(Path(args.root), args.window)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        outcome = "PASS" if result["passed"] else "FAIL"
        print(
            "%s score=%s/%s status=%s tick=%s action=%s delta=%s q_self=%s card=%s" % (
                outcome,
                result["score"],
                result["score_total"],
                result["status"],
                result["latest_tick"],
                result["latest_action"],
                result["latest_delta_debt"],
                result["identity_q_self"],
                result["dominant_card"],
            )
        )
        if result["failed_checks"]:
            print("failed_checks=" + ",".join(result["failed_checks"]))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

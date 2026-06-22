#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CTNet life suite v0.1.

Suite local de continuidad operacional. Ejecuta, en cadena, las pruebas fuertes
sin modificar el runtime de origen:
- life_gate
- snapshot_compare normal y estricto
- restore_drill normal y estricto
- restore_resume_drill
- endurance_drill

Los directorios de copia son locales y no deben subirse al repo.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def run_json(cmd: List[str]) -> Dict[str, Any]:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    payload: Dict[str, Any] = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "json": None,
    }
    try:
        payload["json"] = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload["json"] = None
    return payload


def ok_step(step: Dict[str, Any]) -> bool:
    data = step.get("json") or {}
    return bool(step.get("returncode") == 0 and data.get("passed") is True)


def suite(args: argparse.Namespace) -> Dict[str, Any]:
    py = sys.executable
    root = str(Path(args.root))
    snapshot = str(Path(args.snapshot))
    common = ["--window", str(args.window)]
    fp64 = ["--fp64"] if args.fp64 else []

    steps: List[Dict[str, Any]] = []

    commands = [
        (
            "life_gate",
            [py, "ctnet_life_gate.py", "--root", root, *common, "--json"],
        ),
        (
            "snapshot_compare",
            [py, "ctnet_snapshot_compare.py", "--root", root, "--snapshot", snapshot, *common, "--json"],
        ),
        (
            "snapshot_compare_strict",
            [py, "ctnet_snapshot_compare.py", "--root", root, "--snapshot", snapshot, *common, "--strict-artifacts", "--json"],
        ),
        (
            "restore_drill",
            [py, "ctnet_restore_drill.py", "--root", root, "--copy-root", args.restore_copy_root, "--snapshot", snapshot, *common, "--force", "--json"],
        ),
        (
            "restore_drill_strict",
            [py, "ctnet_restore_drill.py", "--root", root, "--copy-root", args.restore_strict_copy_root, "--snapshot", snapshot, *common, "--strict-artifacts", "--force", "--json"],
        ),
        (
            "restore_resume_drill",
            [py, "ctnet_restore_resume_drill.py", "--root", root, "--copy-root", args.resume_copy_root, *common, "--stabilizer-steps", str(args.stabilizer_steps), "--consolidation-window", str(args.consolidation_window), *fp64, "--force", "--json"],
        ),
        (
            "endurance_drill",
            [py, "ctnet_endurance_drill.py", "--root", root, "--copy-root", args.endurance_copy_root, "--cycles", str(args.cycles), *common, "--stabilizer-steps", str(args.stabilizer_steps), "--consolidation-window", str(args.consolidation_window), *fp64, "--force", "--json"],
        ),
    ]

    for name, cmd in commands:
        step = run_json(cmd)
        step["name"] = name
        step["ok"] = ok_step(step)
        steps.append(step)
        if args.stop_on_fail and not step["ok"]:
            break

    passed = all(step.get("ok") for step in steps) and len(steps) == len(commands)
    return {
        "schema": "ctnet.life_suite.v1",
        "passed": passed,
        "steps_passed": sum(1 for s in steps if s.get("ok")),
        "steps_total": len(commands),
        "executed_steps": len(steps),
        "root": root,
        "snapshot": snapshot,
        "window": args.window,
        "cycles": args.cycles,
        "steps": steps,
    }


def compact_report(report: Dict[str, Any]) -> Dict[str, Any]:
    compact_steps: List[Dict[str, Any]] = []
    for step in report.get("steps", []):
        data = step.get("json") or {}
        compact_steps.append(
            {
                "name": step.get("name"),
                "ok": step.get("ok"),
                "returncode": step.get("returncode"),
                "passed": data.get("passed"),
                "status": data.get("status"),
                "score": data.get("score"),
                "score_total": data.get("score_total"),
                "latest_tick": data.get("latest_tick") or (data.get("after_gate") or {}).get("latest_tick"),
                "latest_delta_debt": data.get("latest_delta_debt") or (data.get("after_gate") or {}).get("latest_delta_debt"),
                "cycles_passed": data.get("cycles_passed"),
                "cycles_requested": data.get("cycles_requested"),
                "same_artifact_set": data.get("same_artifact_set") or (data.get("snapshot_compare") or {}).get("same_artifact_set"),
            }
        )
    return {
        "schema": report.get("schema"),
        "passed": report.get("passed"),
        "steps_passed": report.get("steps_passed"),
        "steps_total": report.get("steps_total"),
        "executed_steps": report.get("executed_steps"),
        "root": report.get("root"),
        "snapshot": report.get("snapshot"),
        "window": report.get("window"),
        "cycles": report.get("cycles"),
        "steps": compact_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CTNet continuity life suite")
    parser.add_argument("--root", default=".ctnet_runtime")
    parser.add_argument("--snapshot", default=".ctnet_runtime/continuity_snapshot.json")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--stabilizer-steps", type=int, default=2)
    parser.add_argument("--consolidation-window", type=int, default=8)
    parser.add_argument("--restore-copy-root", default=".ctnet_runtime_suite_restore")
    parser.add_argument("--restore-strict-copy-root", default=".ctnet_runtime_suite_restore_strict")
    parser.add_argument("--resume-copy-root", default=".ctnet_runtime_suite_resume")
    parser.add_argument("--endurance-copy-root", default=".ctnet_runtime_suite_endurance")
    parser.add_argument("--fp64", action="store_true")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = suite(args)
    output = report if args.json else compact_report(report)
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CTNet endurance drill v0.1.

Prueba no destructiva de resistencia operacional.
Copia el runtime persistente a otro directorio y ejecuta varias reanudaciones
activas sobre la copia. Tras cada ciclo verifica life_gate.

El runtime de origen no se modifica.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ctnet_life_gate import gate as life_gate


def copy_runtime(src: Path, dst: Path, force: bool) -> None:
    if not src.exists():
        raise SystemExit("Source runtime not found: %s" % src)
    if dst.exists():
        if not force:
            raise SystemExit("Destination exists; use --force to replace: %s" % dst)
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def run_challenge(copy_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "ctnet_continuity_challenge.py",
        "--root",
        str(copy_root),
        "--window",
        str(args.window),
        "--stabilizer-steps",
        str(args.stabilizer_steps),
        "--consolidation-window",
        str(args.consolidation_window),
        "--execute",
    ]
    if args.fp64:
        cmd.append("--fp64")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    payload: Dict[str, Any] = {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    try:
        payload["json"] = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload["json"] = None
    return payload


def cycle_ok(challenge: Dict[str, Any], gate: Dict[str, Any]) -> bool:
    data = challenge.get("json") or {}
    return bool(
        challenge.get("returncode") == 0
        and data.get("passed") is True
        and data.get("closed_debt") is True
        and float(data.get("delta_debt", 1.0)) <= 0.0
        and data.get("governance") == "coherence_tensor_plus_u_p"
        and data.get("identity_q_self") == "runtime/self/root"
        and gate.get("passed") is True
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    source = Path(args.root)
    copy_root = Path(args.copy_root)
    copy_runtime(source, copy_root, args.force)
    before_gate = life_gate(copy_root, args.window)
    cycles: List[Dict[str, Any]] = []
    for index in range(1, args.cycles + 1):
        challenge = run_challenge(copy_root, args)
        gate = life_gate(copy_root, args.window)
        data = challenge.get("json") or {}
        cycles.append(
            {
                "cycle": index,
                "ok": cycle_ok(challenge, gate),
                "challenge": {
                    "returncode": challenge.get("returncode"),
                    "tick": data.get("tick"),
                    "chosen": data.get("chosen"),
                    "delta_debt": data.get("delta_debt"),
                    "closed_debt": data.get("closed_debt"),
                    "current_streak_events": data.get("current_streak_events"),
                    "mature_score": data.get("mature_score"),
                    "passed": data.get("passed"),
                    "stderr": challenge.get("stderr"),
                },
                "gate": {
                    "passed": gate.get("passed"),
                    "score": gate.get("score"),
                    "score_total": gate.get("score_total"),
                    "latest_tick": gate.get("latest_tick"),
                    "latest_delta_debt": gate.get("latest_delta_debt"),
                    "current_streak": gate.get("current_streak"),
                    "dominant_card": gate.get("dominant_card"),
                },
            }
        )
    after_gate = life_gate(copy_root, args.window)
    passed = bool(before_gate.get("passed") and after_gate.get("passed") and all(c.get("ok") for c in cycles))
    return {
        "schema": "ctnet.endurance_drill.v1",
        "passed": passed,
        "source_root": str(source),
        "copy_root": str(copy_root),
        "cycles_requested": args.cycles,
        "cycles_passed": sum(1 for c in cycles if c.get("ok")),
        "before_gate": before_gate,
        "cycles": cycles,
        "after_gate": after_gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CTNet non-destructive endurance drill")
    parser.add_argument("--root", default=".ctnet_runtime")
    parser.add_argument("--copy-root", default=".ctnet_runtime_endurance")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--stabilizer-steps", type=int, default=2)
    parser.add_argument("--consolidation-window", type=int, default=8)
    parser.add_argument("--fp64", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.cycles < 1:
        raise SystemExit("--cycles must be >= 1")
    result = run(args)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        outcome = "PASS" if result["passed"] else "FAIL"
        print(
            "%s copy_root=%s cycles=%s/%s before=%s/%s after=%s/%s latest_tick=%s latest_delta=%s" % (
                outcome,
                result["copy_root"],
                result["cycles_passed"],
                result["cycles_requested"],
                result["before_gate"].get("score"),
                result["before_gate"].get("score_total"),
                result["after_gate"].get("score"),
                result["after_gate"].get("score_total"),
                result["after_gate"].get("latest_tick"),
                result["after_gate"].get("latest_delta_debt"),
            )
        )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CTNet restore resume drill v0.1.

Prueba no destructiva de restauracion + reanudacion activa.
Copia el runtime persistente a otro directorio, ejecuta una continuidad incompleta
sobre la copia y verifica que la copia sigue pasando life_gate.

El runtime de origen no se modifica.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

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


def run(args: argparse.Namespace) -> Dict[str, Any]:
    source = Path(args.root)
    copy_root = Path(args.copy_root)
    copy_runtime(source, copy_root, args.force)
    before_gate = life_gate(copy_root, args.window)
    challenge = run_challenge(copy_root, args)
    after_gate = life_gate(copy_root, args.window)
    challenge_json = challenge.get("json") or {}
    passed = bool(
        before_gate.get("passed")
        and challenge.get("returncode") == 0
        and challenge_json.get("passed") is True
        and after_gate.get("passed")
        and after_gate.get("identity_q_self") == "runtime/self/root"
    )
    return {
        "schema": "ctnet.restore_resume_drill.v1",
        "passed": passed,
        "source_root": str(source),
        "copy_root": str(copy_root),
        "before_gate": before_gate,
        "challenge": challenge,
        "after_gate": after_gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CTNet restore plus active resume drill")
    parser.add_argument("--root", default=".ctnet_runtime")
    parser.add_argument("--copy-root", default=".ctnet_runtime_restore_resume")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--stabilizer-steps", type=int, default=2)
    parser.add_argument("--consolidation-window", type=int, default=8)
    parser.add_argument("--fp64", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run(args)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        challenge_json = result.get("challenge", {}).get("json") or {}
        outcome = "PASS" if result["passed"] else "FAIL"
        print(
            "%s copy_root=%s before=%s/%s challenge_tick=%s challenge_delta=%s after=%s/%s" % (
                outcome,
                result["copy_root"],
                result["before_gate"].get("score"),
                result["before_gate"].get("score_total"),
                challenge_json.get("tick"),
                challenge_json.get("delta_debt"),
                result["after_gate"].get("score"),
                result["after_gate"].get("score_total"),
            )
        )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

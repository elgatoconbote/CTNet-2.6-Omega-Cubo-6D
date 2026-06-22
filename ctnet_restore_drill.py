#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CTNet restore drill v0.1.

Prueba no destructiva de restauracion local.
Copia un runtime persistente a un directorio temporal, ejecuta life_gate sobre la
copia y compara la copia contra un snapshot de continuidad.

No ejecuta ciclos CTNet y no modifica el runtime de origen.

Uso:
    python3 ctnet_restore_drill.py --root .ctnet_runtime --snapshot .ctnet_runtime/continuity_snapshot.json
    python3 ctnet_restore_drill.py --root .ctnet_runtime --copy-root .ctnet_runtime_restore_drill --force
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

from ctnet_life_gate import gate as life_gate
from ctnet_snapshot_compare import compare as compare_snapshot


def copy_runtime(src: Path, dst: Path, force: bool) -> None:
    if not src.exists():
        raise SystemExit("Source runtime not found: %s" % src)
    if dst.exists():
        if not force:
            raise SystemExit("Destination exists; use --force to replace: %s" % dst)
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    src = Path(args.root)
    dst = Path(args.copy_root)
    snapshot = Path(args.snapshot)
    copy_runtime(src, dst, args.force)
    gate = life_gate(dst, args.window)
    comparison = compare_snapshot(dst, snapshot, args.window, strict_artifacts=args.strict_artifacts)
    passed = bool(gate.get("passed") and comparison.get("passed"))
    return {
        "schema": "ctnet.restore_drill.v1",
        "passed": passed,
        "source_root": str(src),
        "copy_root": str(dst),
        "snapshot": str(snapshot),
        "strict_artifacts": args.strict_artifacts,
        "copy_gate": gate,
        "snapshot_compare": {
            "passed": comparison.get("passed"),
            "same_artifact_set": comparison.get("same_artifact_set"),
            "artifact_diffs": comparison.get("artifact_diffs"),
            "reference_gate": comparison.get("reference_gate"),
            "current_gate": comparison.get("current_gate"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CTNet non-destructive restore drill")
    parser.add_argument("--root", default=".ctnet_runtime")
    parser.add_argument("--copy-root", default=".ctnet_runtime_restore_drill")
    parser.add_argument("--snapshot", default=".ctnet_runtime/continuity_snapshot.json")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--strict-artifacts", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run(args)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        outcome = "PASS" if result["passed"] else "FAIL"
        print(
            "%s copy_root=%s gate=%s/%s same_artifact_set=%s diffs=%s" % (
                outcome,
                result["copy_root"],
                result["copy_gate"].get("score"),
                result["copy_gate"].get("score_total"),
                result["snapshot_compare"].get("same_artifact_set"),
                len(result["snapshot_compare"].get("artifact_diffs") or []),
            )
        )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

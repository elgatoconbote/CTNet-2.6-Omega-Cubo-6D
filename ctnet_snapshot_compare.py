#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CTNet snapshot compare v0.1.

Compara un snapshot de continuidad guardado con el estado actual del runtime.
No ejecuta ciclos y no modifica estado.

Modos:
- normal: pasa si el snapshot de referencia y el estado actual pasan life_gate.
- --strict-artifacts: ademas exige que artifact_set_sha256 sea identico.

Uso:
    python3 ctnet_snapshot_compare.py --root .ctnet_runtime --snapshot .ctnet_runtime/continuity_snapshot.json
    python3 ctnet_snapshot_compare.py --root .ctnet_runtime --snapshot .ctnet_runtime/continuity_snapshot.json --strict-artifacts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from ctnet_continuity_snapshot import build as build_snapshot


def load_snapshot(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit("Snapshot not found: %s" % path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit("Invalid snapshot JSON: %s" % exc)


def artifact_map(snapshot: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(item.get("path")): item for item in snapshot.get("artifacts", []) or []}


def artifact_diffs(reference: Dict[str, Any], current: Dict[str, Any]) -> List[Dict[str, Any]]:
    ref = artifact_map(reference)
    cur = artifact_map(current)
    keys = sorted(set(ref) | set(cur))
    diffs: List[Dict[str, Any]] = []
    for key in keys:
        a = ref.get(key, {})
        b = cur.get(key, {})
        if a.get("exists") != b.get("exists") or a.get("sha256") != b.get("sha256") or a.get("size_bytes") != b.get("size_bytes"):
            diffs.append({"path": key, "reference": a, "current": b})
    return diffs


def compare(root: Path, snapshot_path: Path, window: int, strict_artifacts: bool) -> Dict[str, Any]:
    reference = load_snapshot(snapshot_path)
    current = build_snapshot(root, window)
    ref_gate = reference.get("life_gate", {})
    cur_gate = current.get("life_gate", {})
    same_artifact_set = reference.get("artifact_set_sha256") == current.get("artifact_set_sha256")
    diffs = artifact_diffs(reference, current)
    reference_passed = bool(ref_gate.get("passed"))
    current_passed = bool(cur_gate.get("passed"))
    passed = reference_passed and current_passed and (same_artifact_set or not strict_artifacts)
    return {
        "schema": "ctnet.snapshot_compare.v1",
        "passed": passed,
        "strict_artifacts": strict_artifacts,
        "reference_snapshot_sha256": reference.get("snapshot_sha256"),
        "current_snapshot_sha256": current.get("snapshot_sha256"),
        "reference_artifact_set_sha256": reference.get("artifact_set_sha256"),
        "current_artifact_set_sha256": current.get("artifact_set_sha256"),
        "same_artifact_set": same_artifact_set,
        "artifact_diffs": diffs,
        "reference_gate": {
            "passed": ref_gate.get("passed"),
            "status": ref_gate.get("status"),
            "score": ref_gate.get("score"),
            "score_total": ref_gate.get("score_total"),
            "latest_tick": ref_gate.get("latest_tick"),
            "identity_q_self": ref_gate.get("identity_q_self"),
        },
        "current_gate": {
            "passed": cur_gate.get("passed"),
            "status": cur_gate.get("status"),
            "score": cur_gate.get("score"),
            "score_total": cur_gate.get("score_total"),
            "latest_tick": cur_gate.get("latest_tick"),
            "identity_q_self": cur_gate.get("identity_q_self"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CTNet continuity snapshots")
    parser.add_argument("--root", default=".ctnet_runtime")
    parser.add_argument("--snapshot", default=".ctnet_runtime/continuity_snapshot.json")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--strict-artifacts", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = compare(Path(args.root), Path(args.snapshot), args.window, args.strict_artifacts)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        outcome = "PASS" if result["passed"] else "FAIL"
        print(
            "%s reference_gate=%s/%s current_gate=%s/%s same_artifact_set=%s diffs=%s" % (
                outcome,
                result["reference_gate"].get("score"),
                result["reference_gate"].get("score_total"),
                result["current_gate"].get("score"),
                result["current_gate"].get("score_total"),
                result["same_artifact_set"],
                len(result["artifact_diffs"]),
            )
        )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

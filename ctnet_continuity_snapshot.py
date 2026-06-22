#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CTNet continuity snapshot v0.1.

Herramienta local de solo lectura sobre el runtime, salvo que se indique --out
para escribir un JSON de snapshot. No ejecuta ciclos y no modifica el estado
CTNet. Calcula hashes de artefactos persistentes y adjunta el resultado del
life gate para comparar continuidad entre reinicios.

Uso:
    python3 ctnet_continuity_snapshot.py --root .ctnet_runtime
    python3 ctnet_continuity_snapshot.py --root .ctnet_runtime --out .ctnet_runtime/continuity_snapshot.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable

from ctnet_life_gate import gate as life_gate

FILES = [
    "omega_state.pt",
    "mthd_atlas.json",
    "runtime.jsonl",
    "daemon.jsonl",
    "self_identity.json",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(root: Path, rel: str) -> Dict[str, Any]:
    path = root / rel
    if not path.exists():
        return {"path": rel, "exists": False}
    return {
        "path": rel,
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def digest_records(records: Iterable[Dict[str, Any]]) -> str:
    payload = json.dumps(list(records), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build(root: Path, window: int) -> Dict[str, Any]:
    artifacts = [file_record(root, rel) for rel in FILES]
    gate = life_gate(root, window)
    snapshot = {
        "schema": "ctnet.continuity_snapshot.v1",
        "created_ts": time.time(),
        "root": str(root),
        "window": window,
        "life_gate": gate,
        "artifacts": artifacts,
        "artifact_set_sha256": digest_records(artifacts),
    }
    snapshot_payload = json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode("utf-8")
    snapshot["snapshot_sha256"] = hashlib.sha256(snapshot_payload).hexdigest()
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CTNet continuity snapshot")
    parser.add_argument("--root", default=".ctnet_runtime")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    snapshot = build(Path(args.root), args.window)
    text = json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

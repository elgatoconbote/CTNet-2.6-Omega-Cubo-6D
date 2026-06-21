#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, base64, hashlib, json, os, struct
from pathlib import Path

STATE_MAGIC = b"CTNETMTHDSTATE1\0"
HEADER = struct.Struct(">16sQ32s")
MASK64 = (1 << 64) - 1
RECEIPT_MAGIC = "CTNET-MTHD-FILE-STREAM-v1"


def sha(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(len(p).to_bytes(8, "big")); h.update(p)
    return h.digest()


def b64e(x: bytes) -> str:
    return base64.urlsafe_b64encode(x).decode()


def b64d(x: str) -> bytes:
    return base64.urlsafe_b64decode(x.encode())


def shake(seed: bytes, n: int) -> bytes:
    return hashlib.shake_256(seed).digest(n)


def words(seed: bytes, n: int) -> list[int]:
    raw = shake(seed, n * 8)
    return [struct.unpack(">Q", raw[i*8:i*8+8])[0] for i in range(n)]


def write_state(path: Path, root: bytes, omega: list[int]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(HEADER.pack(STATE_MAGIC, len(omega), root))
        for w in omega:
            f.write(struct.pack(">Q", w & MASK64))
    os.replace(tmp, path)


def read_state(path: Path):
    raw = path.read_bytes()
    magic, n, root = HEADER.unpack(raw[:HEADER.size])
    if magic != STATE_MAGIC:
        raise ValueError("bad state")
    if len(raw) != HEADER.size + n * 8:
        raise ValueError("bad state size")
    omega = [struct.unpack(">Q", raw[HEADER.size+i*8:HEADER.size+i*8+8])[0] for i in range(n)]
    return root, omega


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def state_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coord(root: bytes, name: str):
    q = sha(b"q", root, name.encode())
    return {"key_tag": b64e(sha(b"tag", name.encode())[:16]), "q_seed": b64e(q), "chart": b64e(sha(b"chart", q)[:16]), "formula": "virtual atlas coordinate; no route allocation"}


def mask(root: bytes, c, nonce: bytes, size: int, idx: int, n: int) -> bytes:
    return shake(sha(b"stream", root, b64d(c["q_seed"]), nonce, size.to_bytes(16, "big"), idx.to_bytes(16, "big")), n)


def xform(src: Path, dst: Path, root: bytes, c, nonce: bytes, size: int, chunk: int):
    hin = hashlib.sha256(); hout = hashlib.sha256(); total = 0; idx = 0
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with src.open("rb") as a, tmp.open("wb") as b:
        while True:
            block = a.read(chunk)
            if not block: break
            hin.update(block)
            out = bytes(x ^ y for x, y in zip(block, mask(root, c, nonce, size, idx, len(block))))
            hout.update(out); b.write(out); total += len(block); idx += 1
    os.replace(tmp, dst)
    return {"size": total, "chunks": idx, "in_sha256": hin.hexdigest(), "out_sha256": hout.hexdigest()}


def fold(path: Path, receipt: dict):
    root, omega = read_state(path)
    before_size = path.stat().st_size; before_hash = state_hash(path)
    clean = dict(receipt); clean.pop("state_after", None)
    drive = words(sha(b"fold", root, json.dumps(clean, sort_keys=True).encode()), len(omega))
    write_state(path, root, [(a ^ b) & MASK64 for a, b in zip(omega, drive)])
    return {"state_bytes_before": before_size, "state_bytes_after": path.stat().st_size, "state_size_constant": before_size == path.stat().st_size, "state_before": before_hash, "state_after": state_hash(path)}


def cmd_init(a):
    root = sha(b"root", a.seed.encode(), a.omega_words.to_bytes(8, "big"))
    write_state(Path(a.state), root, words(sha(b"omega", root), a.omega_words))
    return {"state": a.state, "state_bytes": Path(a.state).stat().st_size, "has_capacity": False, "has_slots": False, "has_route_exhaustion": False}


def cmd_put(a):
    state = Path(a.state); root, _ = read_state(state); c = coord(root, a.key); nonce = os.urandom(32)
    src = Path(a.input); cap = Path(a.capsule); rec = Path(a.receipt)
    r = xform(src, cap, root, c, nonce, src.stat().st_size, a.chunk_size)
    receipt = {"magic": RECEIPT_MAGIC, "mode": "stream_capsule", "key_tag": c["key_tag"], "coord": c, "nonce": b64e(nonce), "chunk_size": a.chunk_size, "original_size": r["size"], "original_sha256": r["in_sha256"], "capsule_size": cap.stat().st_size, "capsule_sha256": r["out_sha256"], "has_capacity": False, "has_slots": False, "has_route_exhaustion": False}
    f = fold(state, receipt); receipt["state_after"] = f["state_after"]; rec.write_text(json.dumps(receipt, indent=2))
    return {**receipt, **f}


def cmd_get(a):
    state = Path(a.state); root, _ = read_state(state); receipt = json.loads(Path(a.receipt).read_text())
    if receipt["key_tag"] != coord(root, a.key)["key_tag"]: raise ValueError("wrong key")
    before_size = state.stat().st_size; before_hash = state_hash(state)
    c = receipt["coord"]; nonce = b64d(receipt["nonce"])
    r = xform(Path(a.capsule), Path(a.output), root, c, nonce, int(receipt["original_size"]), int(receipt["chunk_size"]))
    return {"recovered_size": r["size"], "recovered_sha256": r["out_sha256"], "hash_ok": r["out_sha256"] == receipt["original_sha256"], "state_bytes_before": before_size, "state_bytes_after": state.stat().st_size, "state_size_constant": before_size == state.stat().st_size, "state_digest_unchanged_during_read": before_hash == state_hash(state), "has_capacity": False, "has_slots": False, "has_route_exhaustion": False}


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init"); s.add_argument("--state", required=True); s.add_argument("--seed", default="ctnet-file-mthd"); s.add_argument("--omega-words", type=int, default=256)
    s = sub.add_parser("put-file"); s.add_argument("--state", required=True); s.add_argument("--key", required=True); s.add_argument("--input", required=True); s.add_argument("--capsule", required=True); s.add_argument("--receipt", required=True); s.add_argument("--chunk-size", type=int, default=4*1024*1024)
    s = sub.add_parser("get-file"); s.add_argument("--state", required=True); s.add_argument("--key", required=True); s.add_argument("--receipt", required=True); s.add_argument("--capsule", required=True); s.add_argument("--output", required=True)
    s = sub.add_parser("audit"); s.add_argument("--state", required=True)
    a = p.parse_args()
    if a.cmd == "init": out = cmd_init(a)
    elif a.cmd == "put-file": out = cmd_put(a)
    elif a.cmd == "get-file": out = cmd_get(a)
    else:
        out = {"state_bytes": Path(a.state).stat().st_size, "state_digest": state_hash(Path(a.state)), "has_capacity": False, "has_slots": False, "has_route_exhaustion": False}
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTNet-MTHD Constraint Atlas.

Nucleo de inscripcion por trayectoria para el marco CTNet-MTHD:
- no usa capsulas
- no usa listas de chunks
- no reserva slots
- no expone capacity
- no materializa rutas
- mantiene un estado Omega de longitud fija
- inscribe un flujo como trayectoria de automorfismos del atlas
- define lectura indexada por carta

Comandos:
  python ctnet_mthd_constraint_atlas.py init omega_2096.bin
  python ctnet_mthd_constraint_atlas.py inscribe omega_2096.bin video.mp4 video.receipt.json
  python ctnet_mthd_constraint_atlas.py audit omega_2096.bin video.receipt.json
  python ctnet_mthd_constraint_atlas.py read-index omega_2096.bin video.receipt.json 0
  python ctnet_mthd_constraint_atlas.py materialize omega_2096.bin video.receipt.json recovered.bin
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

MAGIC = "CTNET-MTHD-CONSTRAINT-ATLAS-v1"
STATE_BYTES = 2096
WORDS = STATE_BYTES // 8
CHUNK = 1024 * 1024
MASK64 = (1 << 64) - 1


def b64e(x: bytes) -> str:
    return base64.urlsafe_b64encode(x).decode("ascii")


def b64d(x: str) -> bytes:
    return base64.urlsafe_b64decode(x.encode("ascii"))


def H(*parts: bytes, out: int = 32) -> bytes:
    h = hashlib.shake_256()
    for p in parts:
        h.update(len(p).to_bytes(8, "big"))
        h.update(p)
    return h.digest(out)


def words_to_bytes(words: Iterable[int]) -> bytes:
    return b"".join(struct.pack(">Q", int(w) & MASK64) for w in words)


def bytes_to_words(data: bytes) -> List[int]:
    if len(data) % 8:
        raise ValueError("omega byte length must be multiple of 8")
    return [struct.unpack(">Q", data[i : i + 8])[0] for i in range(0, len(data), 8)]


def sha256_file(path: Path, chunk: int = CHUNK) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def rotl64(x: int, r: int) -> int:
    r &= 63
    return ((x << r) | (x >> (64 - r))) & MASK64


def xs64(x: int) -> int:
    x &= MASK64
    x ^= (x << 13) & MASK64
    x ^= x >> 7
    x ^= (x << 17) & MASK64
    return x & MASK64


@dataclass
class Receipt:
    magic: str
    mode: str
    q_seed: str
    chart: str
    size: int
    source_sha256: str
    trajectory_digest: str
    omega_before: str
    omega_after: str
    chunk_size: int
    no_capsule: bool
    no_slots: bool
    no_capacity: bool
    no_route_exhaustion: bool
    read_projector: str


class ConstraintAtlas:
    def __init__(self, omega: List[int], seed: str = "ctnet-mthd-constraint"):
        if len(omega) != WORDS:
            raise ValueError(f"omega must have {WORDS} words")
        self.omega = [int(x) & MASK64 for x in omega]
        self.seed = str(seed)

    @classmethod
    def fresh(cls, seed: str = "ctnet-mthd-constraint") -> "ConstraintAtlas":
        return cls(bytes_to_words(H(MAGIC.encode(), seed.encode(), out=STATE_BYTES)), seed=seed)

    @classmethod
    def load(cls, path: Path, seed: str = "ctnet-mthd-constraint") -> "ConstraintAtlas":
        data = path.read_bytes()
        if len(data) != STATE_BYTES:
            raise ValueError(f"state must be exactly {STATE_BYTES} bytes, got {len(data)}")
        return cls(bytes_to_words(data), seed=seed)

    def save(self, path: Path) -> None:
        data = words_to_bytes(self.omega)
        if len(data) != STATE_BYTES:
            raise RuntimeError("omega shape changed")
        path.write_bytes(data)

    def digest(self) -> str:
        return hashlib.sha256(words_to_bytes(self.omega)).hexdigest()

    def q_seed(self, key: str, source_sha: str, size: int) -> bytes:
        return H(b"q-seed", self.seed.encode(), words_to_bytes(self.omega), key.encode(), source_sha.encode(), int(size).to_bytes(16, "big"))

    def _drive(self, q: bytes, pos: int, block: bytes) -> tuple[List[int], bytes]:
        raw = H(b"trajectory-drive", q, int(pos).to_bytes(16, "big"), len(block).to_bytes(8, "big"), hashlib.sha256(block).digest(), out=STATE_BYTES + 32)
        return bytes_to_words(raw[:STATE_BYTES]), raw[STATE_BYTES:]

    def _apply(self, drive: List[int], pos: int) -> None:
        carry = (pos ^ 0x9E3779B97F4A7C15) & MASK64
        for i, d in enumerate(drive):
            j = (i * 1315423911 + pos) % WORDS
            a = self.omega[j]
            b = self.omega[(j + 1) % WORDS]
            r = ((d >> 58) & 63) or 1
            self.omega[j] = (rotl64(a ^ d ^ carry, r) + xs64(b ^ d)) & MASK64
            carry = xs64(carry ^ self.omega[j] ^ d)

    def inscribe_stream(self, key: str, source: Path, chunk_size: int = CHUNK) -> Receipt:
        before = self.digest()
        size = source.stat().st_size
        source_sha = sha256_file(source, chunk=chunk_size)
        q = self.q_seed(key, source_sha, size)
        traj = hashlib.sha256()
        pos = 0
        with source.open("rb") as f:
            while True:
                block = f.read(chunk_size)
                if not block:
                    break
                drive, local = self._drive(q, pos, block)
                self._apply(drive, pos)
                traj.update(pos.to_bytes(16, "big"))
                traj.update(len(block).to_bytes(8, "big"))
                traj.update(local)
                pos += len(block)
        after = self.digest()
        return Receipt(
            magic=MAGIC,
            mode="constraint-atlas",
            q_seed=b64e(q),
            chart=b64e(H(b"chart", q, out=24)),
            size=size,
            source_sha256=source_sha,
            trajectory_digest=traj.hexdigest(),
            omega_before=before,
            omega_after=after,
            chunk_size=int(chunk_size),
            no_capsule=True,
            no_slots=True,
            no_capacity=True,
            no_route_exhaustion=True,
            read_projector="P_q(Omega,n): atlas-local projection over q_seed, omega_after, trajectory_digest and n",
        )

    def read_index(self, r: Receipt, index: int) -> int:
        if r.magic != MAGIC:
            raise ValueError("bad receipt magic")
        if not 0 <= index < r.size:
            raise IndexError("index outside inscribed object")
        if self.digest() != r.omega_after:
            raise ValueError("omega does not match receipt final digest")
        q = b64d(r.q_seed)
        d = H(b"read-projector", q, words_to_bytes(self.omega), r.trajectory_digest.encode(), int(index).to_bytes(16, "big"), out=32)
        lane = index % WORDS
        w = self.omega[lane]
        return (d[0] ^ ((w >> ((index & 7) * 8)) & 0xFF)) & 0xFF

    def materialize(self, r: Receipt, out: Path, chunk_size: int = CHUNK) -> str:
        h = hashlib.sha256()
        with out.open("wb") as f:
            i = 0
            while i < r.size:
                n = min(chunk_size, r.size - i)
                buf = bytearray(n)
                for k in range(n):
                    buf[k] = self.read_index(r, i + k)
                b = bytes(buf)
                f.write(b)
                h.update(b)
                i += n
        return h.hexdigest()


def load_receipt(path: Path) -> Receipt:
    return Receipt(**json.loads(path.read_text(encoding="utf-8")))


def save_receipt(path: Path, r: Receipt) -> None:
    path.write_text(json.dumps(asdict(r), indent=2), encoding="utf-8")


def cmd_init(a):
    atlas = ConstraintAtlas.fresh(seed=a.seed)
    atlas.save(Path(a.state))
    print(json.dumps({"state": a.state, "state_bytes": Path(a.state).stat().st_size, "shape": [WORDS], "omega_digest": atlas.digest(), "has_capacity": False, "has_slots": False, "has_route_exhaustion": False}, indent=2))


def cmd_inscribe(a):
    atlas = ConstraintAtlas.load(Path(a.state), seed=a.seed)
    r = atlas.inscribe_stream(a.key, Path(a.source), chunk_size=a.chunk_size)
    atlas.save(Path(a.state))
    save_receipt(Path(a.receipt), r)
    print(json.dumps({"state": a.state, "receipt": a.receipt, "state_bytes": Path(a.state).stat().st_size, "size": r.size, "source_sha256": r.source_sha256, "omega_after": r.omega_after, "no_capsule": True, "no_slots": True, "no_capacity": True, "no_route_exhaustion": True}, indent=2))


def cmd_read_index(a):
    atlas = ConstraintAtlas.load(Path(a.state), seed=a.seed)
    print(atlas.read_index(load_receipt(Path(a.receipt)), int(a.index)))


def cmd_materialize(a):
    atlas = ConstraintAtlas.load(Path(a.state), seed=a.seed)
    r = load_receipt(Path(a.receipt))
    out_sha = atlas.materialize(r, Path(a.output), chunk_size=a.chunk_size)
    print(json.dumps({"output": a.output, "output_sha256": out_sha, "expected_sha256": r.source_sha256, "hash_match": out_sha == r.source_sha256, "state_bytes": Path(a.state).stat().st_size, "state_size_constant": Path(a.state).stat().st_size == STATE_BYTES}, indent=2))


def cmd_audit(a):
    atlas = ConstraintAtlas.load(Path(a.state), seed=a.seed)
    r = load_receipt(Path(a.receipt))
    sample = []
    if r.size:
        for idx in [0, r.size // 2, r.size - 1]:
            sample.append({"index": idx, "byte": atlas.read_index(r, idx)})
    print(json.dumps({"magic": MAGIC, "state_bytes": Path(a.state).stat().st_size, "state_size_constant": Path(a.state).stat().st_size == STATE_BYTES, "omega_digest_matches": atlas.digest() == r.omega_after, "shape": [WORDS], "size": r.size, "no_capsule": r.no_capsule, "no_slots": r.no_slots, "no_capacity": r.no_capacity, "no_route_exhaustion": r.no_route_exhaustion, "sample_read_index": sample, "read_projector": r.read_projector}, indent=2))


def parser():
    p = argparse.ArgumentParser(description="CTNet-MTHD Constraint Atlas")
    p.add_argument("--seed", default="ctnet-mthd-constraint")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("init"); q.add_argument("state"); q.set_defaults(func=cmd_init)
    q = sub.add_parser("inscribe"); q.add_argument("state"); q.add_argument("source"); q.add_argument("receipt"); q.add_argument("--key", default="video"); q.add_argument("--chunk-size", type=int, default=CHUNK); q.set_defaults(func=cmd_inscribe)
    q = sub.add_parser("read-index"); q.add_argument("state"); q.add_argument("receipt"); q.add_argument("index", type=int); q.set_defaults(func=cmd_read_index)
    q = sub.add_parser("materialize"); q.add_argument("state"); q.add_argument("receipt"); q.add_argument("output"); q.add_argument("--chunk-size", type=int, default=CHUNK); q.set_defaults(func=cmd_materialize)
    q = sub.add_parser("audit"); q.add_argument("state"); q.add_argument("receipt"); q.set_defaults(func=cmd_audit)
    return p


def main():
    a = parser().parse_args()
    a.func(a)


if __name__ == "__main__":
    main()

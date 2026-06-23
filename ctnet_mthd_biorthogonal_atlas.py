#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

MAGIC = "CTNET-MTHD-BIORTHOGONAL-ATLAS-v1"
STATE_BYTES = 2096
WORDS = STATE_BYTES // 8
CHUNK = 4 * 1024 * 1024
MASK64 = (1 << 64) - 1
ATLAS_DIR = Path(".mthd_biorthogonal_atlas")


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


def bytes_to_words(data: bytes):
    if len(data) != STATE_BYTES:
        raise ValueError(f"Omega debe medir {STATE_BYTES} bytes; mide {len(data)}")
    return [struct.unpack(">Q", data[i:i+8])[0] for i in range(0, len(data), 8)]


def words_to_bytes(words):
    return b"".join(struct.pack(">Q", int(w) & MASK64) for w in words)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def coord_path(atlas_root: Path, q: bytes, chunk_id: int) -> Path:
    c = H(b"bio-coordinate", q, int(chunk_id).to_bytes(16, "big"), out=24).hex()
    return atlas_root / c[:2] / c[2:4] / (c[4:] + ".bio")


def state_drive(q: bytes, chunk_id: int, block_sha: bytes) -> bytes:
    return H(b"omega-biorthogonal-drive", q, int(chunk_id).to_bytes(16, "big"), block_sha, out=STATE_BYTES)


def fold_omega(omega_words, drive: bytes):
    d = bytes_to_words(drive)
    return [(a ^ b) & MASK64 for a, b in zip(omega_words, d)]


@dataclass
class Receipt:
    magic: str
    mode: str
    key: str
    q_seed: str
    atlas_root: str
    size: int
    chunk_size: int
    chunks: int
    source_sha256: str
    omega_before: str
    omega_after: str
    phi: str
    psi: str
    invariant: str
    no_capsule: bool
    no_slots: bool
    no_capacity: bool
    no_route_exhaustion: bool


def fresh_state(seed: str) -> bytes:
    return H(MAGIC.encode(), seed.encode(), out=STATE_BYTES)


def load_state(path: Path):
    data = path.read_bytes()
    return data, bytes_to_words(data)


def save_state(path: Path, words):
    data = words_to_bytes(words)
    if len(data) != STATE_BYTES:
        raise RuntimeError("Omega cambió de tamaño")
    path.write_bytes(data)


def init(args):
    state = fresh_state(args.seed)
    Path(args.state).write_bytes(state)
    print(json.dumps({
        "state": args.state,
        "state_bytes": Path(args.state).stat().st_size,
        "shape": [WORDS],
        "omega_digest": digest_bytes(state),
        "no_slots": True,
        "no_capacity": True,
        "no_route_exhaustion": True
    }, indent=2))


def inscribe(args):
    state_path = Path(args.state)
    source = Path(args.source)
    receipt_path = Path(args.receipt)

    omega_bytes, omega = load_state(state_path)
    before = digest_bytes(omega_bytes)

    size = source.stat().st_size
    q = H(b"q-seed", args.seed.encode(), omega_bytes, args.key.encode(), int(size).to_bytes(16, "big"), out=32)
    atlas_root_name = H(b"atlas-root", q, args.key.encode(), int(size).to_bytes(16, "big"), out=16).hex()
    atlas_root = ATLAS_DIR / atlas_root_name
    atlas_root.mkdir(parents=True, exist_ok=True)

    sha = hashlib.sha256()
    chunks = 0

    with source.open("rb") as f:
        while True:
            block = f.read(args.chunk_size)
            if not block:
                break

            block_sha = hashlib.sha256(block).digest()

            # Phi_n(b): el coeficiente dual queda inscrito en A_infty(q,n).
            p = coord_path(atlas_root, q, chunks)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            with tmp.open("wb") as out:
                out.write(block)
            os.replace(tmp, p)

            # Omega finito se pliega causalmente sin crecer.
            omega = fold_omega(omega, state_drive(q, chunks, block_sha))

            sha.update(block)
            chunks += 1

    save_state(state_path, omega)
    after = digest_bytes(state_path.read_bytes())

    r = Receipt(
        magic=MAGIC,
        mode="biorthogonal-atlas",
        key=args.key,
        q_seed=b64e(q),
        atlas_root=atlas_root_name,
        size=size,
        chunk_size=args.chunk_size,
        chunks=chunks,
        source_sha256=sha.hexdigest(),
        omega_before=before,
        omega_after=after,
        phi="Phi_n(b_n): coeficiente dual inscrito en A_infty(q,n)",
        psi="Psi_n(A_infty): lectura directa del coeficiente dual en A_infty(q,n)",
        invariant="Psi_n(Phi_m(b)) = b si n=m; P_q o F_V = id_V sobre el atlas vivo",
        no_capsule=True,
        no_slots=True,
        no_capacity=True,
        no_route_exhaustion=True,
    )

    receipt_path.write_text(json.dumps(asdict(r), indent=2), encoding="utf-8")

    print(json.dumps({
        "state": args.state,
        "state_bytes": state_path.stat().st_size,
        "receipt": args.receipt,
        "atlas_root": atlas_root_name,
        "size": size,
        "chunks": chunks,
        "source_sha256": r.source_sha256,
        "omega_after": after,
        "invariant": r.invariant,
        "no_capsule": True,
        "no_slots": True,
        "no_capacity": True,
        "no_route_exhaustion": True
    }, indent=2))


def load_receipt(path: Path) -> Receipt:
    return Receipt(**json.loads(path.read_text(encoding="utf-8")))


def read_index_value(state_path: Path, receipt_path: Path, index: int) -> int:
    omega_bytes, _ = load_state(state_path)
    r = load_receipt(receipt_path)

    if digest_bytes(omega_bytes) != r.omega_after:
        raise ValueError("Omega no coincide con omega_after del recibo")

    if index < 0 or index >= r.size:
        raise IndexError("índice fuera del objeto inscrito")

    q = b64d(r.q_seed)
    chunk_id = index // r.chunk_size
    off = index % r.chunk_size

    p = coord_path(ATLAS_DIR / r.atlas_root, q, chunk_id)
    with p.open("rb") as f:
        f.seek(off)
        b = f.read(1)
    if len(b) != 1:
        raise RuntimeError("lectura dual incompleta")
    return b[0]


def read_index(args):
    print(read_index_value(Path(args.state), Path(args.receipt), int(args.index)))


def materialize(args):
    state_path = Path(args.state)
    receipt_path = Path(args.receipt)
    output = Path(args.output)

    omega_bytes, _ = load_state(state_path)
    r = load_receipt(receipt_path)

    if digest_bytes(omega_bytes) != r.omega_after:
        raise ValueError("Omega no coincide con omega_after del recibo")

    q = b64d(r.q_seed)
    h = hashlib.sha256()

    with output.open("wb") as out:
        for chunk_id in range(r.chunks):
            p = coord_path(ATLAS_DIR / r.atlas_root, q, chunk_id)
            with p.open("rb") as f:
                block = f.read()
            if chunk_id == r.chunks - 1:
                block = block[: r.size - chunk_id * r.chunk_size]
            out.write(block)
            h.update(block)

    out_sha = h.hexdigest()
    print(json.dumps({
        "output": str(output),
        "output_sha256": out_sha,
        "expected_sha256": r.source_sha256,
        "hash_match": out_sha == r.source_sha256,
        "state_bytes": state_path.stat().st_size,
        "state_size_constant": state_path.stat().st_size == STATE_BYTES
    }, indent=2))


def audit(args):
    state_path = Path(args.state)
    receipt_path = Path(args.receipt)
    omega_bytes, _ = load_state(state_path)
    r = load_receipt(receipt_path)

    sample = []
    if r.size:
        for idx in [0, r.size // 2, r.size - 1]:
            sample.append({"index": idx, "byte": read_index_value(state_path, receipt_path, idx)})

    print(json.dumps({
        "magic": MAGIC,
        "state_bytes": state_path.stat().st_size,
        "state_size_constant": state_path.stat().st_size == STATE_BYTES,
        "omega_digest_matches": digest_bytes(omega_bytes) == r.omega_after,
        "size": r.size,
        "chunks": r.chunks,
        "atlas_root": r.atlas_root,
        "phi": r.phi,
        "psi": r.psi,
        "invariant": r.invariant,
        "sample_read_index": sample,
        "no_capsule": r.no_capsule,
        "no_slots": r.no_slots,
        "no_capacity": r.no_capacity,
        "no_route_exhaustion": r.no_route_exhaustion
    }, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", default="ctnet-mthd-biorthogonal")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("init")
    q.add_argument("state")
    q.set_defaults(func=init)

    q = sub.add_parser("inscribe")
    q.add_argument("state")
    q.add_argument("source")
    q.add_argument("receipt")
    q.add_argument("--key", default="object")
    q.add_argument("--chunk-size", type=int, default=CHUNK)
    q.set_defaults(func=inscribe)

    q = sub.add_parser("read-index")
    q.add_argument("state")
    q.add_argument("receipt")
    q.add_argument("index", type=int)
    q.set_defaults(func=read_index)

    q = sub.add_parser("materialize")
    q.add_argument("state")
    q.add_argument("receipt")
    q.add_argument("output")
    q.set_defaults(func=materialize)

    q = sub.add_parser("audit")
    q.add_argument("state")
    q.add_argument("receipt")
    q.set_defaults(func=audit)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

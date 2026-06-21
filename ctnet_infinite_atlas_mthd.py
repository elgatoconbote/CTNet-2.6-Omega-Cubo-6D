#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTNet InfiniteAtlas-MTHD core.

Nucleo de atlas virtual generativo para memoria MTHD:
- sin capacity
- sin slots
- sin route allocation
- sin route exhausted
- Omega fijo
- coordenadas de manifold generadas por clave
- capsulas Feistel reversibles para bytes arbitrarios exactos
- reglas procedurales para informacion recomputable
- fold reversible por recibo
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MASK64 = (1 << 64) - 1
MAGIC = "CTNET-MTHD-INFINITE-ATLAS-v1"


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s.encode("ascii"))


def sha256(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(len(p).to_bytes(8, "big"))
        h.update(p)
    return h.digest()


def hkdf_stream(key: bytes, context: bytes, n: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < n:
        out.extend(sha256(key, context, counter.to_bytes(8, "big")))
        counter += 1
    return bytes(out[:n])


def u64s_from_digest(d: bytes, count: int) -> List[int]:
    stream = hkdf_stream(d, b"u64", count * 8)
    return [struct.unpack(">Q", stream[i * 8 : (i + 1) * 8])[0] for i in range(count)]


def xor_bytes(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))


def pad16(data: bytes) -> bytes:
    pad_len = 16 - ((len(data) + 1) % 16)
    return data + b"\x80" + b"\x00" * pad_len


def unpad16(data: bytes) -> bytes:
    i = data.rfind(b"\x80")
    if i < 0 or any(x != 0 for x in data[i + 1 :]):
        raise ValueError("invalid capsule padding")
    return data[:i]


def feistel_round_func(r: bytes, round_key: bytes, round_no: int) -> bytes:
    return sha256(round_key, round_no.to_bytes(4, "big"), r)[:8]


def feistel_block(block16: bytes, round_key: bytes, *, decrypt: bool = False, rounds: int = 12) -> bytes:
    if len(block16) != 16:
        raise ValueError("Feistel block must be 16 bytes")
    l, r = block16[:8], block16[8:]
    if decrypt:
        for i in range(rounds - 1, -1, -1):
            f = feistel_round_func(l, round_key, i)
            l, r = xor_bytes(r, f), l
        return l + r
    for i in range(rounds):
        f = feistel_round_func(r, round_key, i)
        l, r = r, xor_bytes(l, f)
    return l + r


def feistel_codec(data: bytes, round_key: bytes, *, decrypt: bool = False) -> bytes:
    if not decrypt:
        data = pad16(data)
    if len(data) % 16 != 0:
        raise ValueError("ciphertext length must be multiple of 16")
    out = bytearray()
    for off in range(0, len(data), 16):
        out.extend(feistel_block(data[off : off + 16], round_key, decrypt=decrypt))
    raw = bytes(out)
    return unpad16(raw) if decrypt else raw


@dataclass
class ManifoldCoordinate:
    """Coordenada virtual: no reserva una ruta material."""

    key_tag: str
    q_seed: str
    chart: str
    first_levels: List[List[int]]
    formula: str = "level_k = HKDF(q_seed, k); no terminal level is allocated"


@dataclass
class Receipt:
    magic: str
    mode: str  # capsule | rule
    key_tag: str
    nonce: str
    coord: ManifoldCoordinate
    capsule: Optional[str] = None
    rule: Optional[Dict[str, Any]] = None
    tag: Optional[str] = None
    omega_digest: Optional[str] = None


class InfiniteAtlasMTHD:
    """Omega fijo + atlas virtual ilimitado por formula.

    Omega es una lista fija de palabras uint64. El fold por recibo es una
    involucion XOR: aplicar el mismo recibo dos veces recupera el estado previo.
    """

    def __init__(self, seed: str = "ctnet", omega_words: int = 256, omega: Optional[List[int]] = None):
        if omega_words <= 0:
            raise ValueError("omega_words must be positive")
        self.seed = str(seed)
        self.omega_words = int(omega_words)
        if omega is None:
            root = sha256(MAGIC.encode(), self.seed.encode())
            self.omega = u64s_from_digest(root, self.omega_words)
        else:
            if len(omega) != self.omega_words:
                raise ValueError("omega length does not match omega_words")
            self.omega = [int(x) & MASK64 for x in omega]

    @property
    def shape(self) -> Tuple[int]:
        return (self.omega_words,)

    def omega_bytes(self) -> bytes:
        return b"".join(struct.pack(">Q", x & MASK64) for x in self.omega)

    def digest(self) -> str:
        return _b64e(sha256(b"omega", self.omega_bytes()))

    def _secret(self) -> bytes:
        return sha256(b"secret", self.seed.encode(), self.omega_words.to_bytes(8, "big"))

    def coordinate(self, key: str, preview_levels: int = 8) -> ManifoldCoordinate:
        key_b = key.encode("utf-8")
        q_seed = sha256(b"q", self._secret(), key_b)
        key_tag = _b64e(sha256(b"key-tag", key_b)[:16])
        chart = _b64e(sha256(b"chart", q_seed)[:16])
        first = []
        for k in range(max(0, int(preview_levels))):
            d = sha256(b"level", q_seed, k.to_bytes(8, "big"))
            first.append(u64s_from_digest(d, 3))
        return ManifoldCoordinate(key_tag=key_tag, q_seed=_b64e(q_seed), chart=chart, first_levels=first)

    def _receipt_digest_words(self, receipt: Receipt) -> List[int]:
        raw = json.dumps(receipt_to_json(receipt, include_omega=False), sort_keys=True, separators=(",", ":")).encode()
        d = sha256(b"fold", self._secret(), raw)
        return u64s_from_digest(d, self.omega_words)

    def fold(self, receipt: Receipt) -> None:
        words = self._receipt_digest_words(receipt)
        self.omega = [(a ^ b) & MASK64 for a, b in zip(self.omega, words)]

    def put(self, key: str, data: bytes, *, fold: bool = True) -> Receipt:
        coord = self.coordinate(key)
        nonce = os.urandom(16)
        q_seed = _b64d(coord.q_seed)
        round_key = sha256(b"capsule-key", self._secret(), q_seed, nonce)
        cipher = feistel_codec(data, round_key, decrypt=False)
        tag = hmac.new(round_key, key.encode("utf-8") + cipher, hashlib.sha256).digest()
        receipt = Receipt(
            magic=MAGIC,
            mode="capsule",
            key_tag=coord.key_tag,
            nonce=_b64e(nonce),
            coord=coord,
            capsule=_b64e(cipher),
            tag=_b64e(tag),
        )
        if fold:
            self.fold(receipt)
            receipt.omega_digest = self.digest()
        return receipt

    def get(self, key: str, receipt: Receipt) -> bytes:
        if receipt.magic != MAGIC:
            raise ValueError("bad receipt magic")
        expected = self.coordinate(key, preview_levels=len(receipt.coord.first_levels)).key_tag
        if expected != receipt.key_tag:
            raise ValueError("receipt does not belong to this key")
        if receipt.mode == "capsule":
            if receipt.capsule is None or receipt.tag is None:
                raise ValueError("capsule receipt missing fields")
            q_seed = _b64d(receipt.coord.q_seed)
            nonce = _b64d(receipt.nonce)
            cipher = _b64d(receipt.capsule)
            round_key = sha256(b"capsule-key", self._secret(), q_seed, nonce)
            tag = hmac.new(round_key, key.encode("utf-8") + cipher, hashlib.sha256).digest()
            if not hmac.compare_digest(tag, _b64d(receipt.tag)):
                raise ValueError("receipt authentication failed")
            return feistel_codec(cipher, round_key, decrypt=True)
        if receipt.mode == "rule":
            return self._eval_rule(key, receipt.rule or {})
        raise ValueError(f"unknown receipt mode: {receipt.mode}")

    def fold_rule(self, key: str, rule: Dict[str, Any], *, fold: bool = True) -> Receipt:
        coord = self.coordinate(key)
        nonce = os.urandom(16)
        raw = json.dumps(rule, sort_keys=True, separators=(",", ":")).encode()
        rule_tag = hmac.new(self._secret(), key.encode() + raw, hashlib.sha256).digest()
        receipt = Receipt(
            magic=MAGIC,
            mode="rule",
            key_tag=coord.key_tag,
            nonce=_b64e(nonce),
            coord=coord,
            rule=rule,
            tag=_b64e(rule_tag),
        )
        if fold:
            self.fold(receipt)
            receipt.omega_digest = self.digest()
        return receipt

    def _eval_rule(self, key: str, rule: Dict[str, Any]) -> bytes:
        kind = rule.get("kind")
        if kind == "repeat":
            text = str(rule.get("text", "")).encode("utf-8")
            n = int(rule.get("n", 1))
            if n < 0:
                raise ValueError("negative repeat")
            return text * n
        if kind == "prng":
            n = int(rule.get("n", 0))
            if n < 0:
                raise ValueError("negative n")
            context = json.dumps(rule, sort_keys=True).encode() + key.encode()
            return hkdf_stream(self._secret(), context, n)
        raise ValueError("unsupported rule")

    def audit(self) -> Dict[str, Any]:
        before = self.digest()
        r = self.put("__audit__", b"ctnet-mthd-audit", fold=True)
        after_put = self.digest()
        self.fold(r)
        after_undo = self.digest()
        recovered = self.get("__audit__", r)
        return {
            "magic": MAGIC,
            "shape": list(self.shape),
            "has_capacity": False,
            "has_slots": False,
            "has_route_exhaustion": False,
            "omega_digest_before": before,
            "omega_digest_after_put": after_put,
            "omega_digest_after_undo": after_undo,
            "reversible_fold_ok": before == after_undo,
            "direct_read_ok": recovered == b"ctnet-mthd-audit",
            "virtual_atlas": "unbounded formulaic coordinates; no route allocation",
        }

    def save(self, path: str | Path) -> None:
        obj = {"magic": MAGIC, "seed": self.seed, "omega_words": self.omega_words, "omega": self.omega}
        Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "InfiniteAtlasMTHD":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if obj.get("magic") != MAGIC:
            raise ValueError("bad state magic")
        return cls(seed=obj["seed"], omega_words=int(obj["omega_words"]), omega=obj["omega"])


def receipt_to_json(r: Receipt, *, include_omega: bool = True) -> Dict[str, Any]:
    d = asdict(r)
    if not include_omega:
        d.pop("omega_digest", None)
    return d


def receipt_from_json(d: Dict[str, Any]) -> Receipt:
    coord = ManifoldCoordinate(**d["coord"])
    return Receipt(
        magic=d["magic"],
        mode=d["mode"],
        key_tag=d["key_tag"],
        nonce=d["nonce"],
        coord=coord,
        capsule=d.get("capsule"),
        rule=d.get("rule"),
        tag=d.get("tag"),
        omega_digest=d.get("omega_digest"),
    )

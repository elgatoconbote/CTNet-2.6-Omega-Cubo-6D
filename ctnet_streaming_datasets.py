#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Online streaming datasets for CTNet 2.6 Omega + Cubo 6D.

This module deliberately uses Hugging Face `load_dataset(..., streaming=True)`.
It does not materialize the corpus locally. The only local files expected during
training are checkpoints, reports and normal tiny dataset metadata/cache files.

Regime streams:
  - fineweb_edu      : general educational world modeling
  - openwebmath      : mathematical web structure
  - numina_math_cot  : real/non-synthetic math solution paths, filtered by source
  - swe_bench        : code-patch/test closure

Every yielded sample is normalized to a common CTNet record:

    {
      "x": source condition / prompt / problem,
      "y": target text / solution / patch,
      "regime": regime name,
      "loss_mode": external training interface,
      "source": dataset source,
      "meta": optional metadata
    }
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional

try:
    from datasets import load_dataset, disable_caching
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency `datasets`. Install with: pip install -r requirements.txt"
    ) from exc


DEFAULT_NUMINA_ALLOWED_SOURCES = {
    "aops_forum",
    "amc_aime",
    "cn_k12",
    "gsm8k",
    "math",
    "olympiads",
}


@dataclass(frozen=True)
class StreamMixConfig:
    fineweb_prob: float = 0.55
    openwebmath_prob: float = 0.25
    numina_prob: float = 0.15
    swe_prob: float = 0.05
    seed: int = 42
    shuffle_buffer: int = 0
    fineweb_name: str = "sample-10BT"
    use_fineweb: bool = True
    use_openwebmath: bool = True
    use_numina: bool = True
    use_swe: bool = True


def _safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _load_stream(path: str, *, name: Optional[str] = None, split: str = "train"):
    if name:
        ds = load_dataset(path, name, split=split, streaming=True)
    else:
        ds = load_dataset(path, split=split, streaming=True)
    return ds


def fineweb_edu_stream(*, name: str = "sample-10BT") -> Iterable[Dict]:
    ds = _load_stream("HuggingFaceFW/fineweb-edu", name=name, split="train")
    for ex in ds:
        text = _safe_text(ex.get("text", ""))
        if not text.strip():
            continue
        yield {
            "x": text,
            "y": text,
            "regime": "educational_world_modeling",
            "loss_mode": "next_token_with_coherence",
            "source": "HuggingFaceFW/fineweb-edu",
            "meta": {
                "token_count": ex.get("token_count"),
                "score": ex.get("score"),
                "int_score": ex.get("int_score"),
                "language_score": ex.get("language_score"),
                "url": ex.get("url"),
            },
        }


def openwebmath_stream() -> Iterable[Dict]:
    ds = _load_stream("open-web-math/open-web-math", split="train")
    for ex in ds:
        text = _safe_text(ex.get("text", ""))
        if not text.strip():
            continue
        yield {
            "x": text,
            "y": text,
            "regime": "math_web_structure",
            "loss_mode": "next_token_with_symbolic_coherence",
            "source": "open-web-math/open-web-math",
            "meta": {"url": ex.get("url")},
        }


def numina_math_cot_stream(
    *,
    allowed_sources: Optional[set[str]] = None,
) -> Iterable[Dict]:
    allowed = allowed_sources or DEFAULT_NUMINA_ALLOWED_SOURCES
    ds = _load_stream("AI-MO/NuminaMath-CoT", split="train")
    for ex in ds:
        src = _safe_text(ex.get("source", ""))
        if src not in allowed:
            continue
        problem = _safe_text(ex.get("problem", ""))
        solution = _safe_text(ex.get("solution", ""))
        if not problem.strip() or not solution.strip():
            continue
        yield {
            "x": problem,
            "y": solution,
            "regime": "math_closure_cot",
            "loss_mode": "solution_path_with_closure",
            "source": "AI-MO/NuminaMath-CoT",
            "meta": {"source": src},
        }


def swe_bench_stream() -> Iterable[Dict]:
    ds = _load_stream("princeton-nlp/SWE-bench", split="train")
    for ex in ds:
        problem = _safe_text(ex.get("problem_statement", ""))
        patch = _safe_text(ex.get("patch", ""))
        if not problem.strip() or not patch.strip():
            continue
        yield {
            "x": problem,
            "y": patch,
            "regime": "code_patch_test_closure",
            "loss_mode": "patch_with_test_closure",
            "source": "princeton-nlp/SWE-bench",
            "meta": {
                "repo": ex.get("repo"),
                "instance_id": ex.get("instance_id"),
                "base_commit": ex.get("base_commit"),
                "fail_to_pass": ex.get("FAIL_TO_PASS"),
                "pass_to_pass": ex.get("PASS_TO_PASS"),
            },
        }


class OnlineRegimeMixture:
    """Probability mixture over remote iterable streams.

    It samples a stream name on every step and returns one online record from the
    selected iterator. If a finite stream is exhausted, that iterator is rebuilt.
    """

    def __init__(self, config: StreamMixConfig):
        disable_caching()
        self.config = config
        self.rng = random.Random(config.seed)
        self.builders = {}
        self.weights = {}

        if config.use_fineweb and config.fineweb_prob > 0:
            self.builders["fineweb_edu"] = lambda: iter(fineweb_edu_stream(name=config.fineweb_name))
            self.weights["fineweb_edu"] = float(config.fineweb_prob)

        if config.use_openwebmath and config.openwebmath_prob > 0:
            self.builders["openwebmath"] = lambda: iter(openwebmath_stream())
            self.weights["openwebmath"] = float(config.openwebmath_prob)

        if config.use_numina and config.numina_prob > 0:
            self.builders["numina_math_cot"] = lambda: iter(numina_math_cot_stream())
            self.weights["numina_math_cot"] = float(config.numina_prob)

        if config.use_swe and config.swe_prob > 0:
            self.builders["swe_bench"] = lambda: iter(swe_bench_stream())
            self.weights["swe_bench"] = float(config.swe_prob)

        if not self.builders:
            raise ValueError("No online streams enabled.")

        total = sum(self.weights.values())
        self.names = list(self.builders.keys())
        self.probs = [self.weights[n] / total for n in self.names]
        self.iters = {name: self.builders[name]() for name in self.names}

    def __iter__(self) -> "OnlineRegimeMixture":
        return self

    def __next__(self) -> Dict:
        while True:
            name = self.rng.choices(self.names, weights=self.probs, k=1)[0]
            try:
                item = next(self.iters[name])
            except StopIteration:
                self.iters[name] = self.builders[name]()
                item = next(self.iters[name])
            item["mixture_name"] = name
            return item


def make_online_regime_stream(config: StreamMixConfig) -> Iterator[Dict]:
    return iter(OnlineRegimeMixture(config))


def preview(n: int = 5, *, seed: int = 42) -> List[Dict]:
    stream = make_online_regime_stream(StreamMixConfig(seed=seed))
    rows = []
    for _ in range(n):
        ex = next(stream)
        rows.append(
            {
                "mixture_name": ex["mixture_name"],
                "source": ex["source"],
                "regime": ex["regime"],
                "loss_mode": ex["loss_mode"],
                "x_head": ex["x"][:160],
                "y_head": ex["y"][:160],
            }
        )
    return rows

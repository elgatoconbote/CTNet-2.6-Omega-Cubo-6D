#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAM-limited launcher for CTNet online streaming.

This file sets Hugging Face / datasets / torch / temp cache paths to tmpfs
before importing the training module. It prevents persistent disk cache while
leaving CTNet tensors on GPU when --cuda is used.

Important distinction:
- CTNet state and memory remain fixed-size during training.
- External libraries may cache network metadata/chunks unless forced into tmpfs.

Usage:
    /path/to/venv/bin/python train_ram_streaming_ctnet.py --steps 1000 --batch 1 --cuda --save-every 0

Default output goes to /dev/shm/ctnet_ram_stream, which is tmpfs/RAM on Linux.
The trainer version currently always writes a final checkpoint; with this
launcher that checkpoint is written to /dev/shm, not to persistent disk.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    ram_root = Path(os.environ.get("CTNET_RAM_ROOT", "/dev/shm/ctnet_ram_stream"))
    ram_root.mkdir(parents=True, exist_ok=True)

    paths = {
        "HF_HOME": ram_root / "hf_home",
        "HF_DATASETS_CACHE": ram_root / "hf_datasets",
        "HUGGINGFACE_HUB_CACHE": ram_root / "hf_hub",
        "HF_HUB_CACHE": ram_root / "hf_hub",
        "XDG_CACHE_HOME": ram_root / "xdg_cache",
        "TORCH_HOME": ram_root / "torch",
        "TMPDIR": ram_root / "tmp",
        "TEMP": ram_root / "tmp",
        "TMP": ram_root / "tmp",
    }
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    os.environ.setdefault("PYTHONMALLOC", "malloc")

    argv = ["train_streaming_ctnet.py"] + sys.argv[1:]
    if "--out-dir" not in argv:
        argv += ["--out-dir", str(ram_root / "runs")]
    if "--save-every" not in argv:
        argv += ["--save-every", "0"]

    print("CTNet RAM-limited streaming launcher")
    print("RAM root:", ram_root)
    print("Python:", sys.executable)
    print("HF_HOME:", os.environ["HF_HOME"])
    print("HF_DATASETS_CACHE:", os.environ["HF_DATASETS_CACHE"])
    print("HUGGINGFACE_HUB_CACHE:", os.environ["HUGGINGFACE_HUB_CACHE"])
    print("TMPDIR:", os.environ["TMPDIR"])
    print("argv:", " ".join(argv))

    sys.argv = argv
    runpy.run_module("train_streaming_ctnet", run_name="__main__")


if __name__ == "__main__":
    main()

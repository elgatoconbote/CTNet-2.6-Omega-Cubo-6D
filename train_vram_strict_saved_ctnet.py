#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train CTNet strict zero-disk input mode and save the final trained deformation.

This is the practical default launcher:
- no Hugging Face datasets / hub / pyarrow / xet,
- no corpus cache,
- no intermediate checkpoints,
- final CTNet deformation is saved to disk by default.

The saved file is the trained CTNet state_dict plus small metadata. It is not a
cache, not a corpus, and not a growing external memory.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    argv = ["train_vram_strict_ctnet.py"] + sys.argv[1:]

    if "--save-final" not in argv:
        default_path = os.environ.get("CTNET_SAVE_FINAL", "checkpoints/ctnet_state_final.pt")
        Path(default_path).parent.mkdir(parents=True, exist_ok=True)
        argv += ["--save-final", default_path]

    if "--log-every" not in argv:
        argv += ["--log-every", "10"]

    print("CTNet strict trainer with final deformation save")
    print("argv:", " ".join(argv))

    sys.argv = argv
    runpy.run_module("train_vram_strict_ctnet", run_name="__main__")


if __name__ == "__main__":
    main()

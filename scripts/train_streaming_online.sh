#!/usr/bin/env bash
set -euo pipefail

python train_streaming_ctnet.py \
  --steps "${STEPS:-1000}" \
  --batch "${BATCH:-2}" \
  --out-dir "${OUT_DIR:-runs/online_stream}" \
  --coherence-grad-scale

#!/usr/bin/env bash
# One GPU, 345 frames (14.38 s). group-offload on -- the transformer's 50 blocks
# stream in/out of the card in groups of 1, keeping the resident footprint to
# ~0.72 GB. Denoising peak is ~20 GB so the total fits on 80 GB.
# Encoder shares the card and is offloaded to host RAM after conditioning.
# If OOM mid-layer: FRAMES=243 (~14 GB), or PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/_common.sh

GPU=${GPU:-0}
FRAMES=${FRAMES:-345}
GO=${GO:-1}
CHUNK=${CHUNK:-4096}
OUT_DIR=${OUT_DIR:-runs/1gpu_15s}

mkdir -p "${OUT_DIR}"
echo "1 GPU (cuda:${GPU})  group-offload=${GO}  ${FRAMES} frames  topk ${TOPK}  ${IR}[${INDEX}]"

CUDA_VISIBLE_DEVICES="${GPU}" python -m mcsparse.pipeline \
  --prompt-file "${IR}" --index "${INDEX}" \
  --layout single --encoder-device shared \
  --num-frames "${FRAMES}" --num-steps "${STEPS}" --topk "${TOPK}" \
  --calibrate-steps "${CALIBRATE}" \
  --dense-steps "${DENSE_STEPS}" --dense-layers "${DENSE_LAYERS}" \
  --sink-tokens "${SINK}" --seed "${SEED}" \
  --group-offload "${GO}" \
  --chunk-selector --chunk-selector-size "${CHUNK}" \
  --out "${OUT_DIR}/out.mp4" --json-out "${OUT_DIR}/result.json" \
  ${EXTRA}

report "${OUT_DIR}/result.json"

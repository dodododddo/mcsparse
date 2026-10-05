#!/usr/bin/env bash
# One GPU, 124 frames (5.17 s). No group-offload -- transformer weights stay
# resident (~35.9 GB). Denoising peak is ~7 GB so the total fits on 80 GB.
# Encoder shares the card and is offloaded to host RAM after conditioning.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/_common.sh

GPU=${GPU:-0}
FRAMES=${FRAMES:-124}
CHUNK=${CHUNK:-4096}
OUT_DIR=${OUT_DIR:-runs/1gpu_5s}

mkdir -p "${OUT_DIR}"
echo "1 GPU (cuda:${GPU})  ${FRAMES} frames  topk ${TOPK}  ${IR}[${INDEX}]"

CUDA_VISIBLE_DEVICES="${GPU}" python -m mcsparse.pipeline \
  --prompt-file "${IR}" --index "${INDEX}" \
  --layout single --encoder-device shared \
  --num-frames "${FRAMES}" --num-steps "${STEPS}" --topk "${TOPK}" \
  --calibrate-steps "${CALIBRATE}" \
  --dense-steps "${DENSE_STEPS}" --dense-layers "${DENSE_LAYERS}" \
  --sink-tokens "${SINK}" --seed "${SEED}" \
  --chunk-selector --chunk-selector-size "${CHUNK}" \
  --out "${OUT_DIR}/out.mp4" --json-out "${OUT_DIR}/result.json" \
  ${EXTRA}

report "${OUT_DIR}/result.json"

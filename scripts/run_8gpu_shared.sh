#!/usr/bin/env bash
# 8 ranks of CP, 345 frames (14.38 s). All 8 ranks denoise; the encoder shares
# cuda:0 and is offloaded to host RAM after conditioning. No dedicated encoder
# card -- one fewer GPU needed than the 7+1 layout.
#
#   ./scripts/run_8gpu_shared.sh
#   NPROC=4 ./scripts/run_8gpu_shared.sh
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/_common.sh

NPROC=${NPROC:-8}
FRAMES=${FRAMES:-345}
TOPKS=${TOPKS:-0.15}
OUT_DIR=${OUT_DIR:-runs/8gpu_shared}

(( 56 % NPROC == 0 )) || { echo "NPROC must divide 56: 2/4/7/8" >&2; exit 1; }
mkdir -p "${OUT_DIR}"
echo "${NPROC} CP ranks  encoder shared on cuda:0, offloaded after conditioning  ${FRAMES} frames  ${IR}[${INDEX}]"

run_arm () {
  local tag=$1 topk=$2
  [[ -f ${OUT_DIR}/${tag}.json ]] && { echo "${tag}: exists, skipping"; return; }
  echo "### ${tag}"
  torchrun --nproc_per_node="${NPROC}" -m mcsparse.pipeline \
    --prompt-file "${IR}" --index "${INDEX}" \
    --attention token_sparse --topk "${topk}" \
    --num-frames "${FRAMES}" --num-steps "${STEPS}" \
    --calibrate-steps "${CALIBRATE}" \
    --dense-steps "${DENSE_STEPS}" --dense-layers "${DENSE_LAYERS}" \
    --sink-tokens "${SINK}" --seed "${SEED}" \
    --encoder-device shared \
    --out "${OUT_DIR}/${tag}.mp4" --json-out "${OUT_DIR}/${tag}.json" \
    ${EXTRA} 2>&1 | tee "${OUT_DIR}/${tag}.log" | grep -E "^(mcsparse|loaded|generating|  )" || true
}

for topk in ${TOPKS}; do
  run_arm "topk$(printf '%03d' "$(python3 -c "print(round(float('${topk}')*100))")")" "${topk}"
done
for p in "${OUT_DIR}"/topk*.json; do [[ -f $p ]] && report "$p"; done

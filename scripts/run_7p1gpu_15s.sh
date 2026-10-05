#!/usr/bin/env bash
# 7 CP ranks (cuda:0..6) + encoder resident on cuda:7 (stays up the whole run),
# 345 frames (14.38 s), 49 steps, topk 0.15, calibrate at step 9.
#
#   ./scripts/run_8gpu_15s.sh
#   TOPKS="0.15 0.25" ./scripts/run_8gpu_15s.sh
#   WITH_DENSE=1 ./scripts/run_8gpu_15s.sh     # dense reference, for PSNR
#   NPROC=4 ./scripts/run_8gpu_15s.sh          # on a smaller node
#
# No memory knobs here -- 7 ranks do not need them, and the reference
# implementation had none. This is the script to reproduce published numbers.
#
# NPROC must divide 56 (H3's head count): 2, 4, 7 or 8. It needs NPROC+1 GPUs.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/_common.sh

NPROC=${NPROC:-7}
FRAMES=${FRAMES:-345}
TOPKS=${TOPKS:-0.15}
OUT_DIR=${OUT_DIR:-runs/7p1gpu_15s}

(( 56 % NPROC == 0 )) || { echo "NPROC must divide 56: 2/4/7/8" >&2; exit 1; }

mkdir -p "${OUT_DIR}"
echo "${NPROC} CP ranks + encoder on cuda:${NPROC}   ${FRAMES} frames   ${IR}[${INDEX}]"

run_arm () {
  local tag=$1 attention=$2 topk=$3
  [[ -f ${OUT_DIR}/${tag}.json ]] && { echo "${tag}: exists, skipping"; return; }
  echo "### ${tag}"

  local topk_args=()
  [[ ${attention} == token_sparse ]] && topk_args=(--topk "${topk}")

  torchrun --nproc_per_node="${NPROC}" -m mcsparse.pipeline \
    --prompt-file "${IR}" --index "${INDEX}" \
    --attention "${attention}" "${topk_args[@]}" \
    --num-frames "${FRAMES}" --num-steps "${STEPS}" \
    --calibrate-steps "${CALIBRATE}" \
    --dense-steps "${DENSE_STEPS}" --dense-layers "${DENSE_LAYERS}" \
    --sink-tokens "${SINK}" --seed "${SEED}" \
    --encoder-device "cuda:${NPROC}" \
    --out "${OUT_DIR}/${tag}.mp4" --json-out "${OUT_DIR}/${tag}.json" \
    ${EXTRA} 2>&1 | tee "${OUT_DIR}/${tag}.log" | grep -E \
      "^(mcsparse|loading|loaded|conditioning|generating|  )" || true
}

[[ ${WITH_DENSE:-0} == 1 ]] && run_arm dense dense 0
for topk in ${TOPKS}; do
  run_arm "topk$(printf '%03d' "$(python3 -c "print(round(float('${topk}')*100))")")" \
          token_sparse "${topk}"
done

echo
python3 - "${OUT_DIR}" <<'PY'
import glob, json, os, sys
js = sorted(glob.glob(os.path.join(sys.argv[1], "*.json")))
if not js:
    raise SystemExit(0)
dense = next((json.load(open(p, encoding="utf-8")) for p in js
              if os.path.basename(p) == "dense.json"), None)

print(f"{'arm':10} {'end2end':>9} {'transformer':>12}"
      + (f" {'speedup':>8}" if dense else "")
      + f" {'peak_GB':>8} {'kernel_calls':>12}")
for p in js:
    r = json.load(open(p, encoding="utf-8"))
    b = r.get("backend") or {}
    row = f"{os.path.basename(p)[:-5]:10} {r['seconds']:9.1f} {r['dit_seconds']:12.1f}"
    if dense:
        row += f" {dense['dit_seconds'] / r['dit_seconds']:7.2f}x"
    print(row + f" {r['peak_gb']:8.1f} {str(b.get('kernel_calls')):>12}")

print("\nCompare the transformer column. End-to-end includes the VAE decode, "
      "which\nis the same work in every arm and pulls every speedup toward 1.0x.")
PY

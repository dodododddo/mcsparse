#!/usr/bin/env bash
# Sourced by the run scripts. Defaults are the published config.

# Point these at your checkout if they are not already in the environment.
# export H3_MODEL=/path/to/MiniMax-H3
# export HF_HOME=/path/to/cache/huggingface
# export TRITON_CACHE_DIR=/path/to/cache/triton   # CuteDSL + Triton JIT

IR=${IR:-examples/prompts.jsonl}
INDEX=${INDEX:-0}
TOPK=${TOPK:-0.15}
STEPS=${STEPS:-49}
CALIBRATE=${CALIBRATE:-9}
DENSE_STEPS=${DENSE_STEPS:-10}
DENSE_LAYERS=${DENSE_LAYERS:-0}
SINK=${SINK:-0}
SEED=${SEED:-42}
EXTRA=${EXTRA:-}

report () {
  python3 - "$1" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
b = r.get("backend") or {}
print(f"\n{r['seconds']:.0f}s total  {r['dit_seconds']:.0f}s transformer  "
      f"{r['vae_seconds']:.0f}s vae   peak {r['peak_gb']:.1f} GB")
print(f"kernel_calls={b.get('kernel_calls')} fallback={b.get('fallback_calls')} "
      f"seq_len={b.get('seq_len')} effective_density={b.get('effective_density')}")
if not b.get("kernel_calls"):
    print("WARNING: the sparse kernel never ran -- this is a dense measurement.")
PY
}

#!/usr/bin/env python3
"""Gather-KV kernel cost, against full attention on the same tensors.

    python tests/bench_token_sparse.py
    python tests/bench_token_sparse.py --topk 0.15,0.25
    python tests/bench_token_sparse.py --seq 32768 --heads 12

Measures flash_attn_gather_kv_func alone -- not selection, which runs once at
the calibrate step and is amortised over every reuse step.

The ratio is a ceiling on the attention speedup: end to end also pays for
selection, the PCA permute/unpermute, and the BTHD<->BHSD layout change.

Gather indices are random here, which is the worst case for locality; real
selections cluster. First call includes the CuteDSL JIT, hence --warmup.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Defaults: what one rank sees at 345 frames under 7-way CP.
    ap.add_argument("--heads", type=int, default=8, help="local heads (56/world)")
    ap.add_argument("--seq", type=int, default=104267)
    ap.add_argument("--dim", type=int, default=128, choices=(128, 144))
    ap.add_argument("--blkq", type=int, default=128)
    ap.add_argument("--tile-n", type=int, default=128)
    ap.add_argument("--topk", default="0.15", help="comma-separated")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=("bfloat16", "float16"))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--no-attn", action="store_true")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("needs torch", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("needs CUDA", file=sys.stderr)
        return 1
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 9:
        print(f"the kernel targets sm90+, this is sm{cap[0]}{cap[1]}",
              file=sys.stderr)
        return 1

    from mcsparse.kernels.flash_mcsa.ops.flash_sparse_attn.cute.interface import (
        flash_attn_gather_kv_func)

    dev = torch.device("cuda")
    dt = getattr(torch, args.dtype)
    H, L, D, C, TN = args.heads, args.seq, args.dim, args.blkq, args.tile_n
    M = (L + C - 1) // C
    esz = torch.tensor([], dtype=dt).element_size()

    print(f"H={H} L={L} D={D} BLKQ={C} tile_n={TN} {args.dtype}   "
          f"M={M} q blocks   one (1,L,H,D) = {L * H * D * esz / 2**30:.2f} GB\n")

    g = torch.Generator(device=dev).manual_seed(1234)
    # The kernel takes BSHD.
    q, k, v = (torch.randn(1, L, H, D, device=dev, dtype=torch.float32,
                           generator=g).to(dt) for _ in range(3))
    scale = 1.0 / (D ** 0.5)

    def timed(fn):
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated(dev)
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(dev)
        ts = []
        for _ in range(args.reps):
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            fn()
            t1.record()
            torch.cuda.synchronize()
            ts.append(t0.elapsed_time(t1))
        ts.sort()
        return ts[len(ts) // 2], (torch.cuda.max_memory_allocated(dev) - base) / 2**30

    attn_ms = None
    if not args.no_attn:
        try:
            from mcsparse.h3.dense import fa3_available, fa3_dense
            ok, why = fa3_available()
            if ok:
                attn_ms, gb = timed(lambda: fa3_dense(q, k, v))
                print(f"{'full attn (FA3)':20} {attn_ms:9.2f} ms {gb:8.2f} GB")
            else:
                qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))
                attn_ms, gb = timed(
                    lambda: torch.nn.functional.scaled_dot_product_attention(
                        qt, kt, vt))
                print(f"{'full attn (SDPA)':20} {attn_ms:9.2f} ms {gb:8.2f} GB"
                      f"   no FA3: {why[:36]}")
        except torch.cuda.OutOfMemoryError:
            print(f"{'full attn':20} OOM at L={L}")
            torch.cuda.empty_cache()

    print(f"\n{'topk':>6} {'gather_len':>11} {'kernel_ms':>10} {'peak_GB':>9} "
          f"{'vs full':>9}")
    for tok in args.topk.split(","):
        topk = float(tok.strip())
        # Same arithmetic as _select_topk_tokens: align the budget to tile_n.
        budget = ((max(1, int(topk * L)) + TN - 1) // TN) * TN
        gather_len = min(budget, (L // TN) * TN)
        gi = torch.randint(0, L, (1, H, M, gather_len), device=dev,
                           dtype=torch.int32, generator=g)
        try:
            ms, gb = timed(lambda: flash_attn_gather_kv_func(
                q, k, v, gi, softmax_scale=scale, return_lse=False,
                gather_kv_lengths=None))
            ratio = f"{ms / attn_ms * 100:8.2f}%" if attn_ms else "       -"
            print(f"{topk:6.2f} {gather_len:11} {ms:10.2f} {gb:9.2f} {ratio}")
        except torch.cuda.OutOfMemoryError:
            print(f"{topk:6.2f} {gather_len:11}  OOM")
            torch.cuda.empty_cache()
        del gi
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

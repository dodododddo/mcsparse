#!/usr/bin/env python3
"""Fast PDDP grouping cost, against full attention on the same tensors.

    python tests/bench_pca_group.py
    python tests/bench_pca_group.py --proj-dim 64,32
    python tests/bench_pca_group.py --seq 32768 --heads 12

Reports single-call and amortised cost: grouping runs once at the calibrate
step and the permutation is reused by every reuse step after it (39 of 49 by
default, hence --reuse-steps).

--proj-dim changes the result -- it drops low-variance directions -- so
leaf_inertia is reported alongside the speedup. Lower inertia is tighter
clustering, which is what the reordering optimises.
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
    # Defaults: H3 at 345 frames, all 56 heads (grouping is per-head and runs
    # before the all-to-all in the single-GPU case).
    ap.add_argument("--heads", type=int, default=56)
    ap.add_argument("--seq", type=int, default=104267)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--blkq", type=int, default=128)
    ap.add_argument("--pow-iters", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=("bfloat16", "float16", "float32"))
    ap.add_argument("--proj-dim", default="", help="comma-separated, e.g. 64,32")
    ap.add_argument("--reuse-steps", type=int, default=39,
                    help="reuse steps one permutation serves")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--no-attn", action="store_true")
    ap.add_argument("--no-inertia", action="store_true",
                    help="skip it; needs an fp32 copy (2.8 GB at H3's shape)")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("needs torch", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("needs CUDA", file=sys.stderr)
        return 1

    from mcsparse.kernels.flash_mcsa.pca_group import (
        identity_perm, leaf_inertia, pca_group_perm)

    dev = torch.device("cuda")
    dt = getattr(torch, args.dtype)
    BH, N, D, C = args.heads, args.seq, args.dim, args.blkq
    K = (N + C - 1) // C
    esz = torch.tensor([], dtype=dt).element_size()

    print(f"BH={BH} N={N} D={D} C={C} {args.dtype}   K={K} blocks, "
          f"~{max(K - 1, 1).bit_length()} levels   "
          f"one (BH,N,D) = {BH * N * D * esz / 2**30:.2f} GB\n")

    g = torch.Generator(device=dev).manual_seed(1234)
    x = torch.randn(BH, N, D, device=dev, dtype=torch.float32,
                    generator=g).to(dt)

    def timed(fn):
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated(dev)
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(dev)
        ts, out = [], None
        for _ in range(args.reps):
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            out = fn()
            t1.record()
            torch.cuda.synchronize()
            ts.append(t0.elapsed_time(t1))
        ts.sort()
        peak = (torch.cuda.max_memory_allocated(dev) - base) / 2**30
        return ts[len(ts) // 2], peak, out

    arms = [("full D", None)]
    arms += [(f"proj_dim={t.strip()}", int(t))
             for t in args.proj_dim.split(",") if t.strip()]

    print(f"{'arm':16} {'ms':>9} {'peak_GB':>9}")
    res = {}
    for name, rd in arms:
        if rd is not None and rd >= D:
            print(f"{name:16} skipped (>= D)")
            continue
        try:
            ms, gb, perm = timed(lambda rd=rd: pca_group_perm(
                x, C=C, pow_iters=args.pow_iters, proj_dim=rd))
            res[name] = {"ms": ms, "gb": gb, "perm": perm}
            print(f"{name:16} {ms:9.1f} {gb:9.2f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{name:16} OOM")
            torch.cuda.empty_cache()

    attn_ms = None
    if not args.no_attn:
        try:
            q = x.unsqueeze(1)
            attn_ms, gb, _ = timed(
                lambda: torch.nn.functional.scaled_dot_product_attention(q, q, q))
            print(f"{'full attention':16} {attn_ms:9.1f} {gb:9.2f}")
            del q
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"{'full attention':16} OOM at N={N}")
            torch.cuda.empty_cache()

    if attn_ms:
        print(f"\nvs full attention ({attn_ms:.1f} ms)")
        for name, r in res.items():
            amort = r["ms"] / max(args.reuse_steps, 1)
            print(f"  {name:14} single {r['ms'] / attn_ms * 100:6.2f}%   "
                  f"amortised over {args.reuse_steps} steps "
                  f"{amort / attn_ms * 100:6.3f}%")

    if "full D" in res and len(res) > 1:
        base_ms = res["full D"]["ms"]
        print()
        for name, r in res.items():
            if name != "full D":
                print(f"  {name:14} {base_ms / r['ms']:.2f}x faster")

    if not args.no_inertia and res:
        try:
            i0 = leaf_inertia(x, identity_perm(BH, N, dev), C)
            print(f"\nleaf_inertia (identity = {i0:.4f})")
            base_in = None
            for name, r in res.items():
                ii = leaf_inertia(x, r["perm"], C)
                if name == "full D":
                    base_in = ii
                tail = (f"   {ii / base_in:.4f}x vs full D"
                        if base_in and name != "full D" else "")
                print(f"  {name:14} {ii:9.4f}  ({ii / i0:.4f}x identity){tail}")
        except torch.cuda.OutOfMemoryError:
            print("\nleaf_inertia OOM -- pass --no-inertia")
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

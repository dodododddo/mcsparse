"""Optional stage-level CUDA timing.

Enable with FLASH_MCSA_TIMING=1. Use timed("mode.stage") around a stage and
wall() around a complete operation. Disabled timing uses a no-op context and
does not create CUDA events or synchronize the device.

Each enabled stage synchronizes separately, which can reduce overlap and make
summed stage times exceed wall time. Report wall time alongside stage totals.
FLASH_MCSA_TIMING_OUT optionally saves a per-rank JSON report at process exit.
"""

import os
import json
from collections import defaultdict
from contextlib import contextmanager

import torch

ENABLED = os.getenv("FLASH_MCSA_TIMING", "0").lower() in ("1", "true", "yes", "on")

# stage -> [total milliseconds, call count]
_acc = defaultdict(lambda: [0.0, 0])
_wall = [0.0, 0]          # Accumulated end-to-end wall time


@contextmanager
def _noop():
    yield


@contextmanager
def _timed(stage):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    try:
        yield
    finally:
        e.record()
        torch.cuda.synchronize()
        rec = _acc[stage]
        rec[0] += s.elapsed_time(e)
        rec[1] += 1


def timed(stage):
    """Time a CUDA stage, or return a no-op context when timing is disabled."""
    return _timed(stage) if ENABLED else _noop()


@contextmanager
def wall():
    """Measure end-to-end wall time to compare with synchronized stage totals."""
    if not ENABLED:
        yield
        return
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    try:
        yield
    finally:
        e.record()
        torch.cuda.synchronize()
        _wall[0] += s.elapsed_time(e)
        _wall[1] += 1


def reset():
    _acc.clear()
    _wall[0] = 0.0
    _wall[1] = 0


def get():
    """Return per-stage timing and call counts, plus the __wall__ entry."""
    out = {}
    for stage, (ms, n) in sorted(_acc.items()):
        out[stage] = {"ms": ms, "calls": n, "ms_per_call": ms / max(n, 1)}
    if _wall[1]:
        out["__wall__"] = {"ms": _wall[0], "calls": _wall[1],
                           "ms_per_call": _wall[0] / _wall[1]}
    return out


def summary(title="FlashMCSA timing"):
    """Print per-stage timing and totals grouped by mode."""
    if not _acc:
        print(f"[{title}] No timing data. Enable FLASH_MCSA_TIMING=1.")
        return
    data = get()
    total = sum(v["ms"] for k, v in data.items() if k != "__wall__")
    print(f"\n=== {title} ===")
    print(f"{'stage':<34}{'ms':>12}{'calls':>8}{'ms/call':>10}{'share':>8}")
    by_mode = defaultdict(float)
    for stage, v in data.items():
        if stage == "__wall__":
            continue
        by_mode[stage.split(".")[0]] += v["ms"]
        print(f"{stage:<34}{v['ms']:>12.1f}{v['calls']:>8}"
              f"{v['ms_per_call']:>10.3f}{v['ms'] / total * 100:>7.1f}%")
    print(f"{'-' * 72}")
    print(f"{'By mode':<34}")
    for m, ms in sorted(by_mode.items(), key=lambda x: -x[1]):
        print(f"  {m:<32}{ms:>12.1f}{'':>8}{'':>10}{ms / total * 100:>7.1f}%")
    print(f"{'Stage total':<34}{total:>12.1f}")
    if "__wall__" in data:
        w = data["__wall__"]["ms"]
        print(f"{'End-to-end wall time':<34}{w:>12.1f}")
        print(f"  Stage total / wall time = {total / max(w, 1e-9):.2f}x  "
              f"(values above 1 reflect overlap lost to per-stage synchronization)")


def dump(path):
    """Write the timing report as JSON."""
    d = get()
    d["_meta"] = {"enabled": ENABLED,
                  "note": "Each stage synchronizes separately; summed stage times may exceed wall time."}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"[FlashMCSA timing] Wrote {path}")


# Enable exit-time reporting with FLASH_MCSA_TIMING=1.
# FLASH_MCSA_TIMING_OUT optionally names the JSON output; otherwise use stderr.
# Include the distributed rank in filenames to avoid concurrent writes.
def _rank_suffixed(path: str) -> str:
    rank = os.getenv("RANK") or os.getenv("LOCAL_RANK")
    if rank is None:
        return path
    base, dot, ext = path.rpartition(".")
    return f"{base}.rank{rank}{dot}{ext}" if dot else f"{path}.rank{rank}"


if ENABLED:
    import atexit

    def _on_exit():
        if not _acc:
            return
        out = os.getenv("FLASH_MCSA_TIMING_OUT", "")
        if out:
            try:
                dump(_rank_suffixed(out))
            except Exception as e:      # Reporting failure must not change the main process exit status.
                print(f"[FlashMCSA timing] Failed to write report: {e}")
        summary()

    atexit.register(_on_exit)

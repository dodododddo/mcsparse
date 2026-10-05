"""FlashMCSA token-level sparse attention for MiniMax-H3's diffusers dispatch.

The attention itself goes through FlashMCSA, so PCA reorder, bias correction,
the KV sink and the calibrate/reuse schedule all come from
``TokenSparseContext`` rather than being bolted on around another kernel.

    from mcsa_token_h3 import make_mcsa_token_dispatch
    h3.dispatch_attention_fn = make_mcsa_token_dispatch(orig, sink_tokens=951)

WHY TOKEN-LEVEL AND NOT BLOCK-LEVEL
-----------------------------------
H3 needs a KV sink: it packs one document as
``[text | references | target audio | target video]`` and everything before the
video suffix must stay exact (``models/minimax_h3/H100/adapter.py:1-7``; the
H100 profile refuses to run without it, ``adapter.py:553``).

FlashMCSA's **block**-level config forbids combining that with PCA reorder
(``flash_mcsa/config.py:144-148``) -- block-level PCA permutes K as well, so a
contiguous sink range scatters across many blocks and taking the per-block
intersection eats the whole sparse budget.

The **token**-level config forbids only ``share_lse``
(``flash_mcsa/token_sparse.py:245-248``), because it permutes **Q only**: K/V
keep their original order (``token_sparse.py:1207-1226``), the sink mask is
built in K space (``_select_topk_tokens``, ``token_sparse.py:944-959``) and is
broadcast identically to every query block (``:1076-1079``). The sink set is
therefore *invariant* under any Q permutation -- not merely compatible with it.

So PCA + sink is impossible at block level and free at token level. That
asymmetry is what this whole module rests on; ``tests/test_mcsa_token_h3_logic.py``
asserts it against the FlashMCSA source so an upstream change names itself
instead of silently breaking the arm.

FIVE THINGS THAT BREAK THE RUN IF MISSED
----------------------------------------
1. LAYOUT. The H3 dispatch passes **BTHD** ``[B, T, H, D]``; FlashMCSA is
   **BHSD**. Transpose on both sides.

2. MORE THAN ONE SEQUENCE LENGTH. H3 issues attention at several lengths within
   a *single* forward -- 301 and 38011 were observed on an H100 run. Every other
   FlashMCSA model patch (Wan, Hunyuan, Flux) has exactly one, so the usual
   "length changed => the caller forgot to reset" inference is simply wrong here
   and would abort a perfectly good run. We commit to the LONGEST length at or
   above ``min_seq_len`` -- the first-seen one is not safe, because the text
   token_refiner's length grows with the prompt and can clear the floor -- and
   hand everything else to dense. This costs nothing:
   at L=301 a topk of 0.25 rounds up to one 128-token tile (43% of the sequence)
   for a call worth ~0.006% of the forward's attention work.

3. LAYER NUMBERING. H3 has no per-layer hook, so the layer index is counted
   here. FlashMCSA caches token selections
   *per layer*, so miscounting applies layer N's indices to layer M. Nothing
   downstream catches it: ``cache.check_seq_len`` (``token_sparse.py:348-353``)
   compares only ``M = ceil(L/BLKQ)``, which is identical across layers. It
   would read as an unexplained quality regression. Hence the raising
   invariants in ``_note_layer`` -- and hence dense-routed calls consume no
   layer index, which is what keeps the numbering intact while H3 interleaves
   its short-sequence attention with the long one.

4. SINK LENGTH. Must be passed in; it cannot be inferred here. See
   ``make_mcsa_token_dispatch``'s docstring.

5. STEP CLOCK. ``mcsa_token_begin_forward()`` must be called once per denoising
   step. Without it every step looks like step 0, the run never leaves the dense
   warmup, and the arm reports a plausible ~1.0x with no sign anything is wrong.

NO COMPILED EXTENSION REQUIRED (ONE IS OPTIONAL)
------------------------------------------------
The token-level kernel this module drives is ``flash_attn_gather_kv_func``,
reached through ``TokenSparseBackend._gkv``, and it is written in CuteDSL --
Python, JIT-compiled by ``nvidia-cutlass-dsl`` at first call. The
block-probability pass beside it is Triton. So ``mcsparse`` vendors the kernels
as pure source and installs with pip alone: no nvcc, no prebuilt .so, no CUDA
toolkit. FA3 is a required install for this arm.

The calibrate step's full attention is FA3 too (``ops/full_attn``), which
returns the output and its LSE from one pass -- the LSE is what scores the token
selection.

There used to be an optional path through FlashMCSA's compiled BHSD-native
kernel, on the grounds that the published numbers were measured with it. A/B
measurement says the two are the same: 315.20 vs 314.88 ms/call (7xH20,
L=104179, 49 layers) -- 0.1%, below run-to-run noise. FA3's six extra
full-tensor transposes per layer come to ~0.3 s, which is nothing against
315 ms/call. So that path and its ``.so`` dependency were removed; the fp8 and
block-level paths that also needed ``_C`` were already gone.

ON THE SPEEDUP YOU SHOULD EXPECT
--------------------------------
Below the kernel-level speedup, and that is structural rather than a bug. A
BTHD caller pays transpose+materialise on q/k/v at entry, then ``_sp`` does it
again internally (``token_sparse.py:1695-1697``), and PCA adds two more
full-tensor copies (apply at ``:1222``, invert at ``:1224``). A BSHD-native
caller pays none of that. Read ``kernel_calls`` and ``effective_density`` before
concluding anything about the kernel from the wall clock.
"""

from __future__ import annotations

import math
import os
import sys
import time

# The kernels live inside this package -- mcsparse.kernels.flash_mcsa -- so
# there is no path bootstrap and no external checkout to locate. That is the
# point of vendoring them: the token-sparse path is pure CuteDSL + Triton and
# needs no compiled extension, so `pip install mcsparse` is sufficient.
#
# $FLASHMCSA_ROOT still wins if set, for developing against a working copy.
_MCSA_ROOT = os.environ.get("FLASHMCSA_ROOT")
if _MCSA_ROOT and os.path.isdir(_MCSA_ROOT) and _MCSA_ROOT not in sys.path:
    sys.path.insert(0, _MCSA_ROOT)

# FlashMCSA's token-level kernel constants. BLKQ and tile_n are asserted to
# these exact values by TokenSparseConfig.__post_init__ (token_sparse.py:220),
# so they are not free parameters -- they are mirrored here only so the
# eligibility check and the density arithmetic can run without importing torch.
BLKQ = 128
TILE_N = 128
SUPPORTED_HEAD_DIMS = (128, 144)

DEFAULT_TOPK = 0.25
DEFAULT_NUM_STEPS = 50          # H3's default; see make_mcsa_token_dispatch
DEFAULT_DENSE_STEPS = 10        # the dense arm uses the same, so the A/B is fair
DEFAULT_DENSE_LAYERS = "0-1"    # ditto
DEFAULT_CALIBRATE_EVERY = 10
# Pinned rather than derived from calibrate_every. Step 9 is the last dense step,
# so full_calibrate folds the calibration into a step already being paid for --
# it is the one free calibration available, which is why it is the only one.
# A second calibration mid-trajectory costs a full attention pass over 48 layers
# (~563 ms/layer measured, ~27 s/sample, 2.3% of end-to-end); dropped because the
# speed was worth more here than a fresher residual. The tradeoff it buys back:
# step 9's residual and PCA permutation now serve steps 10-48 rather than 10-24,
# and FlashMCSA/bench/diag_pca_bias.py shows both decay as Q drifts, with
# PCA+bias decaying faster than bias alone. Re-add with calibrate_steps=(9, 25)
# if late-trajectory quality sags.
DEFAULT_CALIBRATE_STEPS = (9,)
# Sequence-length floor for sparsifying at all. H3 issues attention at more than
# one length per forward -- 301 and 38011 were observed on an H100 run -- and the
# short one is not worth taking: at L=301 a topk of 0.25 rounds up to a single
# 128-token tile (43% of the sequence) while the call itself is ~0.006% of the
# forward's attention work. 4096 is well clear of the short bucket and well below
# any real video length.
DEFAULT_MIN_SEQ_LEN = 4096

_DEFAULT_SCALE_SENTINEL = object()


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------
# dispatch_calls / kernel_calls / dense_guard_calls / fallback_calls come
# first so a reporting layer can read them without knowing the backend.
_STATS = {
    "dispatch_calls": 0,
    "kernel_calls": 0,
    "dense_guard_calls": 0,
    "fallback_calls": 0,
    # token-MCSA specific
    "steps_ticked": 0,
    "layers_per_step": 0,
    "calibrate_calls": 0,
    "full_calibrate_calls": 0,
    "splice_calls": 0,
    "sink_tokens": 0,
    "seq_len": 0,
    "batch": 0,
    "nominal_density": 0.0,
    "effective_density": 0.0,
    # Offload instrumentation. FlashMCSA keeps no counters on these paths -- the
    # synchronous fallback in _get_prefetched_indices (token_sparse.py:439-444)
    # is entirely silent -- so "is prefetch working" is otherwise unanswerable.
    # A miss means the layer blocked on a ~639 MB H2D instead of finding the
    # copy already staged, which is exactly when offload stops paying.
    "idx_prefetch_hit": 0,
    "idx_prefetch_miss": 0,
    "idx_block_ms": 0.0,
    "res_prefetch_hit": 0,
    "res_prefetch_miss": 0,
    "res_block_ms": 0.0,
    # layer_idx -> [indices_ms, residual_ms]. Lives in _STATS rather than in a
    # module global so the per-sample reset loop clears it along with everything
    # else -- a stale per-layer map next to fresh counters would be worse than
    # having neither.
    "block_ms_by_layer": {},
}


def get_mcsa_token_stats() -> dict:
    """Run counters, plus token-MCSA specifics.

    Two density numbers, and the gap between them matters:

    ``nominal_density``   what you asked for (``topk``).
    ``effective_density`` what the selection actually computes, because
        ``_select_topk_tokens`` *prepends* the sink to the top-k pool
        (``token_sparse.py:1076-1089``) rather than carving it out of the
        budget. So it is roughly ``(P + topk*L) / L``.

    For HunyuanVideo that gap is a rounding error (text is 0.2%-0.4% of the
    sequence). For H3 the prefix is text + references + audio and can be
    thousands of tokens, so ``topk=0.1`` may really be 0.15 -- which would make
    a comparison against another method's nominal 0.3 not the comparison it
    looks like. Report both; compare on the effective one.

    ``seen_seq_lens`` maps every attention sequence length observed to its call
    count. H3 has more than one, and only ``seq_len`` was sparsified -- use this
    to confirm the split is what you expect rather than assuming the floor was
    set correctly.
    """
    out = dict(_STATS)
    out["seen_seq_lens"] = dict(sorted(_CTX["seen_seq_lens"].items()))
    return out


# --------------------------------------------------------------------------
# step / layer clock
# --------------------------------------------------------------------------
# H3 gives us neither a per-layer nor a per-step hook, so both are counted here.
# _CTX["step"] is kept equal to ctx._sc by construction; see _advance_step().
_CTX = {
    "step": -1,              # denoising step; -1 until the first begin_forward
    "layer": 0,              # reset each begin_forward
    "tick": 0,               # raw pre-hook fires (2 per step under serial CFG)
    "layers_per_step": None,  # locked at the end of step 0
    "declared_layers": None,  # set only if the caller passed num_layers
    "shapes": {},            # layer_idx -> (B, H, L, D), locked on first sight
    # H3 runs attention at MORE THAN ONE sequence length per forward -- observed
    # 301 and 38011. So there is no single "the" sequence length; we sparsify one
    # bucket and hand the rest to dense. See min_seq_len in the factory.
    #
    # THE LONGEST WINS, not the first. Committing to whichever eligible length
    # arrived first was a latent bug. The short call is the text token_refiner
    # (transformer_minimax_h3.py:631), and its length tracks the prompt --
    # about 1.45 tokens per IR character, measured across five prompts. A long
    # enough prompt pushes it past min_seq_len, so the refiner call became the
    # committed length and every 104k-token packed-sequence call after it was
    # refused as "a second long sequence length". Three of five hand-written
    # prompts hit this: stroll committed to 4207 and fell back 2450 times,
    # while cat (refiner 3976, just under the 4096 floor) worked. The packed
    # sequence is always the longest call, so taking the maximum is correct.
    "sparse_seq_len": None,  # the length being sparsified; None until decided
    "seen_seq_lens": {},     # length -> call count, for reporting
    # Lengths seen during the discovery window that clear min_seq_len. The
    # commitment is made from the largest of these once the window closes.
    "candidate_seq_lens": {},
    "ctx": None,             # TokenSparseContext
    "ticks_per_step": 1,
    "cfg_slots": (),
}


def _use_slot(ctx, name: str) -> None:
    """Switch TokenSparseContext to a named cache slot, without the contextmanager.

    ``ctx.cache_context`` (``token_sparse.py:1889-1930``) is a generator
    contextmanager meant to wrap a whole transformer forward. We are driven from
    a ``register_forward_pre_hook``, which has no exit point to close a generator
    against, so its body is transcribed here as a plain state assignment.

    This reaches into privates. It is worth it only if H3 turns out to run CFG
    as two serial forwards, where the uncond pass's calibrate would otherwise
    clobber the cond pass's token selection -- the same problem Wan 2.2 has.
    """
    ctx._slse_per_slot[ctx._active_slot] = ctx._slse
    if name not in ctx._cache_slots:
        ctx._cache_slots[name] = {}
    ctx._active_slot = name
    ctx._caches = ctx._cache_slots[name]
    ctx._slse = ctx._slse_per_slot.get(name)


def mcsa_token_begin_forward() -> None:
    """Advance the denoising-step clock. Call once per transformer forward.

    One call per transformer forward, from a ``register_forward_pre_hook``.
    """
    ctx = _CTX["ctx"]

    # 1. Close the previous forward.
    if _CTX["tick"] > 0:
        seen = _CTX["layer"]
        if _CTX["layers_per_step"] is None:
            _lock_layer_count(seen)
        elif seen != _CTX["layers_per_step"]:
            raise RuntimeError(
                f"layer count changed between steps: saw {seen}, expected "
                f"{_CTX['layers_per_step']}. Every cached token selection is "
                f"keyed by layer index, so from here on layer N's indices would "
                f"be applied to a different layer -- silently, since "
                f"check_seq_len only compares ceil(L/BLKQ). Refusing to continue."
            )

    # 2. Advance. The step>0 guard is what keeps _CTX["step"] == ctx._sc:
    #    TokenSparseContext starts life at _sc=0, so the *first* tick must
    #    move us from -1 to 0 without calling ctx.step().
    _CTX["tick"] += 1
    if _CTX["tick"] % _CTX["ticks_per_step"] == 0:
        _CTX["step"] += 1
        _STATS["steps_ticked"] = _CTX["step"] + 1
        if _CTX["step"] > 0 and ctx is not None:
            ctx.step()

    # 3. Reset the layer counter, and pick the CFG slot if we are using them.
    _CTX["layer"] = 0
    slots = _CTX["cfg_slots"]
    if slots and ctx is not None:
        _use_slot(ctx, slots[(_CTX["tick"] - 1) % len(slots)])


def _lock_layer_count(n: int) -> None:
    """Freeze the layer count at the end of step 0 and tell the context about it.

    ``ctx._nl`` is normally set by FlashMCSA's own ``patch_*`` helpers. Nothing
    sets it for us, and while it is None two things quietly do nothing:
    residual prefetch (``token_sparse.py:2044-2049``) and ``adaptive_layers``
    (``_gc`` at ``:1996-1998`` only assigns ``layer_t`` at cache-creation time,
    and only if ``_nl`` is known).

    Pre-creating the caches here mirrors ``patch_flux_like_dit:2167-2168``.
    """
    if n <= 0:
        raise RuntimeError(
            "step 0 produced no eligible attention calls, so the layer count "
            "could not be determined. Every call fell through to dense -- check "
            "get_mcsa_token_stats()['fallback_calls'] for why."
        )
    _CTX["layers_per_step"] = n
    _STATS["layers_per_step"] = n
    ctx = _CTX["ctx"]
    if ctx is not None and ctx._nl is None:
        ctx._nl = n
        for i in range(n):
            ctx._gc(i)


def _discovering() -> bool:
    """True while the sparse length may still be revised upward.

    The test is "has the kernel run yet", not "is this step 0". Those coincide
    under the default dense_steps=10 -- step 0 routes every layer to dense, so
    nothing is cached -- but dense_steps=0 would build caches immediately, and
    a length change after that would leave selections sized to the old length.
    kernel_calls==0 is exactly the condition under which no cache exists.
    """
    return _STATS["kernel_calls"] == 0


def _maybe_upgrade_seq_len(T: int, B: int, sink_tokens: float, topk: float,
                           min_seq_len: int) -> None:
    """Record T as a candidate and keep the longest as the sparse length.

    Called for every eligible call. The first one commits; a strictly longer
    one seen during the discovery window replaces it, because H3 runs the
    prompt-length text token_refiner BEFORE its main blocks and that shorter
    call would otherwise win purely by arriving first.
    """
    _CTX["candidate_seq_lens"][T] = _CTX["candidate_seq_lens"].get(T, 0) + 1
    cur = _CTX["sparse_seq_len"]
    if cur is not None and T <= cur:
        return
    if cur is not None:
        # An upgrade. Layer bookkeeping is keyed by index, and the shapes map
        # holds the old length for layers already numbered this step, so both
        # have to be dropped -- otherwise _note_layer raises on the shape
        # change it is designed to catch.
        print(f"[mcsa-token] sparse length {cur} -> {T}: a longer eligible "
              f"call appeared in the discovery window. The prompt-length text "
              f"token_refiner runs before the main blocks, so the shorter "
              f"call arrived first; the {T}-token packed sequence is the one "
              f"worth sparsifying.", flush=True)
        _CTX["shapes"] = {}
        _CTX["layer"] = 0
    _CTX["sparse_seq_len"] = T
    _STATS["seq_len"] = T
    _STATS["batch"] = B
    _STATS["effective_density"] = effective_density(
        sink_tokens=sink_tokens, seq_len=T, topk=topk)


def _note_layer(shape) -> int:
    """Consume one layer index, enforcing the numbering invariants.

    Both violations raise rather than warn: they mean the cached selections are
    now attributed to the wrong layers, which produces wrong output with no
    error anywhere downstream.
    """
    li = _CTX["layer"]
    _CTX["layer"] += 1

    lps = _CTX["layers_per_step"]
    if lps is not None and li >= lps:
        raise RuntimeError(
            f"attention call #{li} in a step that had {lps} layers. Either the "
            f"model routed an extra call through dispatch_attention_fn, or the "
            f"step clock is not being ticked (is mcsa_token_begin_forward() "
            f"wired into the transformer's forward pre-hook?)."
        )

    known = _CTX["shapes"].setdefault(li, shape)
    if known != shape:
        raise RuntimeError(
            f"layer {li} changed shape from {known} to {shape}. This usually "
            f"means a non-self-attention call (token refiner, cross-attention) "
            f"is also going through dispatch_attention_fn and has shifted the "
            f"layer numbering."
        )
    return li


# Stats that describe the *configuration* rather than the run. reset() must not
# clear them: the caller resets after building the dispatch, so zeroing these
# would make every report read "sink=0 nominal_density=0" no matter what was
# configured -- and sink=0 is exactly the failure the arm refuses to start with.
_CONFIG_STATS = ("sink_tokens", "nominal_density")


def reset_mcsa_token_state() -> None:
    """Reset per-sample state. Call between samples."""
    ctx = _CTX["ctx"]
    if ctx is not None:
        ctx.reset()
    _CTX["step"] = -1
    _CTX["layer"] = 0
    _CTX["tick"] = 0
    # Keep layers_per_step if it came from an explicit num_layers; re-discover
    # it otherwise.
    _CTX["layers_per_step"] = _CTX.get("declared_layers")
    _CTX["shapes"] = {}
    _CTX["sparse_seq_len"] = None
    _CTX["candidate_seq_lens"] = {}
    _CTX["seen_seq_lens"] = {}
    for k, v in _STATS.items():
        if k not in _CONFIG_STATS:
            _STATS[k] = type(v)()
    if _CTX.get("declared_layers"):
        _STATS["layers_per_step"] = _CTX["declared_layers"]


# --------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# offload instrumentation
# --------------------------------------------------------------------------
# FlashMCSA keeps no counters on either transfer path -- the synchronous
# fallback in _get_prefetched_indices (token_sparse.py:439-444) is entirely
# silent -- so "did prefetch land?" is otherwise unanswerable: a run where it
# never lands produces the same output and the same counters as one where it
# always does, just slower.
_BLOCK_BY_LAYER = {}   # unused; the live map is _STATS["block_ms_by_layer"],
                       # which the per-sample reset loop clears automatically


def _instrument_offload():
    """Patch TokenSparseCache to count prefetch hits/misses and blocked time.

    WHAT IS MEASURED: CPU wall time inside the call. That is the honest number
    for the miss path, where ``.to(device, non_blocking=False)``
    (token_sparse.py:442) blocks until a ~639 MB copy completes. On the hit path
    it is near zero by construction -- the wait is enqueued on the stream, not on
    the CPU -- so a large blocked time next to a high hit rate would itself be a
    finding.

    NOT MEASURED: GPU-side stall. If a kernel later waits on the prefetch event,
    that lands in the kernel's own duration, not here. So blocked time is a
    LOWER BOUND on what offload costs; the hit rate is the primary signal.

    Idempotent, and keyed on the class so repeated dispatch construction (which
    the sweep does per arm) cannot stack wrappers.
    """
    from mcsparse.kernels.flash_mcsa.token_sparse import TokenSparseCache

    if getattr(TokenSparseCache, "_mcsparse_instrumented", False):
        return
    TokenSparseCache._mcsparse_instrumented = True

    _orig_idx = TokenSparseCache._get_prefetched_indices
    _orig_res = TokenSparseCache._get_residual_raw

    def _bucket(cache):
        li = getattr(cache, "_layer_idx", -1)
        return _STATS["block_ms_by_layer"].setdefault(li, [0.0, 0.0])

    def _timed_indices(self, device):
        staged = self._prefetched_indices is not None
        t0 = time.perf_counter()
        out = _orig_idx(self, device)
        dt = (time.perf_counter() - t0) * 1e3
        _STATS["idx_prefetch_hit" if staged else "idx_prefetch_miss"] += 1
        _STATS["idx_block_ms"] += dt
        _bucket(self)[0] += dt
        return out

    def _timed_residual(self, device):
        # _cached_residual means it never left the GPU, so this is not an
        # offload event at all and must not be scored as a hit.
        resident = self._cached_residual is not None
        staged = self._prefetched_residual is not None
        t0 = time.perf_counter()
        out = _orig_res(self, device)
        dt = (time.perf_counter() - t0) * 1e3
        if not resident:
            _STATS["res_prefetch_hit" if staged else "res_prefetch_miss"] += 1
            _STATS["res_block_ms"] += dt
            _bucket(self)[1] += dt
        return out

    TokenSparseCache._get_prefetched_indices = _timed_indices
    TokenSparseCache._get_residual_raw = _timed_residual


def offload_report() -> str:
    """Multi-line verdict on whether the offloads are paying for themselves."""
    lines = []
    for tag, h, m, ms in (
            ("indices", _STATS["idx_prefetch_hit"],
             _STATS["idx_prefetch_miss"], _STATS["idx_block_ms"]),
            ("residual", _STATS["res_prefetch_hit"],
             _STATS["res_prefetch_miss"], _STATS["res_block_ms"])):
        n = h + m
        if n:
            lines.append(f"  {tag:9} {h}/{n} prefetch hit ({h / n:5.1%})  "
                         f"CPU-blocked {ms / 1000:7.2f}s  "
                         f"({ms / n:.1f} ms/call)")
    if not lines:
        return "  offload not active"
    # Per-layer, worst first. The first sparse layer is the one to watch: its
    # prefetch is issued by step() rather than by a preceding sparse layer, and
    # it lands while the dense full_layers ahead of it are still streaming their
    # own weights over the same PCIe link.
    busy = sorted(((li, i + r) for li, (i, r) in _STATS["block_ms_by_layer"].items()),
                  key=lambda kv: -kv[1])[:5]
    if busy and busy[0][1] > 0:
        lines.append("  worst layers by blocked time: " + ", ".join(
            f"L{li}={ms / 1000:.2f}s" for li, ms in busy))
    return "\n".join(lines)


# --------------------------------------------------------------------------
def _parse_layer_ranges(spec: str) -> frozenset:
    """Parse "0-1,5" into {0,1,5}.

    Twelve lines, kept local rather than pulled from a shared utility module
    that would exist only for this.
    """
    layers = set()
    for item in str(spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(item))
    return frozenset(layers)


def build_schedule(*, num_steps: int, dense_steps: int, dense_layers: str,
                   calibrate_every: int, calibrate_steps=None) -> dict:
    """Translate dense_steps + dense_layers into FlashMCSA's schedule.

    ``calibrate_steps``, when given, is used verbatim and ``calibrate_every`` is
    ignored -- for pinning the calibration to specific steps.

    These are NOT the same mechanism, and conflating them costs the single best
    interaction available here:

      ``dense_steps`` here       short-circuits to the original dispatch
                                before the backend is ever consulted.
      FlashMCSA ``full_steps``  is a set consulted by ``is_full()``. A step in
                                **both** ``full_steps`` and ``calibrate_steps``
                                is routed *into* the sparse path by
                                ``_should_forward_full_step_to_sparse_path()``
                                (``token_sparse.py:1982-1990``) to run
                                ``full_calibrate``.

    ``full_calibrate`` (``_full_cal``, ``token_sparse.py:1320-1443``) does one
    full attention that yields the output, the LSE, the token selection *and*
    the bias-correction residual. Putting the last dense step in both lists
    therefore buys the entire calibration for the price of a dense step we were
    going to pay anyway. That only works if dense steps are translated rather
    than short-circuited -- so there is no short-circuit dense guard here.

    With the defaults (50 steps, dense 10, calibrate at 9 and 25):
        full_steps      [0..9]
        calibrate_steps [9, 25]             <- 9 is the free one
        full_layers     {0, 1}

    Cadence note: ``FlashMCSA/bench/diag_pca_bias.py`` recommends recalibrating
    roughly every 25% of the trajectory, because the cached residual and the
    frozen perm both go stale as Q drifts -- and PCA+bias decays *faster* than
    bias alone. Two calibrations over 50 steps is sparser than that, so the
    residual from step 25 is serving 24 reuse steps; if quality sags late in the
    run, that is the first thing to revisit.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if dense_steps < 0:
        raise ValueError(f"dense_steps must be >= 0, got {dense_steps}")
    if calibrate_every <= 0:
        raise ValueError(f"calibrate_every must be positive, got {calibrate_every}")

    full_layers = sorted(_parse_layer_ranges(dense_layers))
    full_steps = list(range(min(dense_steps, num_steps)))

    if calibrate_steps is not None:
        cal = sorted({int(s) for s in calibrate_steps})
        if not cal:
            raise ValueError("calibrate_steps was given but is empty")
    else:
        # The first calibrate is the last dense step, so it costs nothing extra.
        # With dense_steps=0 there is no dense step to piggyback on, so start at 0.
        first_cal = max(0, dense_steps - 1)
        cal = [s for s in range(first_cal, num_steps, calibrate_every)]
        # Drop a calibrate on the very last step: it pays for a full attention
        # plus a sparse forward to build a selection and a residual that no later
        # step ever reuses. Keep it only if it is the sole calibrate we have.
        if len(cal) > 1 and cal[-1] == num_steps - 1:
            cal.pop()
        if not cal:
            cal = [0]

    free = cal[0] if (full_steps and cal[0] in full_steps) else None
    return {
        "full_steps": full_steps or None,
        "full_layers": full_layers,
        "calibrate_steps": cal,
        "free_calibrate_step": free,
    }


def validate_schedule(*, num_steps, calibrate_steps, update_steps, approx_steps):
    """Reject overlapping step lists.

    ``TokenSparseConfig`` has **no** ``_validate_steps`` -- that method exists
    only on the block-level ``MCSAConfig`` (``config.py:139,156``). So the token
    path silently accepts a step that is in two lists at once, where
    ``TokenSparseCache.mode`` (``token_sparse.py:292-306``) just resolves it by
    precedence and the loser is quietly ignored. Validate it ourselves.

    ``full_steps`` is deliberately *not* checked against ``calibrate_steps``:
    that overlap is the ``full_calibrate`` path and is the whole point.
    """
    named = [("calibrate_steps", set(calibrate_steps or [])),
             ("update_steps", set(update_steps or [])),
             ("approx_steps", set(approx_steps or []))]
    for i in range(len(named)):
        for j in range(i + 1, len(named)):
            overlap = named[i][1] & named[j][1]
            if overlap:
                raise ValueError(
                    f"{named[i][0]} and {named[j][0]} overlap at "
                    f"{sorted(overlap)}. TokenSparseConfig will not catch this "
                    f"(it has no _validate_steps); cache.mode would silently "
                    f"resolve by precedence and drop one of them."
                )
    for name, steps in named:
        bad = [s for s in steps if s >= num_steps or s < 0]
        if bad:
            raise ValueError(
                f"{name} contains {sorted(bad)}, outside [0, {num_steps}). "
                f"Those entries would never fire."
            )


def effective_density(*, sink_tokens: int, seq_len: int, topk: float,
                      tile_n: int = TILE_N) -> float:
    """Fraction of KV actually attended, replaying _select_topk_tokens' arithmetic.

    ``token_sparse.py:1076-1089``: the sink is concatenated onto the top-k
    selection and the per-position count becomes ``topk_budget + n_sink``,
    aligned up to ``tile_n`` and clamped to the pool size. So the sink adds to
    the budget rather than consuming it.
    """
    if seq_len <= 0:
        return 0.0
    budget = max(1, int(topk * seq_len))
    budget = ((budget + tile_n - 1) // tile_n) * tile_n
    total = min(seq_len, budget + max(0, sink_tokens))
    return total / seq_len


# --------------------------------------------------------------------------
# eligibility
# --------------------------------------------------------------------------
# "fallback" -> hand the call to dense; the kernel cannot express it.
# "raise"    -> the configuration is wrong, not merely inexpressible.
#
# The split matters because a silent fallback in this arm produces zero
# sparsity, a ~1.0x reading and no error -- which is what the kernel_calls==0
# warning in mcsparse.pipeline exists to catch. Keeping kernel_calls honest is
# what keeps that warning working.
def check_eligibility(*, ndim, is_cuda, dtype_is_bf16, shapes_match, head_dim,
                      seq_len_q, seq_len_k, has_attn_mask, is_causal, dropout_p,
                      enable_gqa, has_parallel_config, scale, sink_tokens,
                      sparse_seq_len, min_seq_len, discovering=False):
    """Pure-scalar eligibility check. Returns (ok, reason, severity).

    ``sparse_seq_len`` is the long sequence length this run committed to on its
    first eligible call, or None before that. ``min_seq_len`` is the floor below
    which a call is left dense.

    On the length check, the important part: H3 issues attention at more than one
    sequence length within a single forward (301 and 38011 were observed), so a
    changed length is NOT by itself a missing reset. Short calls are routed to
    dense; only a *second long* length is a real problem, because that is the one
    that would collide with the cached per-layer selection.
    """
    if has_parallel_config:
        # Under context parallelism each rank owns a slice of the sequence, so a
        # global [0, P) sink range no longer names the prefix.
        return False, "parallel_config is not None (context parallelism)", "fallback"
    if has_attn_mask:
        return False, "attn_mask is not None", "fallback"
    if is_causal:
        return False, "is_causal=True", "fallback"
    if dropout_p:
        return False, f"dropout_p={dropout_p}", "fallback"
    if enable_gqa:
        # The gather_kv path needs indices heads == q heads.
        return False, "enable_gqa=True", "fallback"
    if ndim != 4:
        return False, f"query.ndim={ndim}, expected 4", "fallback"
    if not is_cuda:
        return False, "query is not on CUDA", "fallback"
    if not dtype_is_bf16:
        return False, "dtype is not bfloat16", "fallback"
    if not shapes_match:
        return False, "q/k/v shapes differ", "fallback"
    if seq_len_q != seq_len_k:
        return False, f"L_q={seq_len_q} != L_k={seq_len_k} (cross-attention)", "fallback"
    if head_dim not in SUPPORTED_HEAD_DIMS:
        return False, (f"head_dim={head_dim}, kernel supports "
                       f"{SUPPORTED_HEAD_DIMS}"), "fallback"

    # --- the multi-length routing ---------------------------------------
    # Below the floor there is nothing worth taking: at L=301 a topk of 0.25
    # rounds up to one 128-token tile, i.e. 43% of the sequence, and the whole
    # call is ~0.006% of the forward's attention work. Sparsifying it costs
    # selection overhead to save nothing.
    if seq_len_q < max(min_seq_len, BLKQ):
        return False, (f"L={seq_len_q} below the sparse floor "
                       f"{max(min_seq_len, BLKQ)}"), "fallback"
    if sparse_seq_len is not None and seq_len_q != sparse_seq_len:
        # A second long length. Cached selections are per layer and sized to the
        # committed one, so this cannot share them -- but it is also not
        # obviously a missing reset, so leave it dense rather than aborting.
        #
        # EXCEPT while the commitment is still provisional. The text
        # token_refiner runs before the main blocks and its length grows with
        # the prompt, so on a long enough prompt that call arrives first and
        # clears min_seq_len. Committing to it refused every packed-sequence
        # call for the rest of the run. During the discovery window a strictly
        # longer length supersedes the provisional one; see
        # _maybe_upgrade_seq_len.
        if not (discovering and seq_len_q > sparse_seq_len):
            return False, (f"L={seq_len_q} is a second long sequence length "
                           f"(already committed to {sparse_seq_len})"), "fallback"
    if scale is not None and not math.isclose(float(scale), head_dim ** -0.5,
                                              rel_tol=1e-6):
        # FlashMCSA hardcodes 1/sqrt(D) throughout (_sp:1707, _cal:1465,
        # _full_cal:1328); honouring a different scale would need kernel changes.
        return False, f"non-default scale={scale}", "fallback"

    # Configuration errors below this line.
    if sink_tokens >= seq_len_k:
        return False, (f"sink_tokens={sink_tokens} >= L_k={seq_len_k}: the sink "
                       f"covers the whole sequence, nothing would be sparse"), "raise"
    return True, "", ""


# --------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------
def make_mcsa_token_dispatch(
    original_dispatch,
    *,
    sink_tokens: int,
    num_layers: int = None,
    num_steps: int = DEFAULT_NUM_STEPS,
    topk: float = DEFAULT_TOPK,
    dense_steps: int = DEFAULT_DENSE_STEPS,
    dense_layers: str = DEFAULT_DENSE_LAYERS,
    calibrate_every: int = DEFAULT_CALIBRATE_EVERY,
    calibrate_steps=DEFAULT_CALIBRATE_STEPS,
    pca_reorder: bool = True,
    pca_pow_iters: int = 4,
    pca_proj_dim: int = None,
    bias_correction: bool = True,
    selection_mode: str = "topk",
    topp: float = None,
    adaptive_layers: bool = False,
    use_chunk_selector: bool = False,
    chunk_selector_size: int = 16384,
    offload_indices: bool = True,
    cache_offload: str = "residual",
    offload_prefetch_ahead: int = 1,
    update_steps=None,
    approx_steps=None,
    full_attn_impl: str = "fa3",
    allow_no_sink: bool = False,
    min_seq_len: int = DEFAULT_MIN_SEQ_LEN,
    ticks_per_step: int = 1,
    cfg_slots=(),
):
    """Wrap the H3 attention dispatch with FlashMCSA token-level sparse attention.

    ``dense_steps`` and ``dense_layers`` control the warmup and dense layers;
    ``build_schedule`` translates them into FlashMCSA's full-set schedule.

    Args:
        sink_tokens: length of H3's exact prefix, i.e. ``target_video_start``.
            **Required, with no default, and it cannot be inferred here.** The
            dispatch sees only q/k/v; the Sol-Engine adapter derives this from
            host-side token-position arrays (``adapter.py:220``,
            ``_last_contiguous_run_start``) that never reach a dispatch
            function. Nor is there a signal in the tensors themselves: the
            prefix is text+references+audio, all real values, and H3's padding
            sits at the *tail* (``adapter.py:211``). Any inference would be a
            heuristic whose failure mode is silently dropping the conditioning,
            which is precisely what the sink exists to prevent.

            Unlike ``hunyuan_patch._ensure_text_sink`` (``hunyuan_patch.py:205-226``),
            which has to discover its range at runtime and mutate a live config,
            we know P up front and set ``sink_ranges`` at construction -- then
            only *validate* at runtime. Same "it must never change" invariant,
            established earlier and more safely.
        num_layers: H3's transformer depth. Optional -- it is discovered at the
            end of step 0 -- but required if ``adaptive_layers=True``, because
            ``_gc`` assigns ``layer_t`` only at cache-creation time and only when
            ``ctx._nl`` is already known (``token_sparse.py:1996-1998``);
            discovering it later would leave ``adaptive_layers`` a silent no-op.
        num_steps: denoising steps, used to lay out the calibrate schedule. The
            default 50 comes from ``models/minimax_h3/H100/profiles.py``
            (``H3_MEASURED_NUM_STEPS``) and the H3 README, *not* from the
            pipeline. ``get_mcsa_token_stats()['steps_ticked']`` reports what
            actually happened -- check it once.
        ticks_per_step: how many times the transformer is forwarded per denoising
            step. ``mcsparse.pipeline`` drives the step clock from a
            ``register_forward_pre_hook``, which fires once per *forward*, not
            once per *step*. Those are the same thing only if H3 runs a single
            pass. If it does classifier-free guidance as two serial forwards
            (cond then uncond), the hook fires twice per step, and with
            ``ticks_per_step=1`` a 50-step run is counted as 100 steps -- so
            ``calibrate_steps=(9, 25)`` would land near steps 4.5 and 12.5
            instead, and the uncond pass would clobber the cond pass's token
            selection. Setting 2 (with ``cfg_slots=("cond","uncond")``) makes two
            ticks equal one step and gives each pass its own cache.

            Whether H3 needs this is **unverified** -- diffusers is not
            installed on the box this was written on. You do not have to guess:
            run once and read ``steps_ticked``. 50 means leave it at 1; ~100
            means set 2. The stats block prints this diagnosis itself.
        min_seq_len: only sparsify attention at or above this sequence length;
            shorter calls go to dense. H3 issues attention at **more than one
            length per forward** -- 301 and 38011 on the run this was calibrated
            against -- and there is nothing to win on the short one: a topk of
            0.25 at L=301 rounds up to a single 128-token tile, i.e. 43% of the
            sequence, for a call that is ~0.006% of the forward's attention work.
            ``get_mcsa_token_stats()['seen_seq_lens']`` reports every length
            observed, so you can check the split rather than trust this default.
        offload_indices: keep the per-layer token selection on pinned CPU memory
            and prefetch it back one layer ahead. Saves the most GPU memory of
            any knob here -- at H3's shape the indices are
            (1, 56, 297, 9600) int32 = 639 MB per layer, so ~32 GB across 50
            layers.
        cache_offload: ``'residual'`` does the same for the bias-correction
            residual (545 MB/layer, ~27 GB over 50 layers), ``'none'`` keeps it
            resident. Requires ``bias_correction=True``.
        offload_prefetch_ahead: layers to prefetch ahead; only 0 or 1 are
            implemented.

        ON TURNING BOTH OFFLOADS ON AT ONCE. They do not conflict logically --
        FlashMCSA gives each its own CUDA stream precisely so the two D2H copies
        do not serialise. They compete for **PCIe**, and at H3's shape that is
        the binding constraint: 639 MB + 545 MB = 1184 MB per layer is ~47 ms on
        PCIe4 x16 or ~24 ms on PCIe5, against a sparse-attention kernel of
        roughly 15-26 ms. Prefetch can only run one layer ahead, so the excess
        cannot be hidden, and H3 already streams a 144 GB checkpoint over the
        same link (``enable_auto_cpu_offload`` on the ComponentsManager).
        Expect both-on to be slower than either-on unless your link is faster
        than assumed. Both default to on because memory pressure is the thing
        that stops a run outright; if you are not near the limit, turn them off
        and keep the bandwidth. Measure, do not assume -- these are arithmetic on
        estimated bandwidth, not something benchmarked on your box.
        allow_no_sink: permit ``sink_tokens=0``. Off by default, echoing
            ``adapter.py:553``'s refusal to run H3 without the full prefix sink.
    """
    # Argument validation runs BEFORE importing torch/flash_mcsa, so a bad
    # config is reported as a bad config rather than as whatever the import
    # happens to fail with -- and so the pure-logic tests can reach it.
    sink_tokens = int(sink_tokens)
    if sink_tokens < 0:
        raise ValueError(f"sink_tokens must be >= 0, got {sink_tokens}")
    if sink_tokens == 0 and not allow_no_sink:
        raise ValueError(
            "sink_tokens=0. H3 packs one document as "
            "[text | references | target audio | target video] and everything "
            "before the video suffix must stay exact; the H100 profile itself "
            "refuses to run without it (adapter.py:553). Without the sink the "
            "text and reference conditioning can simply be dropped by top-k, "
            "which is a correctness failure, not a quality one. Pass "
            "allow_no_sink=True only for a deliberate ablation."
        )
    if adaptive_layers and num_layers is None:
        raise ValueError(
            "adaptive_layers=True requires num_layers. Layer depth is read by "
            "_gc at cache-creation time and only when ctx._nl is known "
            "(token_sparse.py:1996-1998); if we discover the depth later, "
            "layer_t stays None and adaptive_layers silently does nothing."
        )
    if bias_correction and pca_reorder and approx_steps:
        raise ValueError(
            "bias_correction + pca_reorder + approx_steps is incoherent: "
            "'approx' is in _SELECTION_RECOMPUTE_MODES (token_sparse.py:1148), "
            "so it recomputes the PCA permutation and bumps _perm_generation, "
            "and drop_stale_residual() then discards the residual from that "
            "step onward -- bias correction would quietly stop working. "
            "('update' is deliberately excluded from that list and is fine.)"
        )
    if sink_tokens and use_chunk_selector:
        raise ValueError(
            "sink_tokens>0 cannot be combined with use_chunk_selector=True: "
            "the chunked selector does not accept sink_ranges and would silently ignore the sink.")
    if chunk_selector_size <= 0:
        raise ValueError(
            f"chunk_selector_size must be positive, got {chunk_selector_size}")
    if use_chunk_selector and chunk_selector_size % TILE_N:
        # The chunk is walked in tile_n-sized steps; a ragged tail would be
        # selected over but never aligned, so keep it a whole number of tiles.
        raise ValueError(
            f"chunk_selector_size={chunk_selector_size} must be a multiple of "
            f"{TILE_N}")

    if cache_offload not in ("none", "residual"):
        raise ValueError(
            f"cache_offload must be 'none' or 'residual', got {cache_offload!r}")
    if offload_prefetch_ahead not in (0, 1):
        # token_sparse.py reads this via getattr and only implements 0 or 1;
        # a larger number would silently behave as 1.
        raise ValueError(
            f"offload_prefetch_ahead only supports 0 or 1, got "
            f"{offload_prefetch_ahead}")
    if cache_offload == "residual" and not bias_correction:
        # _residual_offload_enabled() (token_sparse.py:2036-2039) requires
        # bias_correction, so this combination is a no-op with no warning --
        # you would think residual was being offloaded when there is no residual.
        raise ValueError(
            "cache_offload='residual' requires bias_correction=True: there is "
            "no residual to offload otherwise, and _residual_offload_enabled() "
            "would silently return False (token_sparse.py:2036-2039). Pass "
            "cache_offload='none' if you meant to turn bias correction off.")

    sched = build_schedule(num_steps=num_steps, dense_steps=dense_steps,
                           dense_layers=dense_layers,
                           calibrate_every=calibrate_every,
                           calibrate_steps=calibrate_steps)
    validate_schedule(num_steps=num_steps,
                      calibrate_steps=sched["calibrate_steps"],
                      update_steps=update_steps, approx_steps=approx_steps)

    import torch
    from mcsparse.kernels.flash_mcsa.token_sparse import TokenSparseConfig, TokenSparseContext

    config = TokenSparseConfig(
        topk=topk,
        topp=topp,
        selection_mode=selection_mode,
        sink_ranges=[(0, sink_tokens)] if sink_tokens > 0 else None,
        calibrate_steps=sched["calibrate_steps"],
        update_steps=update_steps,
        approx_steps=approx_steps,
        full_steps=sched["full_steps"],
        full_layers=sched["full_layers"],
        share_lse=False,          # asserted incompatible with pca_reorder
        # CP uses local heads, so adaptive_heads is not exposed by this dispatcher.
        # Standalone TokenSparseConfig callers can still enable it outside CP.
        adaptive_heads=False,
        adaptive_layers=adaptive_layers,
        use_chunk_selector=use_chunk_selector,
        chunk_selector_size=chunk_selector_size,
        offload_indices=offload_indices,
        cache_offload=cache_offload,
        bias_correction=bias_correction,
        pca_reorder=pca_reorder,
        pca_pow_iters=pca_pow_iters,
        pca_proj_dim=pca_proj_dim,
        full_attn_impl=full_attn_impl,
    )
    # NOT a TokenSparseConfig field -- only the block-level MCSAConfig declares it
    # (config.py:81). The token path reads it with
    # getattr(self.config, 'offload_prefetch_ahead', 1) at token_sparse.py:2050,
    # so it has to be attached after construction; passing it to the dataclass
    # raises TypeError.
    config.offload_prefetch_ahead = int(offload_prefetch_ahead)

    # Releasing any previous context BEFORE building the new one is load-bearing,
    # not tidiness. Callers legitimately rebuild the dispatch per sample
    # (a sweep may call install() once per prompt), and each
    # rebuild used to overwrite _CTX["ctx"] while the old context still held 50
    # layers of cache. reset_mcsa_token_state() then reset the *new* context --
    # already empty -- so the old one leaked indices + residual, ~59 GB per
    # sample at H3's shape. Sample 1 passed, sample 2 started ~59 GB in the hole
    # and OOM'd. Order matters: free first, then allocate.
    _old = _CTX.get("ctx")
    if _old is not None:
        _old.release_residuals()   # drops indices, LSE, residual, CPU copies
        _old.unpatch()             # we never patch modules, but keep it symmetric
        _CTX["ctx"] = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ctx = TokenSparseContext(config)
    _CTX["ctx"] = ctx
    # Cleared on every rebuild: a second construction with a different
    # num_layers would otherwise keep the first one's value and mis-assert.
    _CTX["declared_layers"] = None
    _CTX["layers_per_step"] = None
    _CTX["shapes"] = {}
    _CTX["sparse_seq_len"] = None
    _CTX["seen_seq_lens"] = {}
    _CTX["step"] = -1
    _CTX["layer"] = 0
    _CTX["tick"] = 0
    _CTX["ticks_per_step"] = max(1, int(ticks_per_step))
    _CTX["cfg_slots"] = tuple(cfg_slots)
    _STATS["sink_tokens"] = sink_tokens
    _STATS["nominal_density"] = float(topk)
    full_layers_set = frozenset(sched["full_layers"])

    if num_layers is not None:
        ctx._nl = int(num_layers)
        for i in range(int(num_layers)):
            ctx._gc(i)
        _CTX["declared_layers"] = int(num_layers)
        _CTX["layers_per_step"] = int(num_layers)
        _STATS["layers_per_step"] = int(num_layers)

    print(f"[mcsa-token] topk={topk} sink={sink_tokens} pca={pca_reorder} "
          f"bias={bias_correction} "
          + (f"full_steps=0-{dense_steps - 1} " if dense_steps
             else "full_steps=none ")
          + f"full_layers={sched['full_layers'] or 'none'} "
          f"calibrate={sched['calibrate_steps']}"
          + (f" (step {sched['free_calibrate_step']} is free: full_calibrate)"
             if sched["free_calibrate_step"] is not None
             else " (not free: no dense step to fold it into, so it costs one "
                  "extra full-attention pass)"),
          flush=True)

    # Where full_attn_impl reaches, stated once at startup rather than left to
    # be discovered from a profile. It governs pure `full` steps only. Calibrate
    # steps go through _full_cal, which always uses FA3 -- it needs the output
    # and the LSE from one pass, and FA3 is the only implementation here that
    # returns both. So on a calibrate step, full_attn_impl='sdpa' does not make
    # that step run on SDPA.
    _covers_full = [s for s in (sched["full_steps"] or [])
                    if s not in set(sched["calibrate_steps"])]
    print(f"[mcsa-token] full_attn_impl={config._full_attn_resolved!r} applies to "
          f"{len(_covers_full)} of {len(sched['full_steps'] or [])} full steps; "
          f"calibrate steps always use FA3",
          flush=True)
    print("[mcsa-token] calibrate full-attn = FA3 (output+LSE in one pass)",
          flush=True)

    if offload_indices or cache_offload == "residual":
        _instrument_offload()

    if offload_indices and cache_offload == "residual":
        print(f"[mcsa-token] offload: indices + residual (bf16). "
              f"Saves ~59 GB across 50 layers, but both cross the same PCIe link "
              f"(~1.2 GB/layer, i.e. H2D on the order of the sparse kernel "
              f"itself) and prefetch only reaches one layer ahead. If this run is "
              f"slower than with one of them off, that is why -- memory was "
              f"traded for bandwidth.", flush=True)
    elif offload_indices or cache_offload == "residual":
        print(f"[mcsa-token] offload: "
              f"{'indices' if offload_indices else 'residual'} only",
              flush=True)

    def _splice_prefix_dense(out, query, key, value, n_sink):
        """Recompute the prefix query rows densely and splice them over `out`.

        ``sink_ranges`` handles the *key* side -- the prefix is always attended.
        This is the *query* side: the prefix rows attend the full sequence
        exactly, which ``adapter.py:6`` says is "required for multimodal joint
        attention". A KV-side-only sink covers just this half; here we
        get both, and the two halves come from different mechanisms.

        Why the whole sequence goes through the sparse path first, rather than
        slicing Q down to the video suffix: the dense kernel requires
        ``L_q == L_k`` (``csrc/attention/entry.cc:72``), and the calibrate step
        needs that kernel. Slicing Q would make the calibrate step fail outright.
        Same reasoning as ``hunyuan_patch._attend:238-244``. The prefix rows are
        therefore computed twice and one copy discarded.

        The recomputation uses the *original* dispatch rather than FlashMCSA's
        own ``full_attention``, so the spliced rows are bit-identical to the
        dense arm we are compared against. (``hunyuan_patch`` reaches for
        ``_full_attn`` only because, living inside the library, it has no
        original dispatch to reuse.)
        """
        if n_sink <= 0:
            return out
        q_prefix = query[:, :n_sink].contiguous()
        dense_prefix = original_dispatch(
            q_prefix, key, value, None, 0.0, False, None, False, None,
            backend=None, parallel_config=None)
        if dense_prefix.shape[:2] != (query.shape[0], n_sink):
            raise RuntimeError(
                f"dense prefix attention returned {tuple(dense_prefix.shape)}, "
                f"expected batch={query.shape[0]} rows={n_sink}. The H3 dense "
                f"dispatch appears not to accept L_q != L_k, so the prefix "
                f"cannot be recomputed densely this way."
            )
        # cat rather than index assignment: `out` may be a kernel output buffer
        # and in-place rewriting is hard to reason about (hunyuan_patch.py:264-266).
        _STATS["splice_calls"] += 1
        return torch.cat([dense_prefix.to(out.dtype), out[:, n_sink:]], dim=1)

    def dispatch(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        attention_kwargs=None,
        *,
        backend=None,
        parallel_config=None,
    ):
        _STATS["dispatch_calls"] += 1

        def _dense():
            # Timed under "dense.guard". Without this the 490 dense-guard calls
            # (steps 0-8 plus layer 0 of every step) are a blind spot: they are
            # ~40% of all dispatch calls and the whole of the first denoising
            # step, so "why is step 1 slow" could not be answered from a
            # profile. FLASH_MCSA_TIMING=1 to see it; zero cost when off.
            from mcsparse.kernels.flash_mcsa.timing import timed as _t
            with _t("dense.guard"):
                return original_dispatch(
                    query, key, value, attn_mask, dropout_p, is_causal, scale,
                    enable_gqa, attention_kwargs,
                    backend=backend, parallel_config=parallel_config,
                )

        is_4d = getattr(query, "ndim", 0) == 4
        if is_4d:
            _CTX["seen_seq_lens"][query.shape[1]] = (
                _CTX["seen_seq_lens"].get(query.shape[1], 0) + 1)
        ok, reason, severity = check_eligibility(
            ndim=getattr(query, "ndim", 0),
            is_cuda=bool(getattr(query, "is_cuda", False)),
            dtype_is_bf16=(query.dtype == torch.bfloat16),
            shapes_match=(query.shape == key.shape == value.shape),
            head_dim=query.shape[-1] if is_4d else -1,
            seq_len_q=query.shape[1] if is_4d else -1,
            seq_len_k=key.shape[1] if getattr(key, "ndim", 0) == 4 else -1,
            has_attn_mask=attn_mask is not None,
            is_causal=bool(is_causal),
            dropout_p=float(dropout_p or 0.0),
            enable_gqa=bool(enable_gqa),
            has_parallel_config=parallel_config is not None,
            scale=scale,
            sink_tokens=sink_tokens,
            sparse_seq_len=_CTX["sparse_seq_len"],
            min_seq_len=min_seq_len,
            discovering=_discovering(),
        )
        if not ok:
            if severity == "raise":
                raise RuntimeError(f"[mcsa-token] {reason}")
            _STATS["fallback_calls"] += 1
            # An ineligible call consumes no layer index -- it is not part of the
            # sparse layer sequence. This is what keeps the numbering intact when
            # H3 interleaves its short-sequence attention with the long one.
            return _dense()

        # BTHD [B, T, H, D]
        B, T, H, D = query.shape
        first = _CTX["sparse_seq_len"] is None
        _maybe_upgrade_seq_len(T, B, sink_tokens, topk, min_seq_len)
        eff = _STATS["effective_density"]
        if first:
            others = sorted(L for L in _CTX["seen_seq_lens"] if L != T)
            if others:
                print(f"[mcsa-token] sparsifying L={T}; leaving "
                      f"{others} to dense (below the {min_seq_len} floor or a "
                      f"second long length). H3 runs attention at more than one "
                      f"length per forward, so this is expected -- the short "
                      f"calls are a negligible share of the work.",
                      flush=True)
            if eff > 0.5:
                print(f"[mcsa-token] WARNING: effective density {eff:.3f} "
                      f"(sink {sink_tokens}/{T} = {sink_tokens / T:.3f} plus "
                      f"topk {topk}). The sink adds to the budget rather than "
                      f"consuming it (token_sparse.py:1076-1089), so this arm is "
                      f"attending over half the sequence -- comparisons against "
                      f"other arms' nominal density are not like-for-like.",
                      file=sys.stderr, flush=True)

        li = _note_layer((B, T, H, D))

        # full_layers is enforced here by the caller, not by cache.mode.
        # Route to original_dispatch (not FlashMCSA's full_attention) so these
        # layers stay bit-identical to the dense arm.
        #
        # The layer check comes FIRST and unconditionally.
        # _should_forward_full_step_to_sparse_path() is a *step*-level predicate
        # (token_sparse.py:1982-1990) -- on a step that is both full and
        # calibrate it returns True for every layer, so folding it in as
        # `is_full(li) and not should_forward()` let full_layers slip through on
        # exactly that step. Those layers then built a cache they never used
        # (later reuse steps route them dense again), which showed up as
        # full_calibrate_calls=50 instead of 48 and ~2.4 GB of dead cache.
        if li in full_layers_set:
            _STATS["dense_guard_calls"] += 1
            return _dense()
        if ctx.is_full(li) and not ctx._should_forward_full_step_to_sparse_path():
            _STATS["dense_guard_calls"] += 1
            return _dense()

        mode = ctx._gc(li).mode
        if mode == "calibrate":
            _STATS["calibrate_calls"] += 1
        elif mode == "full_calibrate":
            _STATS["full_calibrate_calls"] += 1

        # BTHD -> BHSD as a *view*: TokenSparseBackend.forward already calls
        # .contiguous() on all three unconditionally (token_sparse.py:1184), so
        # materialising here would just do it twice.
        out = ctx.forward(li, query.transpose(1, 2), key.transpose(1, 2),
                          value.transpose(1, 2))
        # Back to BTHD. contiguous is required: the H3 processor flattens dims
        # 2-3 right after this (pisa_h3.py docstring, item 1).
        out = out.transpose(1, 2).contiguous().type_as(query)

        out = _splice_prefix_dense(out, query, key, value, sink_tokens)
        _STATS["kernel_calls"] += 1
        return out

    return dispatch


__all__ = [
    "get_mcsa_token_stats",
    "make_mcsa_token_dispatch",
    "mcsa_token_begin_forward",
    "reset_mcsa_token_state",
]

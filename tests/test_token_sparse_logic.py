"""Pure-logic verification of mcsa_token_h3 (no torch, no CUDA, no diffusers).

Modelled on FlashMCSA/tests/test_hunyuan_logic.py. Two kinds of test here:

  * Logic we own -- schedule, counters, sink arithmetic, eligibility.
  * Facts about FlashMCSA that mcsa_token_h3 silently depends on (section 8).
    Those are checked by reading the FlashMCSA source, so that if upstream
    changes one, the failure names the assumption instead of showing up as an
    unexplained quality regression in a 40-minute H3 run.

What this CANNOT check: anything numerical, anything needing a GPU, and the
three facts about H3 itself that are not knowable on this box (head_dim, CFG
behaviour, layer count). Those are listed in mcsa_token_h3's docstring.

Run: python3 tests/test_mcsa_token_h3_logic.py
"""
import ast
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# The module under test, at its path in this package.
_SELF = os.path.join(_ROOT, "mcsparse", "h3", "token_sparse_dispatch.py")
_PISA = os.environ.get("PISA_H3", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The kernels are vendored, so the source-invariant checks read them from
# inside the package. $FLASHMCSA_ROOT still redirects to a working copy.
_VENDORED = os.path.join(_ROOT, "mcsparse", "kernels", "flash_mcsa")
_MCSA_ROOT = os.environ.get("FLASHMCSA_ROOT")
_TOKEN_SPARSE = (os.path.join(_MCSA_ROOT, "flash_mcsa", "token_sparse.py")
                 if _MCSA_ROOT else os.path.join(_VENDORED, "token_sparse.py"))
# config.py (the block-level MCSAConfig) is intentionally absent: that path was
# removed along with MCSAContext/MCSACache/backends. H3 is token-level only.

from mcsparse.h3 import token_sparse_dispatch as M

FAIL = []
SKIP = []


def check(name, cond, extra=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {extra}")
        FAIL.append(name)


def raises(name, fn, needle=""):
    try:
        fn()
    except Exception as e:
        if needle and needle.lower() not in str(e).lower():
            print(f"  FAIL  {name} (raised, but not about {needle!r}: {e})")
            FAIL.append(name)
        else:
            print(f"  PASS  {name}")
        return
    print(f"  FAIL  {name} (did not raise)")
    FAIL.append(name)


def _src(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


# --------------------------------------------------------------------------
# 1. schedule builder
# --------------------------------------------------------------------------
def test_schedule():
    print("\n[1] build_schedule")
    s = M.build_schedule(num_steps=50, dense_steps=10, dense_layers="0-1",
                         calibrate_every=10)
    check("full_steps == 0..9", s["full_steps"] == list(range(10)), s["full_steps"])
    check("full_layers == [0,1]", s["full_layers"] == [0, 1], s["full_layers"])
    check("calibrate_steps == [9,19,29,39]",
          s["calibrate_steps"] == [9, 19, 29, 39], s["calibrate_steps"])

    # THE load-bearing property: the first calibrate is also the last dense
    # step, so full_calibrate runs it for free. If this ever stops holding, the
    # schedule silently starts paying for a full attention it used to get free.
    check("step 9 is in BOTH full_steps and calibrate_steps (free calibrate)",
          9 in s["full_steps"] and 9 in s["calibrate_steps"])
    check("free_calibrate_step reported as 9", s["free_calibrate_step"] == 9,
          s["free_calibrate_step"])

    # No calibrate on the final step: nothing would ever reuse its selection.
    check("no calibrate on the last step",
          (50 - 1) not in s["calibrate_steps"], s["calibrate_steps"])

    print("\n[1a] pinned calibrate_steps (the configured default)")
    p = M.build_schedule(num_steps=50, dense_steps=10, dense_layers="0-1",
                         calibrate_every=10, calibrate_steps=(9, 25))
    check("pinned calibrate_steps used verbatim",
          p["calibrate_steps"] == [9, 25], p["calibrate_steps"])
    check("calibrate_every ignored when pinned",
          p["calibrate_steps"] == [9, 25])
    check("step 9 still free when pinned", p["free_calibrate_step"] == 9,
          p["free_calibrate_step"])
    check("module default is (9,)", M.DEFAULT_CALIBRATE_STEPS == (9,),
          M.DEFAULT_CALIBRATE_STEPS)
    check("module default topk is 0.25", M.DEFAULT_TOPK == 0.25, M.DEFAULT_TOPK)

    # The shipped default: one calibration, on the step that is free.
    one = M.build_schedule(num_steps=50, dense_steps=10, dense_layers="0-1",
                           calibrate_every=10,
                           calibrate_steps=M.DEFAULT_CALIBRATE_STEPS)
    check("single calibrate accepted", one["calibrate_steps"] == [9],
          one["calibrate_steps"])
    check("that single calibrate is the free one",
          one["free_calibrate_step"] == 9, one["free_calibrate_step"])
    check("it is in full_steps, so full_calibrate runs it at no extra cost",
          9 in one["full_steps"])

    p2 = M.build_schedule(num_steps=50, dense_steps=10, dense_layers="0-1",
                          calibrate_every=10, calibrate_steps=(25, 9, 25))
    check("pinned steps are sorted and deduped",
          p2["calibrate_steps"] == [9, 25], p2["calibrate_steps"])

    # A pinned calibrate that misses every dense step forfeits the free one --
    # legal, but it should be reported as not-free rather than claimed as free.
    p3 = M.build_schedule(num_steps=50, dense_steps=10, dense_layers="0-1",
                          calibrate_every=10, calibrate_steps=(12, 25))
    check("pinned calibrate outside full_steps -> no free calibrate",
          p3["free_calibrate_step"] is None, p3["free_calibrate_step"])

    raises("empty pinned calibrate_steps rejected",
           lambda: M.build_schedule(num_steps=50, dense_steps=10,
                                    dense_layers="0-1", calibrate_every=10,
                                    calibrate_steps=()),
           "empty")

    s0 = M.build_schedule(num_steps=50, dense_steps=0, dense_layers="",
                          calibrate_every=10)
    check("dense_steps=0 -> full_steps None", s0["full_steps"] is None)
    check("dense_steps=0 -> calibrate starts at 0",
          s0["calibrate_steps"][0] == 0, s0["calibrate_steps"])
    check("dense_steps=0 -> no free calibrate",
          s0["free_calibrate_step"] is None)

    s1 = M.build_schedule(num_steps=8, dense_steps=4, dense_layers="0",
                          calibrate_every=100)
    check("calibrate_every > num_steps -> single calibrate",
          s1["calibrate_steps"] == [3], s1["calibrate_steps"])

    for st in M.build_schedule(num_steps=50, dense_steps=10, dense_layers="0-1",
                               calibrate_every=7)["calibrate_steps"]:
        if st >= 50:
            check("all calibrate steps < num_steps", False, st)
            break
    else:
        check("all calibrate steps < num_steps", True)

    raises("num_steps=0 rejected",
           lambda: M.build_schedule(num_steps=0, dense_steps=0,
                                    dense_layers="", calibrate_every=10),
           "num_steps")

    print("\n[1b] validate_schedule")
    raises("overlapping calibrate/update rejected",
           lambda: M.validate_schedule(num_steps=50, calibrate_steps=[9, 19],
                                       update_steps=[19], approx_steps=None),
           "overlap")
    raises("out-of-range step rejected",
           lambda: M.validate_schedule(num_steps=10, calibrate_steps=[9, 19],
                                       update_steps=None, approx_steps=None),
           "outside")
    try:
        # full_steps vs calibrate_steps overlap must NOT be rejected -- that
        # overlap is the full_calibrate path and is the point of the schedule.
        M.validate_schedule(num_steps=50, calibrate_steps=[9, 19, 29, 39],
                            update_steps=None, approx_steps=None)
        check("non-overlapping schedule accepted", True)
    except Exception as e:
        check("non-overlapping schedule accepted", False, e)


# --------------------------------------------------------------------------
# 2. mode oracle, cross-checked against the real TokenSparseCache.mode
# --------------------------------------------------------------------------
def test_mode_oracle():
    print("\n[2] cache.mode agreement")
    src = _src(_TOKEN_SPARSE)
    if src is None:
        print(f"  SKIP  token_sparse.py not found at {_TOKEN_SPARSE}")
        SKIP.append("mode oracle")
        return

    # Pull the real `mode` property body out of the source and run it against a
    # stub config. Importing the module would need torch.
    start = src.index("    @property\n    def mode(self):")
    end = src.index("    @property\n    def has_cache", start)
    body = src[start:end].replace("    @property\n", "", 1)
    body = "\n".join(line[4:] if line.startswith("    ") else line
                     for line in body.splitlines())
    ns = {}
    exec(body, ns)
    real_mode = ns["mode"]

    sched = M.build_schedule(num_steps=50, dense_steps=10, dense_layers="0-1",
                             calibrate_every=10, calibrate_steps=(9, 25))

    class Cfg:
        _full_set = set(sched["full_steps"])
        _calibrate_set = set(sched["calibrate_steps"])
        _approx_set = None
        _update_set = None
        _cal_ds_q = 1
        _cal_ds_k = 1

    class Cache:
        def __init__(self, step, has_cache):
            self._step_count = step
            self.config = Cfg()
            self.has_cache = has_cache

    def expected(step, has_cache):
        if step in Cfg._full_set:
            if step in Cfg._calibrate_set:
                return "full_calibrate"
            return "full"
        if step in Cfg._calibrate_set:
            return "calibrate"
        if not has_cache:
            return "calibrate"
        return "reuse"

    mismatches = []
    for step in range(50):
        has_cache = step > 9
        got = real_mode(Cache(step, has_cache))
        want = expected(step, has_cache)
        if got != want:
            mismatches.append((step, got, want))
    check("our mode model agrees with token_sparse.py for all 50 steps",
          not mismatches, mismatches[:5])

    check("step 9 resolves to full_calibrate",
          real_mode(Cache(9, False)) == "full_calibrate")
    check("steps 0-8 resolve to full",
          all(real_mode(Cache(s, False)) == "full" for s in range(9)))
    check("step 25 resolves to calibrate (pinned)",
          real_mode(Cache(25, True)) == "calibrate")
    check("step 19 is now reuse, not calibrate (only 9 and 25 calibrate)",
          real_mode(Cache(19, True)) == "reuse")
    check("step 26 resolves to reuse",
          real_mode(Cache(26, True)) == "reuse")
    # With only two calibrations, step 25's residual serves every step after it.
    reuse_after_25 = [s for s in range(26, 50)
                      if real_mode(Cache(s, True)) != "reuse"]
    check("all 24 steps after 25 are reuse", not reuse_after_25, reuse_after_25)


# --------------------------------------------------------------------------
# 3. layer counter
# --------------------------------------------------------------------------
def test_layer_counter():
    print("\n[3] layer counter")
    N = 6
    shape = (1, 4096, 8, 128)

    M.reset_mcsa_token_state()
    M._CTX["ctx"] = None
    M.mcsa_token_begin_forward()
    got = [M._note_layer(shape) for _ in range(N)]
    check("first step numbers layers 0..N-1", got == list(range(N)), got)
    check("layers_per_step not locked until the step ends",
          M._CTX["layers_per_step"] is None)

    M.mcsa_token_begin_forward()
    check("layers_per_step locked to N at next begin",
          M._CTX["layers_per_step"] == N, M._CTX["layers_per_step"])
    got2 = [M._note_layer(shape) for _ in range(N)]
    check("second step restarts at 0", got2 == list(range(N)), got2)

    # One call too many, mid-step.
    raises("N+1 calls in a step raises", lambda: M._note_layer(shape),
           "attention call")

    # A short step is caught at the next begin_forward.
    M.reset_mcsa_token_state()
    M._CTX["ctx"] = None
    M.mcsa_token_begin_forward()
    for _ in range(N):
        M._note_layer(shape)
    M.mcsa_token_begin_forward()
    for _ in range(N - 1):
        M._note_layer(shape)
    raises("short step raises at next begin", M.mcsa_token_begin_forward,
           "layer count changed")

    # Shape change for a FIXED index across steps -- this is how an extra call
    # type (token refiner, cross-attention) sneaking into the dispatch would
    # show up: it shifts the numbering, so some index starts seeing a different
    # tensor than it saw last step.
    M.reset_mcsa_token_state()
    M._CTX["ctx"] = None
    M.mcsa_token_begin_forward()
    for _ in range(N):
        M._note_layer(shape)
    M.mcsa_token_begin_forward()
    raises("same layer index with a different shape next step raises",
           lambda: M._note_layer((1, 2048, 8, 128)), "changed shape")

    # Different indices legitimately may differ in shape -- do not over-reject.
    M.reset_mcsa_token_state()
    M._CTX["ctx"] = None
    M.mcsa_token_begin_forward()
    M._note_layer(shape)
    try:
        M._note_layer((1, 2048, 8, 128))
        check("different indices may have different shapes", True)
    except Exception as e:
        check("different indices may have different shapes", False, e)

    # Zero eligible calls in step 0 is a configuration failure, not a silent 0.
    M.reset_mcsa_token_state()
    M._CTX["ctx"] = None
    M.mcsa_token_begin_forward()
    raises("no eligible calls in step 0 raises", M.mcsa_token_begin_forward,
           "no eligible attention calls")

    print("\n[3a] interleaved short/long calls (H3's real pattern)")
    # The regression the user hit: H3 emits 301-long and 38011-long attention in
    # ONE forward. Dense-routed short calls must NOT consume a layer index, or
    # the long calls' numbering shifts and each layer gets another layer's
    # cached token selection -- silently, since check_seq_len only compares
    # ceil(L/BLKQ), which is equal across layers.
    M.reset_mcsa_token_state()
    M._CTX["ctx"] = None
    long_shape = (1, 38011, 8, 128)
    M.mcsa_token_begin_forward()
    # Only the long calls reach _note_layer; the short ones returned dense before
    # it, which is what the dispatch does now.
    seen = [M._note_layer(long_shape) for _ in range(N)]
    check("long-call indices are 0..N-1 despite interleaved short calls",
          seen == list(range(N)), seen)
    M.mcsa_token_begin_forward()
    seen2 = [M._note_layer(long_shape) for _ in range(N)]
    check("numbering repeats identically next step", seen2 == list(range(N)), seen2)
    check("only the long length was recorded in shapes{}",
          all(s == long_shape for s in M._CTX["shapes"].values()),
          M._CTX["shapes"])
    check("layers_per_step counts only sparse calls",
          M._CTX["layers_per_step"] == N, M._CTX["layers_per_step"])


# --------------------------------------------------------------------------
# 4. step clock
# --------------------------------------------------------------------------
class FakeCtx:
    """Stands in for TokenSparseContext, tracking _sc the same way it does."""

    def __init__(self):
        self._sc = 0
        self._nl = None
        self.steps = 0
        self.resets = 0
        self._caches = {}
        self._cache_slots = {"_default": self._caches}
        self._active_slot = "_default"
        self._slse = None
        self._slse_per_slot = {}

    def step(self):
        self._sc += 1
        self.steps += 1

    def reset(self):
        self._sc = 0
        self.resets += 1

    def _gc(self, li):
        self._caches.setdefault(li, object())
        return self._caches[li]


def test_step_clock():
    print("\n[4] step clock")
    fake = FakeCtx()
    M.reset_mcsa_token_state()
    M._CTX["ctx"] = fake
    M._CTX["ticks_per_step"] = 1
    M._CTX["cfg_slots"] = ()

    M.mcsa_token_begin_forward()
    check("first tick -> step 0", M._CTX["step"] == 0, M._CTX["step"])
    check("first tick does NOT call ctx.step()", fake.steps == 0, fake.steps)
    check("our step == ctx._sc after tick 1", M._CTX["step"] == fake._sc)

    for _ in range(3):
        M._note_layer((1, 4096, 8, 128))

    drift = []
    for i in range(1, 10):
        M.mcsa_token_begin_forward()
        for _ in range(3):
            M._note_layer((1, 4096, 8, 128))
        if M._CTX["step"] != fake._sc:
            drift.append((i, M._CTX["step"], fake._sc))
    check("our step tracks ctx._sc for 10 ticks", not drift, drift)
    check("after 10 ticks step == 9", M._CTX["step"] == 9, M._CTX["step"])
    check("ctx.step() called exactly 9 times", fake.steps == 9, fake.steps)
    check("steps_ticked stat == 10", M.get_mcsa_token_stats()["steps_ticked"] == 10)

    M.reset_mcsa_token_state()
    check("reset -> step -1", M._CTX["step"] == -1)
    check("reset calls ctx.reset()", fake.resets == 1, fake.resets)

    # Serial CFG: two forwards per denoising step, alternating cache slots.
    fake2 = FakeCtx()
    M.reset_mcsa_token_state()
    M._CTX["ctx"] = fake2
    M._CTX["ticks_per_step"] = 2
    M._CTX["cfg_slots"] = ("cond", "uncond")
    slots = []
    for _ in range(10):
        M.mcsa_token_begin_forward()
        slots.append(fake2._active_slot)
        for _ in range(3):
            M._note_layer((1, 4096, 8, 128))
    check("ticks_per_step=2 -> 10 ticks give 5 steps",
          M._CTX["step"] == 4, M._CTX["step"])
    check("slots alternate cond/uncond",
          slots == ["cond", "uncond"] * 5, slots)
    check("each CFG pass gets its own cache slot",
          set(fake2._cache_slots) >= {"cond", "uncond"},
          list(fake2._cache_slots))

    M._CTX["ticks_per_step"] = 1
    M._CTX["cfg_slots"] = ()
    M.reset_mcsa_token_state()

    # The caller resets AFTER building the dispatch, so reset must not wipe the
    # stats that describe the configuration -- otherwise every report reads
    # "sink=0", which is the one value the arm refuses to start with.
    M._STATS["sink_tokens"] = 951
    M._STATS["nominal_density"] = 0.1
    M._STATS["kernel_calls"] = 123
    M.reset_mcsa_token_state()
    s = M.get_mcsa_token_stats()
    check("reset keeps sink_tokens (config, not run state)",
          s["sink_tokens"] == 951, s["sink_tokens"])
    check("reset keeps nominal_density", s["nominal_density"] == 0.1,
          s["nominal_density"])
    check("reset clears kernel_calls (run state)", s["kernel_calls"] == 0,
          s["kernel_calls"])

    # An explicitly declared layer count must survive reset too, or the second
    # arm/sample would silently fall back to re-discovering it.
    M._CTX["declared_layers"] = 40
    M.reset_mcsa_token_state()
    check("reset keeps an explicitly declared layer count",
          M._CTX["layers_per_step"] == 40, M._CTX["layers_per_step"])
    M._CTX["declared_layers"] = None
    M.reset_mcsa_token_state()
    check("reset re-arms discovery when layers were not declared",
          M._CTX["layers_per_step"] is None, M._CTX["layers_per_step"])


# --------------------------------------------------------------------------
# 5. sink arithmetic
# --------------------------------------------------------------------------
def test_sink():
    print("\n[5] sink arithmetic / effective density")
    L = 81920
    # The sink ADDS to the budget rather than consuming it, so effective > topk
    # whenever there is a sink at all. That is the whole reason this is reported.
    e0 = M.effective_density(sink_tokens=0, seq_len=L, topk=0.1)
    e1 = M.effective_density(sink_tokens=951, seq_len=L, topk=0.1)
    check("no sink -> effective ~= topk", abs(e0 - 0.1) < 0.01, e0)
    check("sink raises effective density above nominal", e1 > e0, (e0, e1))
    check("effective ~= (P + topk*L)/L",
          abs(e1 - (951 + 0.1 * L) / L) < 0.01, e1)

    # A big prefix is exactly the case that makes a topk-vs-density comparison
    # against another method's nominal density misleading.
    big = M.effective_density(sink_tokens=int(0.25 * L), seq_len=L, topk=0.1)
    check("25% prefix + topk 0.1 -> ~0.35 effective", 0.33 < big < 0.37, big)
    check("effective density is capped at 1.0",
          M.effective_density(sink_tokens=L, seq_len=L, topk=0.5) <= 1.0)
    check("empty sequence -> 0.0", M.effective_density(
        sink_tokens=0, seq_len=0, topk=0.1) == 0.0)


# --------------------------------------------------------------------------
# 6. prefix splice index arithmetic
# --------------------------------------------------------------------------
def test_splice():
    print("\n[6] prefix splice")
    # Model the seq dim as a list, as test_hunyuan_logic does. H3's prefix is at
    # the FRONT (Hunyuan's text is at the tail) -- getting this backwards would
    # overwrite video rows with prefix attention and leave the conditioning
    # rows sparse, which is the exact failure the sink exists to prevent.
    L, P = 10, 3
    sparse = [f"s{i}" for i in range(L)]
    dense = [f"d{i}" for i in range(P)]
    merged = dense + sparse[P:]
    check("length preserved", len(merged) == L, len(merged))
    check("prefix rows come from dense", merged[:P] == dense, merged[:P])
    check("suffix rows untouched", merged[P:] == sparse[P:], merged[P:])
    check("no dense row leaks past the prefix",
          not any(x.startswith("d") for x in merged[P:]))
    check("P=0 -> untouched", ([] + sparse[0:]) == sparse)


# --------------------------------------------------------------------------
# 7. eligibility truth table
# --------------------------------------------------------------------------
def test_eligibility():
    print("\n[7] eligibility")
    good = dict(ndim=4, is_cuda=True, dtype_is_bf16=True, shapes_match=True,
                head_dim=128, seq_len_q=38011, seq_len_k=38011,
                has_attn_mask=False, is_causal=False, dropout_p=0.0,
                enable_gqa=False, has_parallel_config=False, scale=None,
                sink_tokens=951, sparse_seq_len=None, min_seq_len=4096)
    ok, reason, _ = M.check_eligibility(**good)
    check("the all-good case is eligible", ok, reason)

    fallbacks = [
        ("parallel_config", dict(has_parallel_config=True)),
        ("attn_mask", dict(has_attn_mask=True)),
        ("is_causal", dict(is_causal=True)),
        ("dropout", dict(dropout_p=0.1)),
        ("enable_gqa", dict(enable_gqa=True)),
        ("ndim != 4", dict(ndim=3)),
        ("not cuda", dict(is_cuda=False)),
        ("not bf16", dict(dtype_is_bf16=False)),
        ("shape mismatch", dict(shapes_match=False)),
        ("L_q != L_k", dict(seq_len_q=256, seq_len_k=38011)),
        ("head_dim 64", dict(head_dim=64)),
        ("head_dim 96", dict(head_dim=96)),
        ("non-default scale", dict(scale=0.5)),
    ]
    for name, override in fallbacks:
        ok, reason, sev = M.check_eligibility(**{**good, **override})
        check(f"{name} -> fallback", (not ok) and sev == "fallback",
              f"ok={ok} sev={sev} reason={reason}")

    print("\n[7a] H3's multiple sequence lengths (the bug that bit on real H3)")
    # H3 issues attention at 301 AND 38011 in ONE forward. Treating the change as
    # a missing reset aborted the run; the short one must simply go dense.
    ok, reason, sev = M.check_eligibility(
        **{**good, "seq_len_q": 301, "seq_len_k": 301, "sink_tokens": 0})
    check("short H3 sequence (301) -> fallback, NOT raise",
          (not ok) and sev == "fallback", f"sev={sev} reason={reason}")
    check("the reason names the floor", "floor" in reason.lower(), reason)

    # Once committed to the long length, the short one keeps going dense rather
    # than raising -- this is the exact traceback the user hit.
    ok, reason, sev = M.check_eligibility(
        **{**good, "seq_len_q": 301, "seq_len_k": 301, "sink_tokens": 0,
           "sparse_seq_len": 38011})
    check("301 after committing to 38011 -> fallback, NOT raise",
          (not ok) and sev == "fallback", f"sev={sev} reason={reason}")

    # A second LONG length cannot share the cached selection, but it is still not
    # clearly a missing reset, so stay dense rather than abort.
    ok, reason, sev = M.check_eligibility(
        **{**good, "seq_len_q": 20000, "seq_len_k": 20000,
           "sparse_seq_len": 38011})
    check("a second long length -> fallback, NOT raise",
          (not ok) and sev == "fallback", f"sev={sev} reason={reason}")
    check("the reason says 'second long'", "second long" in reason, reason)

    # The committed length keeps being eligible across steps.
    ok, _, _ = M.check_eligibility(**{**good, "sparse_seq_len": 38011})
    check("the committed length stays eligible", ok)

    print("\n[7b] longest-wins commitment (the stroll/tokyo regression)")
    # The text token_refiner's length tracks the prompt (~1.45 tokens per IR
    # character). On a long prompt it clears min_seq_len and, being emitted
    # before the main blocks, used to become the committed length -- after
    # which every packed-sequence call was refused and the arm silently ran at
    # dense speed. Measured: stroll's refiner was 4207 and it fell back 2450
    # times; cat's was 3976, just under the floor, and worked.
    ok, reason, sev = M.check_eligibility(
        **{**good, "seq_len_q": 104179, "seq_len_k": 104179,
           "sparse_seq_len": 4207, "discovering": True})
    check("a LONGER length is accepted while discovering", ok,
          f"reason={reason}")
    ok, reason, sev = M.check_eligibility(
        **{**good, "seq_len_q": 104179, "seq_len_k": 104179,
           "sparse_seq_len": 4207, "discovering": False})
    check("the same length is refused once discovery is over",
          (not ok) and sev == "fallback", f"sev={sev}")
    ok, reason, sev = M.check_eligibility(
        **{**good, "seq_len_q": 4207, "seq_len_k": 4207,
           "sparse_seq_len": 104179, "discovering": True})
    check("a SHORTER length never downgrades the commitment", not ok,
          f"reason={reason}")

    # _maybe_upgrade_seq_len owns the state transition; check it end to end in
    # H3's real order (refiner first, then the packed sequence, per layer).
    for name, refiner, packed in (("stroll", 4207, 104179),
                                  ("cat", 3976, 104534),
                                  ("trooper", 1491, 104179),
                                  ("tokyo", 8548, 106000)):
        M.reset_mcsa_token_state()
        M._CTX["ctx"] = None
        for _ in range(4):
            for L in (refiner, packed):
                ok, _, _ = M.check_eligibility(
                    **{**good, "seq_len_q": L, "seq_len_k": L,
                       "sink_tokens": 0,
                       "sparse_seq_len": M._CTX["sparse_seq_len"],
                       "discovering": M._discovering()})
                if ok:
                    M._maybe_upgrade_seq_len(L, 1, 0, 0.25, 4096)
        check(f"{name}: commits to the packed sequence {packed}",
              M._CTX["sparse_seq_len"] == packed,
              f"got {M._CTX['sparse_seq_len']}")

    # An upgrade must clear the layer bookkeeping, or _note_layer raises on the
    # shape change it exists to catch.
    M.reset_mcsa_token_state()
    M._CTX["ctx"] = None
    M._maybe_upgrade_seq_len(4207, 1, 0, 0.25, 4096)
    M._CTX["layer"] = 7
    M._CTX["shapes"] = {0: (1, 4207, 56, 128)}
    M._maybe_upgrade_seq_len(104179, 1, 0, 0.25, 4096)
    check("an upgrade resets layer and shapes",
          M._CTX["layer"] == 0 and M._CTX["shapes"] == {},
          f"layer={M._CTX['layer']} shapes={M._CTX['shapes']}")

    # Discovery is keyed on "no kernel call yet", not "step 0": dense_steps=0
    # would cache immediately, and a later length change would leave the
    # selections sized to the old length.
    M.reset_mcsa_token_state()
    M._CTX["ctx"] = None
    check("discovering while nothing has been cached", M._discovering())
    M._STATS["kernel_calls"] = 1
    check("not discovering once the kernel has run", not M._discovering())
    M._STATS["kernel_calls"] = 0

    # The floor is never below BLKQ, whatever the caller passes.
    ok, reason, sev = M.check_eligibility(
        **{**good, "seq_len_q": 64, "seq_len_k": 64, "sink_tokens": 0,
           "min_seq_len": 1})
    check("L < BLKQ rejected even with min_seq_len=1",
          (not ok) and sev == "fallback", f"sev={sev} reason={reason}")

    # head_dim 144 is supported and must NOT be rejected.
    ok, _, _ = M.check_eligibility(**{**good, "head_dim": 144})
    check("head_dim 144 is eligible", ok)

    # The default scale for the head dim must be accepted.
    ok, reason, _ = M.check_eligibility(**{**good, "scale": 128 ** -0.5})
    check("scale == 1/sqrt(D) is eligible", ok, reason)

    # Only a genuinely impossible config raises now.
    ok, reason, sev = M.check_eligibility(**{**good, "sink_tokens": 38011})
    check("sink >= L_k -> raise", (not ok) and sev == "raise",
          f"ok={ok} sev={sev} reason={reason}")
    check("module default min_seq_len is 4096",
          M.DEFAULT_MIN_SEQ_LEN == 4096, M.DEFAULT_MIN_SEQ_LEN)
def test_source_invariants():
    print("\n[8] FlashMCSA source invariants")
    ts = _src(_TOKEN_SPARSE)
    if ts is None:
        print(f"  SKIP  FlashMCSA source not found under {_MCSA_ROOT}")
        SKIP.append("source invariants")
        return

    def between(src, start_marker, end_marker):
        i = src.index(start_marker)
        j = src.index(end_marker, i)
        return src[i:j]

    # -- THE load-bearing asymmetry --------------------------------------
    # Block-level forbids pca_reorder + sink_ranges; token-level does not.
    # If upstream ever adds that assert to the token path, this whole arm
    # becomes unconstructible and this test says exactly why.
    tok_post = between(ts, "    def __post_init__(self):", "class TokenSparseCache")
    tok_pca = tok_post[tok_post.index("if self.pca_reorder:"):]
    check("token-level pca_reorder does NOT forbid sink_ranges",
          "sink_ranges" not in tok_pca,
          "upstream added the assert -> mcsa_token_h3 can no longer combine "
          "PCA with H3's required prefix sink")
    check("token-level pca_reorder DOES forbid share_lse",
          "share_lse" in tok_pca)

    # The block-level contrast (MCSAConfig forbids pca_reorder + sink_ranges)
    # used to be asserted here against config.py. That whole path -- MCSAConfig,
    # MCSAContext, MCSACache, backends/ -- has been removed from this repo: H3
    # runs token-level only, so the block-level arm was unreachable code whose
    # assert nothing could trip. The asymmetry it documented is still the reason
    # this integration is token-level, and is stated in this module's docstring
    # and in h3/token_sparse_dispatch.py's header.

    # fp8 and the compiled extension are gone. The config fields went with them
    # rather than becoming validated-and-rejected: a knob you can set but that
    # can only ever fail is worse than no knob. bf16 is not a default here, it
    # is the only path.
    check("TokenSparseConfig has no fp8 field",
          "\n    fp8:" not in ts)
    check("TokenSparseConfig has no residual_dtype field",
          "\n    residual_dtype:" not in ts)
    check("no fp8 plumbing left at the call sites",
          "fp8=cfg.fp8" not in ts and "cfg.fp8" not in ts)
    check("no reference to the compiled extension remains",
          ".ops.attention" not in ts and ".ops.fp8_quant" not in ts
          and "from .ops import attention" not in ts)
    check("calibrate gets output+LSE from one FA3 call",
          "full_attention_with_lse" in ts)

    # -- PCA permutes Q only ---------------------------------------------
    fwd = between(ts, "    def forward(self, q, k, v, cache, mode",
                  "    def _dispatch(self")
    pca_branch = fwd[fwd.index("perm = cache.get_or_make_perm"):]
    check("forward permutes q", "_apply_perm_bhld(q, perm)" in pca_branch)
    check("forward does NOT permute k or v",
          "_apply_perm_bhld(k" not in pca_branch
          and "_apply_perm_bhld(v" not in pca_branch)
    check("dispatch receives the ORIGINAL k, v",
          "self._dispatch(q_perm, k, v" in pca_branch)

    # -- sink lives in K space and is broadcast to every q block ----------
    sel = between(ts, "def _select_topk_tokens(", "def _apply_perm_bhld")
    check("sink mask is built over L_K_ds (key space)",
          "sink_mask = torch.zeros(L_K_ds" in sel)
    check("sink probabilities are zeroed so they cannot be double-selected",
          "token_probs[:, :, :, sink_mask] = 0.0" in sel)
    check("sink indices are concatenated onto the selection "
          "(so they ADD to the budget -- see effective_density)",
          "sink_block" in sel and "torch.cat" in sel)

    # -- approx recomputes the perm, update does not ----------------------
    i = ts.index("_SELECTION_RECOMPUTE_MODES")
    modes = ts[i:ts.index("\n", i)]
    check("_SELECTION_RECOMPUTE_MODES contains 'approx' "
          "(justifies rejecting approx + pca + bias)", "approx" in modes, modes)
    check("_SELECTION_RECOMPUTE_MODES does NOT contain 'update'",
          "'update'" not in modes, modes)

    # -- _full_cal's attention comes from FA3, with no fallback -----------
    # It used to dispatch on head_dim between two _C-backed kernels
    # (dim128/dim144) and fall back to SDPA + a separate logsumexp pass. FA3
    # handles any supported head_dim itself and returns the LSE alongside the
    # output, so there is no dispatch and no second QK^T.
    fc = between(ts, "    def _full_cal(self", "    def _resolve_topk")
    check("_full_cal calls full_attention_with_lse",
          "full_attention_with_lse" in fc)
    check("_full_cal has no SDPA fallback "
          "(calibrate hard-depends on FA3)",
          "scaled_dot_product_attention" not in fc)
    check("_full_cal still asserts the LSE shape",
          "lse.shape == (B, H, L)" in fc)

    # -- full_attn_impl does NOT cover calibrate steps ---------------------
    # bias_correction routes calibrate through _full_cal, whose full attention
    # is the bf16 dim kernel above -- not full_attn_impl. This is why the
    # startup log spells out how many steps 'fa3' actually reaches.
    check("_full_cal does not consult _full_attn_resolved "
          "(so full_attn_impl='fa3' does not cover calibrate steps)",
          "_full_attn_resolved" not in fc)
    fwd_full = between(ts, "        if mode == 'full':", "        if not cache.config.pca_reorder")
    check("mode=='full' DOES consult _full_attn_resolved "
          "(so fa3 covers pure full steps)", "_full_attn_resolved" in fwd_full)

    # -- chunked selector silently disables adaptive_heads ----------------
    check("_full_cal branches on use_chunk_selector before _resolve_topk "
          "(justifies rejecting adaptive_heads + use_chunk_selector)",
          fc.index("use_chunk_selector") < fc.index("_resolve_topk"))
    cal = between(ts, "    def _cal(self, q, k, v, cache, ext_lse):",
                  "    def _upd(self")
    check("_cal branches the same way", "use_chunk_selector" in cal)

    # -- _nl gates residual prefetch and layer_t --------------------------
    pf = between(ts, "    def _prefetch_next_residual_layers",
                 "    def _prefetch_next_sparse_layer")
    check("residual prefetch early-returns when _nl is None "
          "(so we must set ctx._nl ourselves)", "self._nl is None" in pf)
    gc = between(ts, "    def _gc(self, li):", "    def forward(self, li")
    check("_gc sets layer_t only when _nl is not None "
          "(so adaptive_layers needs num_layers up front)",
          "self._nl is not None" in gc and "layer_t" in gc)

    # -- TokenSparseConfig has no step validation -------------------------
    check("TokenSparseConfig has no _validate_steps "
          "(so mcsa_token_h3.validate_schedule must do it)",
          "_validate_steps" not in between(ts, "class TokenSparseConfig",
                                           "class TokenSparseCache"))
    # The MCSAConfig contrast is gone with the block-level path; the point that
    # matters is the one above -- nothing upstream validates the step lists.

    # -- BLKQ / tile_n are pinned ----------------------------------------
    check("BLKQ is asserted == 128 upstream", "assert self.BLKQ == 128" in ts)
    check("tile_n is asserted == 128 upstream", "assert self.tile_n == 128" in ts)
    check("our mirrored constants match", M.BLKQ == 128 and M.TILE_N == 128)


# --------------------------------------------------------------------------
# 9. api surface
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# 10. API surface
# --------------------------------------------------------------------------
def test_api_surface():
    print("\n[10] API surface")
    check("__all__ is the four public names",
          sorted(M.__all__) == ["get_mcsa_token_stats", "make_mcsa_token_dispatch",
                                "mcsa_token_begin_forward",
                                "reset_mcsa_token_state"], M.__all__)
    for name in M.__all__:
        check(f"{name} exists and is callable", callable(getattr(M, name, None)))

    mod = ast.parse(_src(_SELF))

    # No top-level torch / flash_mcsa import: this is what keeps THIS test file
    # runnable on a box with no GPU stack.
    top = []
    for node in mod.body:
        if isinstance(node, ast.Import):
            top += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            top.append(node.module or "")
    bad = [n for n in top if n and (n == "torch" or n.startswith("torch.")
                                    or n.startswith("flash_mcsa")
                                    or "kernels.flash_mcsa" in n)]
    check("no top-level torch/flash_mcsa import", not bad, bad)

    # The dispatch signature, for comparison against a peer backend when one
    # is available.
    def dispatch_sig(path, factory):
        src = _src(path) if path else None
        if src is None:
            return None
        m = ast.parse(src)
        for node in ast.walk(m):
            if isinstance(node, ast.FunctionDef) and node.name == factory:
                for inner in node.body:
                    if isinstance(inner, ast.FunctionDef) and inner.name == "dispatch":
                        a = inner.args
                        return ([p.arg for p in a.args],
                                [p.arg for p in a.kwonlyargs])
        return None

    # Compared against a peer backend's when one is on hand ($PISA_H3), since
    # the two were kept swappable during evaluation. Normally skipped.
    ours = dispatch_sig(_SELF, "make_mcsa_token_dispatch")
    theirs = dispatch_sig(_PISA, "make_pisa_dispatch") if _PISA else None
    if theirs is None:
        SKIP.append("dispatch signature vs a peer backend (set $PISA_H3)")
    else:
        check("dispatch positional args match the peer backend's",
              ours[0] == theirs[0],
              f"{ours[0]} != {theirs[0]}")
        check("dispatch keyword-only args match the peer backend's",
              ours[1] == theirs[1],
              f"{ours[1]} != {theirs[1]}")

    print("\n[10b] construction guards (no torch needed -- they fire first)")
    raises("sink_tokens=0 rejected by default",
           lambda: M.make_mcsa_token_dispatch(None, sink_tokens=0),
           "sink")
    raises("negative sink rejected",
           lambda: M.make_mcsa_token_dispatch(None, sink_tokens=-1), "sink_tokens")
    raises("adaptive_layers without num_layers rejected",
           lambda: M.make_mcsa_token_dispatch(None, sink_tokens=951,
                                              adaptive_layers=True),
           "num_layers")
    raises("pca + bias + approx_steps rejected",
           lambda: M.make_mcsa_token_dispatch(None, sink_tokens=951,
                                              pca_reorder=True,
                                              bias_correction=True,
                                              approx_steps=[5]),
           "approx")
    # The chunked selector does not take sink_ranges, so the sink would be
    # silently dropped rather than applied.
    raises("sink_tokens + use_chunk_selector rejected",
           lambda: M.make_mcsa_token_dispatch(None, sink_tokens=951,
                                              use_chunk_selector=True),
           "sink")

    print("\n[10c] factory defaults match what was asked for")
    import inspect
    d = {n: p.default
         for n, p in inspect.signature(M.make_mcsa_token_dispatch).parameters.items()}
    check("default topk == 0.25", d["topk"] == 0.25, d["topk"])
    check("default calibrate_steps == (9,)",
          d["calibrate_steps"] == (9,), d["calibrate_steps"])
    check("adaptive_heads is not exposed", "adaptive_heads" not in d)
    check("default full_attn_impl == 'fa3'", d["full_attn_impl"] == "fa3",
          d["full_attn_impl"])
    check("default ticks_per_step == 1 (verify with steps_ticked before changing)",
          d["ticks_per_step"] == 1, d["ticks_per_step"])
    check("default pca_reorder is True", d["pca_reorder"] is True)
    check("default bias_correction is True", d["bias_correction"] is True)

    print("\n[10d] memory knobs")
    check("offload_indices defaults on", d["offload_indices"] is True)
    check("cache_offload defaults to 'residual'",
          d["cache_offload"] == "residual", d["cache_offload"])
    check("offload_prefetch_ahead defaults to 1",
          d["offload_prefetch_ahead"] == 1)
    raises("bad cache_offload rejected",
           lambda: M.make_mcsa_token_dispatch(None, sink_tokens=951,
                                              cache_offload="cpu"),
           "cache_offload")
    # Only 0 and 1 are implemented; 2 would silently behave as 1.
    raises("offload_prefetch_ahead=2 rejected",
           lambda: M.make_mcsa_token_dispatch(None, sink_tokens=951,
                                              offload_prefetch_ahead=2),
           "offload_prefetch_ahead")
    # _residual_offload_enabled() requires bias_correction, so this pairing is a
    # silent no-op upstream -- reject it instead of pretending it works.
    raises("cache_offload='residual' without bias_correction rejected",
           lambda: M.make_mcsa_token_dispatch(None, sink_tokens=951,
                                              bias_correction=False,
                                              cache_offload="residual"),
           "bias_correction")

    print("\n[10e] TokenSparseConfig field surface")
    # offload_prefetch_ahead is NOT a TokenSparseConfig field -- only the
    # block-level MCSAConfig declares it, and the token path reads it through
    # getattr. Passing it as a dataclass kwarg raises TypeError, which is why it
    # is attached after construction.
    ts_src = _src(_TOKEN_SPARSE)
    if ts_src is None:
        print("  SKIP  token_sparse.py not found")
        SKIP.append("config fields")
    else:
        import re
        cfg_body = ts_src[ts_src.index("class TokenSparseConfig"):
                          ts_src.index("    def __post_init__")]
        fields = set(re.findall(r"^    ([a-z_]+)\s*:", cfg_body, re.M))
        for f in ("offload_indices", "cache_offload", "use_chunk_selector"):
            check(f"TokenSparseConfig has {f}", f in fields)
        check("TokenSparseConfig does NOT have offload_prefetch_ahead",
              "offload_prefetch_ahead" not in fields)
        # Removed with the fp8 path; see section 8.
        for f in ("fp8", "residual_dtype"):
            check(f"TokenSparseConfig does NOT have {f}", f not in fields)
        check("token_sparse reads it via getattr",
              "getattr(self.config, 'offload_prefetch_ahead'" in ts_src)
        self_src = _src(_SELF)
        check("we attach it after construction, not as a kwarg",
              "config.offload_prefetch_ahead = " in self_src
              and "offload_prefetch_ahead=offload_prefetch_ahead" not in self_src)
        # Residual offload needs bias_correction upstream too -- pin the reason.
        check("_residual_offload_enabled requires bias_correction",
              "and self.config.bias_correction" in ts_src)

    print("\n[10f] rebuild does not leak the previous context")
    # The bug: a sweep may call install(arm) per prompt, so the dispatch is
    # rebuilt per sample. Each rebuild overwrote _CTX["ctx"] while the old
    # context still held 50 layers of cache (~59 GB at H3's shape), and
    # reset_mcsa_token_state() then reset the NEW, already-empty context. Sample
    # 1 passed; sample 2 started 59 GB in the hole and OOM'd.
    self_src2 = _src(_SELF)
    i = self_src2.index("def make_mcsa_token_dispatch")
    body = self_src2[i:]
    rel = body.find("release_residuals()")
    new = body.find("TokenSparseContext(config)")
    check("the old context is released BEFORE the new one is built",
          rel != -1 and new != -1 and rel < new, f"release@{rel} new@{new}")
    check("release_residuals is called on rebuild",
          "_old.release_residuals()" in body)
    check("declared_layers is cleared on rebuild",
          '_CTX["declared_layers"] = None' in body)

    sw2 = _src(os.environ.get("SWEEP_VBENCH50", ""))
    if sw2 is not None:
        check("sweep builds the mcsa_token dispatch once, not per prompt",
              "_dispatch_cache" in sw2)

    print("\n[10g] full_layers are dense even on a calibrate step")
    # The bug: `is_full(li) and not should_forward()` folded a LAYER check
    # together with a STEP predicate. On step 9 (both full and calibrate)
    # should_forward() is True for every layer, so full_layers 0 and 1 slipped
    # into the sparse path -- full_calibrate_calls read 50 instead of 48, and
    # those two layers built a cache that later reuse steps never touched
    # (~2.4 GB of dead cache).
    disp = body[body.index("    def dispatch("):]
    guard = disp[:disp.index("mode = ctx._gc(li).mode")]
    # Compare executable lines only -- the explanatory comment above the guard
    # also names the predicate, and matching that made this test fail on
    # correct code.
    code = "\n".join(ln for ln in guard.splitlines()
                     if not ln.lstrip().startswith("#"))
    li_check = code.find("li in full_layers_set")
    step_check = code.find("_should_forward_full_step_to_sparse_path")
    check("the full_layers check exists and is unconditional",
          li_check != -1, code[-200:] if li_check == -1 else "")
    check("it comes BEFORE the step-level predicate",
          li_check != -1 and step_check != -1 and li_check < step_check,
          f"layer@{li_check} step@{step_check}")
    check("full_layers_set is precomputed from the schedule",
          "full_layers_set = frozenset(sched[" in self_src2)


def main():
    print("=" * 72)
    print("mcsparse token-sparse pure-logic tests")
    _kern = _MCSA_ROOT or _VENDORED
    print(f"kernels: {_kern}"
          f"{'' if os.path.isdir(_kern) else '  (NOT FOUND)'}"
          f"{'' if _MCSA_ROOT else '  [vendored]'}")
    print("=" * 72)

    test_schedule()
    test_mode_oracle()
    test_layer_counter()
    test_step_clock()
    test_sink()
    test_splice()
    test_eligibility()
    test_source_invariants()
    test_api_surface()

    print("\n" + "=" * 72)
    if SKIP:
        print(f"SKIPPED (source not reachable): {', '.join(SKIP)}")
    if FAIL:
        print(f"FAILED ({len(FAIL)}): {', '.join(FAIL)}")
        return 1
    print("all pass")
    print("\nNOT covered here -- needs a GPU and the real model:")
    print("  * whether H3 runs CFG (ticks_per_step); read steps_ticked once")
    print("  * the real prefix length, hence the real effective density")
    print("  * whether H3's dense dispatch accepts L_q != L_k (prefix splice)")
    print("  * every numeric claim: does bias correction help at H3's drift,")
    print("    is a 2-calibration schedule enough, is the speedup real")
    print("  * whether the offloads pay off -- the PCIe-vs-kernel arithmetic")
    print("    behind their defaults is estimated, not benchmarked")
    print("\nVERIFIED from diffusers source (transformer_minimax_h3.py):")
    print("  * head_dim=128, 56 heads, num_layers=50, num_refiner_layers=2")
    print("  * the 301-long attention is the text token_refiner (:631)")
    print("  * no cross-attention anywhere, so the sink is the only path")
    print("    text conditioning has into the packed sequence")
    return 0


if __name__ == "__main__":
    sys.exit(main())

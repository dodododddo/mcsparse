"""Run MiniMax-H3 with token-sparse attention, on one GPU or several.

Three deployment shapes, chosen by --layout:

    auto      pick from the visible GPU count (the default)
    single    everything on one GPU, offloading to host RAM between stages
    cp        N ranks denoise under Ulysses context parallelism

and, orthogonal to that, --encoder-device decides whether the TEXT ENCODER gets
a GPU of its own. It does not move the VAE -- see the note below.

WHY THE ENCODER PLACEMENT IS A SEPARATE DECISION
------------------------------------------------
H3's checkpoint is 144 GB: text_encoder 66.7, transformer 66.3, video_vae 10.4.
The encoder runs once per sample and the VAE once at the end, while the
transformer runs 49 times. So the encoder's GPU sits idle for ~93% of a
request -- worth a card only if you are serving continuously and want to skip
the reload, wasteful if you are generating one clip.

    --encoder-device shared    the encoder shares the denoising GPU(s) and is
                               offloaded to host RAM after use. One fewer card.
                               Costs a reload per request (~20 s from page
                               cache, minutes from cold disk).

    --encoder-device cuda:N    a dedicated card, resident. This is the serving
                               layout: with 8 GPUs, 7 denoise and cuda:7 holds
                               the encoder. Conditioning for request k+1 can
                               overlap request k's denoise.

    --encoder-device auto      dedicate the last card when there are spare
                               GPUs (world+1 <= visible), else share.

THE VAE STAYS ON THE DENOISING CARD
-----------------------------------
Whatever --encoder-device says. The encoder is movable because it is popped out
of the workflow and called separately; the VAE is not -- `decode` is one of the
workflow's own blocks. Retargeting its offload hook loads the weights on the
other card but then crashes at the end of decode, because the decode step takes
a single `_execution_device` for both its own constants and the VAE's output.
The full reasoning is at the retarget site further down. Consequence: the VAE's
10.4 GB and its decode peak land on rank 0.

Also note what a dedicated encoder card looks like in nvidia-smi: near zero
until conditioning, then 66.7 GB for the rest of the run. CustomOffloadHook
onloads lazily -- at the first forward -- so the card is idle until the encoder
is actually called, and stays resident afterwards. That is the point of giving
it a card: the next request skips the reload.

On `shared` it is the opposite: this file pushes the encoder back to host RAM
once conditioning is done, because there it would otherwise sit on the card
that has to denoise. See the offload site below for why nothing else does it.

MEASURED, on 8xH20 at 345 frames (14.38 s), topk 0.15:
    dense      1137.5 s end to end, 1063.3 s in the transformer
    sparse      686.7 s              594.2 s        -> 1.79x on attention
    PSNR 27.30  SSIM 0.8823  LPIPS 0.1763 against the dense output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))

# Dense schedule. The first DENSE_STEPS steps and DENSE_LAYERS layers run dense
# in every arm, sparse or not: early steps set the composition and are the
# least tolerant of approximation, and layer 0 sees the rawest features. These
# are the values every number in the README was measured with -- change them
# and the comparison against dense is no longer the one that was validated.
DENSE_STEPS, DENSE_LAYERS = 10, "0"

# 17n+5 and at most 360. 345 = 14.38 s at 24 fps is the longest that fits;
# 360 itself is not of the form and 362 would exceed the cap.
DEFAULT_FRAMES = 345


def _packed_seq_len_guess(num_frames: int) -> int:
    """Estimate H3 packed sequence length for attention warm-up only.

    Scale with the requested frame count and round up to a query-tile boundary.
    The actual packed length depends on the prompt. This estimate warms
    allocator buckets and is not used to size the generation inputs.
    """
    L = int(104267 * max(num_frames, 1) / 345)
    return max(128, ((L + 127) // 128) * 128)


def frames_ok(n: int) -> bool:
    return (n - 5) % 17 == 0 and n <= 360


def valid_frames() -> list[int]:
    return [17 * k + 5 for k in range(1, 22) if 17 * k + 5 <= 360]


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--prompt", help="H3-Context-IR text, or @file to read one")
    ap.add_argument("--prompt-file", help="jsonl with an 'ir' field per line")
    ap.add_argument("--index", type=int, default=0,
                    help="which record of --prompt-file")
    ap.add_argument("--model", default=os.environ.get("H3_MODEL",
                                                      "MiniMaxAI/MiniMax-H3"))
    ap.add_argument("--out", default="out.mp4")
    ap.add_argument("--num-frames", type=int, default=DEFAULT_FRAMES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layout", default="auto",
                    choices=("auto", "single", "cp"))
    ap.add_argument("--encoder-device", default="auto",
                    help="auto | shared | cuda:N. Where the TEXT ENCODER "
                         "lives. It does not move the VAE, which stays on the "
                         "denoising card; see the module docstring.")
    ap.add_argument("--attention", default="token_sparse",
                    choices=("token_sparse", "dense"),
                    help="token_sparse is this package's method; dense is the "
                         "reference it is measured against")
    ap.add_argument("--topk", type=float, default=0.15,
                    help="fraction of tokens each query attends to. The sink "
                         "is ADDED to this rather than carved out of it, so "
                         "the realised density is topk + sink/L.")
    ap.add_argument("--sink-tokens", type=int, default=0,
                    help="length of the exact prefix (text + references + "
                         "audio). 0 disables the sink.")
    ap.add_argument("--num-steps", type=int, default=49)
    ap.add_argument("--calibrate-steps", default="9")
    # The dense guard. Defaults are the values every number in the README was
    # measured with; exposed because "calibrate on step 0" needs dense-steps 0,
    # and because on one GPU the dense warmup is the most expensive part.
    ap.add_argument("--dense-steps", type=int, default=DENSE_STEPS,
                    help=f"steps at the start of the trajectory that run dense "
                         f"in every arm (default {DENSE_STEPS}). 0 makes the "
                         f"run sparse from step 1, which needs "
                         f"--calibrate-steps 0.")
    ap.add_argument("--dense-layers", default=DENSE_LAYERS,
                    help=f"layers that always run dense, e.g. '0' or '0-1' "
                         f"(default {DENSE_LAYERS!r}). '' disables the guard.")
    # Memory. These were unreachable before: the CP wrapper dropped them on the
    # floor, so every run silently used offload-on / chunk-selector-off.
    ap.add_argument("--chunk-selector", action="store_true", default=False,
                    help="select tokens in chunks instead of materialising the "
                         "full (B,H,M,L_K) probability tensor -- 8.8 GB at 345 "
                         "frames. The single biggest saving on one GPU; "
                         "incompatible with adaptive_heads (off here anyway).")
    ap.add_argument("--chunk-selector-size", type=int, default=16384,
                    help="chunk size for --chunk-selector; must be a multiple "
                         "of 128. Smaller is less peak memory and more kernel "
                         "launches (4096 -> ~5.4 GB, 2048 -> ~4.7 GB).")
    ap.add_argument("--no-offload-indices", dest="offload_indices",
                    action="store_false", default=True,
                    help="keep the gather indices resident (~639 MB/layer, "
                         "~32 GB over 50 layers). Offloaded by default.")
    # THE knob for one GPU. ComponentsManager.enable_auto_cpu_offload works at
    # whole-model granularity, and the transformer has to be resident during its
    # own forward -- so on one card its 66.3 GB is a floor that no amount of
    # attention-level saving gets under. Group offload streams the 50 blocks in
    # and out instead, so only num_blocks_per_group of them are ever resident.
    ap.add_argument("--group-offload", type=int, default=0, metavar="N",
                    help="stream the transformer's 50 blocks through the GPU N "
                         "at a time instead of keeping all 66.3 GB resident. "
                         "REQUIRED to fit one GPU; 0 (default) disables it. "
                         "Start with 2. Lower N is less memory and more PCIe "
                         "traffic.")
    ap.add_argument("--group-offload-type", default="block_level",
                    choices=("block_level", "leaf_level"),
                    help="block_level streams whole transformer blocks; "
                         "leaf_level goes finer (less memory, more overhead). "
                         "leaf_level ignores --group-offload.")
    ap.add_argument("--no-group-offload-stream", dest="group_offload_stream",
                    action="store_false", default=True,
                    help="load each group on the default stream instead of a "
                         "side stream. This DISABLES prefetch (diffusers only "
                         "installs the lazy-prefetch hook when a stream is "
                         "given), so every group boundary becomes a blocking "
                         "H2D. Use it to measure what prefetch is worth.")
    ap.add_argument("--profile-blocks", action="store_true", default=False,
                    help="time every transformer block and report the "
                         "per-block distribution at the end. With group "
                         "offload on, a block that waited on its weights shows "
                         "up as a slow outlier; comparing the mean against a "
                         "--group-offload 0 run is the cost of offloading.")
    ap.add_argument("--cache-offload", default="residual",
                    choices=("none", "residual"),
                    help="offload the bias-correction residual to host RAM "
                         "(~545 MB/layer, ~27 GB over 50). Default residual.")
    ap.add_argument("--fa3", action="store_true", default=True,
                    help="FlashAttention-3 for the dense parts (the warmup "
                         "steps, layer 0, and the dense arm)")
    ap.add_argument("--no-fa3", dest="fa3", action="store_false")
    ap.add_argument("--warmup-attention", action="store_true", default=False,
                    help="Warm up attention before timing generation, moving first-call cuBLAS/"
                         "cuDNN setup and allocator growth outside the timing window. This changes "
                         "the meaning of `seconds`; leave it disabled for published-result comparisons. "
                         "The CuTeDSL sparse kernel is still first compiled during calibration.")
    ap.add_argument("--pca-proj-dim", type=int, default=None, metavar="K",
                    help="project Q onto its K highest-variance directions before "
                         "grouping (default: keep all D). This changes the "
                         "result -- low-variance directions are dropped -- so "
                         "measure with tests/bench_pca_group.py first.")
    ap.add_argument("--ulysses-anything", action="store_true", default=True)
    ap.add_argument("--no-ulysses-anything", dest="ulysses_anything",
                    action="store_false")
    ap.add_argument("--json-out", default=None,
                    help="write timings and kernel counters here")


def read_prompt(args) -> str:
    if args.prompt_file:
        recs = {}
        with open(args.prompt_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("ir"):
                    recs[r.get("index", len(recs))] = r["ir"]
        if args.index not in recs:
            raise SystemExit(f"--index {args.index} not in {args.prompt_file} "
                             f"(have {sorted(recs)})")
        return recs[args.index]
    if args.prompt:
        if args.prompt.startswith("@"):
            return open(args.prompt[1:], encoding="utf-8").read().strip()
        return args.prompt
    raise SystemExit("need --prompt or --prompt-file")


def resolve_layout(args, world: int, visible: int, log):
    """Return (layout, encoder_device). Pure decision, no side effects."""
    layout = args.layout
    if layout == "auto":
        layout = "cp" if world > 1 else "single"
    if layout == "cp" and world == 1:
        log("--layout cp with one rank behaves as single; launch with "
            "torchrun --nproc_per_node=N to get context parallelism")
        layout = "single"

    enc = args.encoder_device
    if enc == "auto":
        # A card of its own only if one is genuinely spare. On 8 GPUs with 7
        # denoising ranks that is cuda:7; on 8 ranks there is nothing left.
        enc = f"cuda:{world}" if world + 1 <= visible else "shared"
    return layout, enc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    args = ap.parse_args(argv)

    if not frames_ok(args.num_frames):
        near = [f for f in valid_frames() if abs(f - args.num_frames) <= 20]
        raise SystemExit(
            f"--num-frames {args.num_frames} must be 17n+5 and <= 360. "
            f"Nearest valid: {near} (max {valid_frames()[-1]} = "
            f"{valid_frames()[-1] / 24:.2f} s)")

    if args.group_offload < 0:
        raise SystemExit(
            f"--group-offload {args.group_offload} must be >= 0 "
            f"(0 disables it)")

    import torch

    distributed = "RANK" in os.environ
    if distributed:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(rank)
    else:
        rank, world = 0, 1

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    visible = torch.cuda.device_count()
    layout, enc_device = resolve_layout(args, world, visible, log)
    prompt = read_prompt(args)

    log(f"mcsparse: {args.attention}  layout={layout}  ranks={world}/{visible}")
    log(f"  frames {args.num_frames} ({args.num_frames / 24:.2f} s)  "
        f"seed {args.seed}")
    # "encoder", not "encoder/vae": the VAE is not affected by this setting.
    log(f"  text encoder: {enc_device}"
        + ("  (shares the denoising card, offloaded to host RAM after use)"
           if enc_device == "shared"
           else "  (own card; on CPU except during conditioning)"))
    log(f"  vae: cuda:{rank}  (stays on the denoising card -- "
        f"--encoder-device does not move it)")
    if args.attention == "token_sparse":
        guard = (f"steps 0-{args.dense_steps - 1}" if args.dense_steps
                 else "no dense warmup")
        log(f"  topk {args.topk}  sink {args.sink_tokens}  "
            f"dense guard: {guard}, layers {args.dense_layers or 'none'}")
        log(f"  memory: chunk_selector={args.chunk_selector}"
            + (f"(size {args.chunk_selector_size})" if args.chunk_selector
               else "")
            + f"  offload_indices={args.offload_indices}"
            f"  cache_offload={args.cache_offload}")

    # Verify FA3 before loading 144 GB, so a missing install is not discovered
    # mid-generation or, worse, silently read as an FA3 measurement.
    #
    # token_sparse REQUIRES FA3, and not just for the dense parts: the
    # calibrate step needs one full-attention pass that yields the output and
    # its LSE together, and FA3 is the only implementation here that returns
    # both (ops/full_attn.full_attention_with_lse). Computing the LSE
    # separately would cost an extra full QK^T per layer per calibrate step.
    if args.attention == "token_sparse" and not args.fa3:
        log("--no-fa3 cannot be combined with --attention token_sparse: the "
            "calibrate step needs full attention + LSE from a single pass, "
            "which only FA3 provides here.")
        log("use --attention dense --no-fa3, or install FlashAttention-3.")
        return 1
    if args.fa3:
        from mcsparse.h3.dense import fa3_available
        ok, why = fa3_available()
        if not ok:
            log(f"--fa3 requested but FlashAttention-3 is unavailable: {why}")
            if args.attention == "token_sparse":
                log("token_sparse requires FA3 (calibrate needs output+LSE in "
                    "one pass). Install it, or use --attention dense --no-fa3.")
            else:
                log("pass --no-fa3 to run the dense parts on SDPA instead")
            return 1
        log("  dense parts: FlashAttention-3")
    else:
        log("  dense parts: SDPA")

    from diffusers import ComponentsManager, ModularPipeline

    log("loading...")
    t_load = time.perf_counter()
    workflow = ModularPipeline.from_pretrained(args.model).blocks.get_workflow("t2va")

    # THE ENCODER IS BUILT ON RANK 0 ONLY. Conditioning must be identical on
    # every rank -- they denoise one shared latent -- so it is computed once
    # and broadcast. Building it everywhere would also cost 66.7 GB per rank.
    # Both stay None off rank 0; the offload after conditioning reads `cm`, so
    # bind it here rather than relying on the branch below having run.
    conditioner = None
    cm = None
    if rank == 0:
        cm = ComponentsManager()
        cm.enable_auto_cpu_offload(
            device=(f"cuda:{rank}" if enc_device == "shared" else enc_device))
        conditioner = workflow.sub_blocks.pop("text_encoder").init_pipeline(
            args.model, components_manager=cm)
        conditioner.load_components(dtype=torch.bfloat16)
    else:
        workflow.sub_blocks.pop("text_encoder")

    manager = ComponentsManager()
    # enable_auto_cpu_offload does two jobs: it keeps idle components in host
    # RAM, and it moves each forward's inputs to the execution device. The
    # second is why it stays on even when a card could hold everything --
    # without it the conditioner's CPU-built index tensors never reach the GPU.
    manager.enable_auto_cpu_offload(device=f"cuda:{rank}")
    rest = workflow.init_pipeline(args.model, components_manager=manager)
    rest.load_components(dtype=torch.bfloat16)
    log(f"loaded in {time.perf_counter() - t_load:.0f}s")

    # WHY THE VAE STAYS ON THE DENOISING CARD, even with --encoder-device cuda:N.
    #
    # The flag moves the text encoder only. That is a real limitation and it is
    # not one this file can fix, so it is written down here rather than left to
    # be discovered from nvidia-smi.
    #
    # The encoder can move because it is popped out of the workflow into its own
    # manager (cm, above) and called separately. The VAEs cannot: `decode` is one
    # of the workflow's own blocks and is what produces videos/audio, so popping
    # it would break `rest(...)`.
    #
    # Retargeting the hook instead (manager.model_hooks[i].hook.execution_device)
    # loads the weights on the right card but then crashes in decode. The step
    # takes ONE device for everything:
    #
    #     device = components._execution_device            # decoders.py:175
    #     latents_mean = torch.tensor(..., device=device)  # :183
    #     video = components.vae.decode(latents)           # :188, runs on the VAE's card
    #     video = video.float() * pixel_std + pixel_mean   # :191, pixel_std on `device`
    #
    # `_execution_device` walks `components` and returns the first hooked
    # module's execution device (modular_pipeline.py:2257-2270) -- the
    # transformer's, not the VAE's -- and CustomOffloadHook has no post_forward
    # to bring the output back. So the last line multiplies a cuda:N tensor by a
    # cuda:rank one: RuntimeError at the very end of a 10-minute run.
    #
    # Doing this properly means teaching the decode blocks about a second
    # device, i.e. a diffusers change, not a flag here. The cost of leaving it:
    # the VAE's 10.4 GB and its decode peak land on rank 0. It is offloaded
    # between stages like everything else in this manager, so it only shows up
    # during decode itself.

    transformer = None
    for name in ("transformer", "dit", "denoiser", "unet"):
        transformer = getattr(rest, name, None)
        if transformer is not None:
            break
    if transformer is None:
        log("no transformer on the pipeline")
        return 1

    # Group offload, before enable_parallelism and before the dispatch is
    # patched. Applied to the transformer only: the encoder and VAE run once
    # each and enable_auto_cpu_offload already keeps them out of the way, while
    # the transformer is the one model that is resident during its own forward.
    if args.group_offload or args.group_offload_type == "leaf_level":
        n_blocks = len(getattr(transformer, "transformer_blocks", []) or [])
        transformer.enable_group_offload(
            onload_device=torch.device(f"cuda:{rank}"),
            offload_device=torch.device("cpu"),
            offload_type=args.group_offload_type,
            # Ignored for leaf_level, which splits at the leaf-module level.
            num_blocks_per_group=(args.group_offload
                                  if args.group_offload_type == "block_level"
                                  else None),
            # THE STREAM IS WHAT BUYS PREFETCH, not just overlap in general.
            # diffusers installs LazyPrefetchGroupOffloadingHook only when
            # config.stream is not None (group_offloading.py, the
            # _apply_group_offloading_block_level tail). Without it every group
            # is created with onload_self=True and loads itself synchronously at
            # its own pre_forward. With it, the first forward traces the
            # execution order, then flips next_group.onload_self=False so group
            # N's pre_forward kicks off group N+1's H2D on the side stream and
            # only syncs when N+1's turn arrives.
            #
            # Two consequences worth knowing when reading a profile: the FIRST
            # forward is synchronous (the order is not known yet), and
            # low_cpu_mem_usage=False (the default) pins the CPU copies, which
            # is what makes the async copy actually async.
            use_stream=args.group_offload_stream,
            record_stream=args.group_offload_stream,
        )
        if args.group_offload_type == "block_level":
            log(f"  group offload: {args.group_offload} of {n_blocks} blocks "
                f"resident (~{66.3 * args.group_offload / max(n_blocks, 1):.1f} "
                f"GB of weights instead of 66.3)")
        else:
            log(f"  group offload: leaf_level over {n_blocks} blocks")

    if layout == "cp" and world > 1:
        from diffusers.models._modeling_parallel import ContextParallelConfig
        transformer.enable_parallelism(config=ContextParallelConfig(
            ulysses_degree=world, ring_degree=1,
            ulysses_anything=args.ulysses_anything))
        log(f"ulysses context parallelism over {world} ranks "
            f"({56 / world:.1f} heads each)")

    from diffusers.models.transformers import transformer_minimax_h3 as h3
    from mcsparse.h3 import token_sparse_dispatch as tsd
    from mcsparse.h3.context_parallel import (get_cp_stats, reset_cp_stats,
                                              with_ulysses_sparse)

    dense_dispatch = h3.dispatch_attention_fn
    if args.attention == "token_sparse":
        cal = [int(x) for x in str(args.calibrate_steps).split(",") if x != ""]
        h3.dispatch_attention_fn = with_ulysses_sparse(
            dense_dispatch, kind="mcsa_token", fa3=args.fa3,
            topk=args.topk, num_steps=args.num_steps, calibrate_steps=cal,
            sink_tokens=args.sink_tokens,
            dense_steps=args.dense_steps, dense_layers=args.dense_layers,
            pca_reorder=True, bias_correction=True,
            pca_proj_dim=args.pca_proj_dim,
            # Ranks hold H/world heads each, so a head ranking computed locally
            # is not the global one. Off under CP by construction.
            adaptive_heads=False,
            use_chunk_selector=args.chunk_selector,
            chunk_selector_size=args.chunk_selector_size,
            offload_indices=args.offload_indices,
            cache_offload=args.cache_offload,
            full_attn_impl=("fa3" if args.fa3 else "sdpa"),
            allow_no_sink=(args.sink_tokens == 0))
    elif args.fa3:
        from mcsparse.h3.dense import make_fa3_dense_dispatch
        h3.dispatch_attention_fn = make_fa3_dense_dispatch(dense_dispatch)

    # Time the transformer separately from the VAE. The VAE decode is identical
    # work in every arm and is ~7% of dense's wall clock, so including it pulls
    # every speedup toward 1.0x; reporting both keeps either claim honest.
    dit = {"total": 0.0, "t0": None, "calls": 0}

    def dit_start(*_):
        torch.cuda.synchronize()
        dit["t0"] = time.perf_counter()

    def dit_end(*_):
        if dit["t0"] is not None:
            torch.cuda.synchronize()
            dit["total"] += time.perf_counter() - dit["t0"]
            dit["calls"] += 1
            dit["t0"] = None

    def tick(*_):
        dit_start()
        if args.attention == "token_sparse":
            tsd.mcsa_token_begin_forward()

    transformer.register_forward_pre_hook(tick)
    transformer.register_forward_hook(dit_end)

    # Per-block timing, for costing group offload. CUDA events rather than
    # perf_counter: the H2D runs on a side stream and the block's kernels are
    # async, so CPU-side timing would measure launch, not execution.
    #
    # What to read: with prefetch working, blocks are uniform and the offload is
    # hidden. A sawtooth -- one slow block per group of num_blocks_per_group --
    # means the copy is NOT being hidden and you are bandwidth-bound. Compare
    # `block_ms mean` against a --group-offload 0 run for the real cost.
    blocks = list(getattr(transformer, "transformer_blocks", []) or [])
    block_times: list[list[float]] = [[] for _ in blocks]
    if args.profile_blocks and blocks:
        ev = [(torch.cuda.Event(enable_timing=True),
               torch.cuda.Event(enable_timing=True)) for _ in blocks]
        armed = [False] * len(blocks)

        def _mk_pre(i):
            def pre(*_):
                # Harvest the PREVIOUS step's pair before overwriting it. By now
                # the GPU is a whole denoising step past it, so query() is true
                # and no synchronise is needed -- which matters, because
                # synchronising here would serialise the overlap we are
                # measuring.
                if armed[i] and ev[i][1].query():
                    block_times[i].append(ev[i][0].elapsed_time(ev[i][1]))
                ev[i][0].record()
            return pre

        def _mk_post(i):
            def post(*_):
                ev[i][1].record()
                armed[i] = True
            return post

        for i, b in enumerate(blocks):
            b.register_forward_pre_hook(_mk_pre(i))
            b.register_forward_hook(_mk_post(i))
        log(f"  profiling {len(blocks)} blocks with CUDA events")

    log("conditioning...")
    t_enc = time.perf_counter()
    state = None
    if rank == 0:
        state = conditioner(prompt=prompt)
    enc_seconds = time.perf_counter() - t_enc

    # THE ENCODER IS DONE. On `shared` it has to go back to host RAM.
    #
    # Only on `shared`. With a dedicated --encoder-device cuda:N the encoder is
    # on a card that never denoises, so evicting it buys nothing and costs a
    # reload on the next request -- and it is a divergence from the reference
    # implementation, which does not offload at all. Matching it in the layout
    # the published numbers were measured in is worth more than the tidiness.
    #
    # On `shared` it is not optional. Nothing else does it, which is easy to
    # miss because enable_auto_cpu_offload sounds like it would:
    #
    #  * The encoder has its OWN ComponentsManager (cm above), separate from the
    #    denoising one. A CustomOffloadHook only ever evicts models in its own
    #    manager's `self.components` -- `other_hooks` is built from that one list
    #    (components_manager.py, enable_auto_cpu_offload). So the transformer's
    #    hook does not know the encoder exists and cannot displace it.
    #  * Even in one manager it would not fire. AutoOffloadStrategy returns []
    #    -- evict nobody -- as soon as the incoming model's *weights* fit in free
    #    memory. It never accounts for activations, which at 345 frames are the
    #    ~20 GB that actually decide whether the run survives. Group offload
    #    makes that worse: get_memory_footprint() still reports the full 35.9 GB
    #    for a model that only ever holds a couple of blocks.
    #
    # So on `shared` the encoder's 66.7 GB used to sit resident on the denoising
    # card for the whole run, doing nothing -- which is why trimming activations
    # (fewer frames, chunked selector, the PCA fix) only ever moved the OOM later
    # instead of preventing it.
    if rank == 0 and conditioner is not None and enc_device == "shared":
        enc_dev = torch.device(f"cuda:{rank}")
        before = torch.cuda.memory_allocated(enc_dev) / 2**30
        for h in getattr(cm, "model_hooks", None) or []:
            h.offload()          # CustomOffloadHook.init_hook -> model.to("cpu")
        import gc
        gc.collect()
        # device= is load-bearing even here, where it happens to equal the
        # current device: torch.cuda.empty_cache() acts on the *current* device,
        # so writing it bare made this line's meaning depend on whatever device
        # context happened to be active. Being explicit is what keeps it correct
        # if this block ever moves.
        with torch.cuda.device(enc_dev):
            torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated(enc_dev) / 2**30
        log(f"  encoder offloaded to host RAM: {enc_dev} "
            f"{before:.1f} -> {after:.1f} GB allocated")

    if distributed and world > 1:
        obj = [state]
        dist.broadcast_object_list(obj, src=0)
        state = obj[0]
        # Move only the CUDA tensors. build_packed_sequence assembles its
        # index and tag tensors on the CPU deliberately, and moving those to
        # the GPU makes token_tags[text_indices] = text_token_tags mix devices.
        dev = torch.device(f"cuda:{rank}")
        vals = getattr(state, "values", None)
        if isinstance(vals, dict):
            for key, val in list(vals.items()):
                if torch.is_tensor(val) and val.is_cuda and val.device != dev:
                    vals[key] = val.to(dev)
                elif isinstance(val, (list, tuple)) and val and all(
                        torch.is_tensor(x) for x in val):
                    vals[key] = type(val)(
                        x.to(dev) if x.is_cuda and x.device != dev else x
                        for x in val)

    # WARM THE ATTENTION PATH BEFORE THE CLOCK STARTS.
    #
    # Measured at 345 frames / 7 ranks (tqdm elapsed differenced per step): the
    # first denoising step costs 69 s against 21 s for steps 2-9, and step 10 --
    # the calibrate step, where the sparse kernel and the Triton token_probs
    # pass first run -- costs 69 s against 8 s steady state. Steady state itself
    # is identical to the reference (8.0 s/step over steps 12-49), so the gap is
    # entirely first-call cost.
    #
    # What is actually being paid here is NOT FA3 compilation -- FA3 ships a
    # prebuilt .so (flash_attn_3/_C.abi3.so) and does no JIT. The first-call
    # costs on this path are:
    #   * the CuteDSL gather-KV kernel, which nvidia-cutlass-dsl JIT-compiles at
    #     first call, for each distinct shape it sees;
    #   * the Triton token_probs kernel, same;
    #   * cuBLAS/cuDNN algorithm selection and the allocator growing its pools.
    # The warmup below only covers the last two cheaply; the CuteDSL kernel
    # needs the real selection indices, so it cannot be warmed without doing a
    # real calibrate step. Hence this is an instrument, not a fix: it moves what
    # it can out of the measured window and reports what it moved.
    #
    # Off by default -- it changes what `seconds` means, so a run meant to be
    # compared against published numbers should not have it on.
    if args.warmup_attention and args.attention != "dense":
        t_warm = time.perf_counter()
        # THE REAL SHAPE, not a token one. FA3 is prebuilt and does not compile
        # per shape, so what a warmup can actually move out of the measured
        # window is cuBLAS/cuDNN algorithm selection, CUDA context lazy init,
        # and the allocator's first request to the driver for blocks of this
        # size. The allocator part is size-bucketed: a 4096-token warmup does
        # not prepare the pool for a 104k-token step, which is most of what
        # step 1 pays for. Under CP each rank holds all tokens for H/world
        # heads, which is the shape the kernel sees after the all-to-all.
        L_w = _packed_seq_len_guess(args.num_frames)
        H_w, D_w = 56 // max(world, 1), 128
        try:
            qw = torch.randn(1, L_w, H_w, D_w, device=f"cuda:{rank}",
                             dtype=torch.bfloat16)
            if args.fa3:
                from mcsparse.h3.dense import fa3_dense
                # Both call shapes the run will use: plain dense, and the
                # output+LSE form the calibrate step needs.
                fa3_dense(qw, qw, qw)
                from mcsparse.kernels.flash_mcsa.ops.full_attn import (
                    full_attention_with_lse)
                full_attention_with_lse(qw.transpose(1, 2).contiguous(),
                                        qw.transpose(1, 2).contiguous(),
                                        qw.transpose(1, 2).contiguous())
            del qw
            torch.cuda.synchronize()
            log(f"  warmed the attention path in "
                f"{time.perf_counter() - t_warm:.1f}s  "
                f"(L={L_w} x {H_w} heads: FA3 dense + output/LSE. The CuteDSL "
                f"sparse kernel is NOT warmed -- it needs real selection "
                f"indices, so it still compiles at the calibrate step.)")
        except Exception as exc:
            # A warmup failure must not take the run down -- it is measurement
            # scaffolding, and the real call sites would report the same error.
            log(f"  warmup skipped: {type(exc).__name__}: {exc}")
        torch.cuda.empty_cache()

    if args.attention == "token_sparse":
        tsd.reset_mcsa_token_state()
    reset_cp_stats()
    # Free on EVERY rank, not just the one that ran the encoder. Conditioning
    # leaves fragmentation behind on rank 0 (it held the encoder) and the
    # broadcast leaves some on the others, and the first denoising step is the
    # largest allocation of the run -- going into it with a fragmented pool is
    # what makes step 1 cost several times step 2.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # START THE CLOCK TOGETHER.
    #
    # Only rank 0 runs the conditioner and the encoder offload, so without this
    # the other ranks reach the generate call first and sit inside the first
    # all-to-all waiting for rank 0. Each rank then times from its own start,
    # and `dit_seconds` picks up however far apart they were -- charged to the
    # transformer, where it looks like the kernel got slower.
    if distributed and world > 1:
        dist.barrier()

    log("generating...")
    t0 = time.perf_counter()
    res = rest(state=state, num_frames=args.num_frames,
               generator=torch.Generator().manual_seed(args.seed),
               output=["videos", "audio", "sampling_rate"])
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    cps = get_cp_stats()

    log(f"\n{seconds:.1f}s total   {dit['total']:.1f}s transformer "
        f"({dit['calls']} forwards)   {seconds - dit['total']:.1f}s vae+rest")
    # cp_calls counts attention calls that went through the Ulysses all-to-all,
    # local_calls those that did not -- so under --layout single every call is
    # local and cp_calls is 0. That is the expected reading on one GPU, not a
    # sign the sparse path was skipped; kernel_calls below is what says that.
    log(f"peak {peak:.1f} GB   cp_calls={cps['cp_calls']} "
        f"local_calls={cps['local_calls']}")

    block_stats = None
    if args.profile_blocks and any(block_times):
        means = [sum(t) / len(t) for t in block_times if t]
        tot = sum(means)
        srt = sorted(means)
        n = len(srt)
        p50 = srt[n // 2]
        p95 = srt[min(n - 1, int(n * 0.95))]
        block_stats = {
            "blocks": n, "samples_per_block": len(next(t for t in block_times if t)),
            "mean_ms": tot / n, "min_ms": srt[0], "p50_ms": p50, "p95_ms": p95,
            "max_ms": srt[-1], "sum_ms": tot,
            "per_block_mean_ms": means,
        }
        log(f"\nper-block (CUDA events, mean over steps):")
        log(f"  {n} blocks   mean {tot / n:.1f} ms   p50 {p50:.1f}   "
            f"p95 {p95:.1f}   max {srt[-1]:.1f}   sum {tot / 1000:.2f} s/step")
        # A group boundary that failed to hide its H2D shows up here: with
        # num_blocks_per_group=N, every Nth block is the one that waits.
        if args.group_offload and args.group_offload > 0:
            g = args.group_offload
            lead = [m for i, m in enumerate(means) if i % g == 0]
            rest = [m for i, m in enumerate(means) if i % g != 0]
            if lead and rest:
                log(f"  group leaders (every {g}th block) mean "
                    f"{sum(lead) / len(lead):.1f} ms vs others "
                    f"{sum(rest) / len(rest):.1f} ms")
                log(f"  -> a large gap means the prefetch is NOT hiding the "
                    f"H2D; compare against --group-offload 0")

    backend = None
    if args.attention == "token_sparse":
        backend = tsd.get_mcsa_token_stats()
        kc = backend.get("kernel_calls", 0)
        log(f"kernel_calls={kc} fallback={backend.get('fallback_calls')} "
            f"dense_guard={backend.get('dense_guard_calls')} "
            f"seq_len={backend.get('seq_len')} "
            f"density nominal={backend.get('nominal_density')} "
            f"effective={backend.get('effective_density')}")
        if not kc:
            log("WARNING: the sparse kernel never ran -- every call fell back "
                "to dense, so this is a dense measurement wearing a sparse "
                f"label. Lengths seen: {backend.get('seen_seq_lens')}")

    if rank == 0:
        from diffusers.utils.export_utils import encode_video
        get = res.get if hasattr(res, "get") else (lambda k: getattr(res, k))
        # Unlink first: the output name may be a hard link to an earlier run's
        # video, and writing through it would rewrite that one too.
        if os.path.lexists(args.out):
            os.unlink(args.out)
        encode_video(get("videos")[0], fps=24, output_path=args.out,
                     audio=get("audio")[0],
                     audio_sample_rate=get("sampling_rate"))
        log(f"wrote {args.out}")

        if args.json_out:
            json.dump({
                "attention": args.attention, "layout": layout,
                "world": world, "encoder_device": enc_device,
                "frames": args.num_frames, "seconds_video": args.num_frames / 24,
                "num_steps": args.num_steps,
                "seed": args.seed, "fa3": args.fa3,
                "topk": args.topk if args.attention == "token_sparse" else None,
                "sink_tokens": args.sink_tokens,
                "seconds": seconds, "dit_seconds": dit["total"],
                "vae_seconds": seconds - dit["total"],
                "encoder_seconds": enc_seconds,
                "dit_calls": dit["calls"], "peak_gb": peak,
                "cp": cps, "backend": backend,
                # Persist every option that can change numerical results or the selection path.
                "chunk_selector": args.chunk_selector,
                "chunk_selector_size": args.chunk_selector_size,
                "offload_indices": args.offload_indices,
                "cache_offload": args.cache_offload,
                "pca_proj_dim": args.pca_proj_dim,
                "dense_steps": args.dense_steps,
                "dense_layers": args.dense_layers,
                "calibrate_steps": args.calibrate_steps,
                "warmup_attention": args.warmup_attention,
                "group_offload": args.group_offload,
                "group_offload_type": args.group_offload_type,
                "group_offload_stream": args.group_offload_stream,
                "blocks": block_stats,
            }, open(args.json_out, "w", encoding="utf-8"), indent=2, default=str)
            log(f"wrote {args.json_out}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

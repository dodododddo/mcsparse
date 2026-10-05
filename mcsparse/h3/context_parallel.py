"""Sparse attention that survives Ulysses context parallelism.

THE PROBLEM. Sparse backends are installed by replacing
h3.dispatch_attention_fn, which sits at the OUTERMOST layer:

    dispatch_attention_fn                  <- where we patch
      backend_fn(..., _parallel_config)
        _parallel_config is None -> compute directly
        else -> _templated_context_parallel_attention
                  all-to-all      (scatter heads, gather sequence)
                  forward_op      <- attention actually happens here
                  all-to-all      (undo)

With CP enabled a wrapper at the outer layer receives SEQUENCE-SHARDED,
full-head tensors of shape (B, S_local, H, D). Block-sparse routing there is
wrong: each rank sees 1/N of the tokens, so no global top-k or threshold
decision is possible. That is why every sparse dispatch bails out when
`parallel_config is not None`.

WHY ULYSSES IS NONETHELESS COMPATIBLE. Inside
_templated_context_parallel_attention the all-to-all runs BEFORE forward_op
(attention_dispatch.py:2541), turning those into full-sequence, head-local
tensors. Each rank then owns every token for a disjoint set of heads -- exactly
the position the released H100 runtime takes, which its adapter describes as
"runs after the Ulysses all-to-all, where each rank owns all packed tokens for a
disjoint set of heads". Routing there is identical to single-GPU routing; only
the head count differs.

Ring stays unsupported, and not by choice: its KV arrives in rotating chunks, so
no point in the computation ever holds a whole KV sequence.

HOW THIS HOOKS IN WITHOUT TOUCHING ANYTHING. dispatch_attention_fn is handed
`parallel_config` by the H3 processor, and _templated_context_parallel_attention
is importable and takes `forward_op` as a keyword. So a wrapper can do the CP
call itself, passing the sparse kernel as forward_op -- the communication is
reused verbatim from diffusers and the kernel lands after the all-to-all.

Nothing is mutated: no registry entry is overwritten, no diffusers module is
patched, no existing file changes behaviour. Import this and wrap; ignore it and
everything behaves exactly as before.

    from mcsparse.h3.context_parallel import with_ulysses_sparse
    d = with_ulysses_sparse(dense_dispatch, kind="mcsa_token", topk=0.15,
                            sink_tokens=0, dense_steps=10,
                            dense_layers="0", fa3=True)
    h3.dispatch_attention_fn = d

Head divisibility is the caller's: ulysses_degree must divide the head count
(H3 has 56, so 2/4/7/8 work) unless ulysses_anything is set.
"""

from __future__ import annotations

import math

# cp_calls: went through the Ulysses all-to-all. local_calls: did not (one GPU,
# or a call diffusers handed us without a parallel_config).
#
# There is deliberately no "dense_guard" here. This layer cannot see the dense
# guard -- full_steps/full_layers are decided inside the dispatch, which owns
# `dense_guard_calls` in its own counters. A key of that name used to sit in
# this dict, was never incremented by anyone, and pipeline.py printed it: the
# run always reported dense_guard=0 even though layer 0 is dense on every step.
_STATS = {"cp_calls": 0, "local_calls": 0}


def get_cp_stats() -> dict:
    return dict(_STATS)


def reset_cp_stats() -> None:
    for k in _STATS:
        _STATS[k] = 0


def _make_dense(fa3: bool):
    def dense(q, k, v):
        if fa3:
            from mcsparse.h3.dense import fa3_dense

            return fa3_dense(q, k, v)
        import torch

        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)

    return dense


def _make_kernel(kind, cfg, dense):
    """The sparse call itself, on full-sequence head-local BTHD tensors.

    "mcsa_token" is the method; "dense" is the reference it is measured
    against, routed through the same CP machinery so the comparison differs
    only in the kernel.
    """
    if kind == "mcsa_token":
        from mcsparse.h3 import token_sparse_dispatch as mcsa_token_h3

        # Head adaptation is disabled under CP: each rank sees only local heads.
        # Layer-depth adaptation is independent of the head partition.
        inner = mcsa_token_h3.make_mcsa_token_dispatch(
            lambda q, k, v, *a, **kw: dense(q, k, v),
            sink_tokens=int(cfg.get("sink_tokens", 0)),
            num_steps=int(cfg.get("num_steps", 49)),
            topk=float(cfg.get("topk", 0.25)),
            dense_steps=int(cfg.get("dense_steps", 0)),
            dense_layers=str(cfg.get("dense_layers", "")),
            calibrate_steps=cfg.get("calibrate_steps"),
            pca_reorder=bool(cfg.get("pca_reorder", True)),
            pca_proj_dim=cfg.get("pca_proj_dim"),
            bias_correction=bool(cfg.get("bias_correction", True)),
            adaptive_layers=bool(cfg.get("adaptive_layers", False)),
            full_attn_impl=str(cfg.get("full_attn_impl", "fa3")),
            allow_no_sink=bool(cfg.get("allow_no_sink", True)),
            # MEMORY KNOBS MUST BE FORWARDED EXPLICITLY. They were missing here,
            # so a caller passing offload_indices=False or
            # use_chunk_selector=True saw no effect: the kwargs landed in **cfg,
            # were never read, and make_mcsa_token_dispatch applied its own
            # defaults instead. That silently pinned every run to
            # offload_indices=True / cache_offload='residual' /
            # use_chunk_selector=False, and left the chunked selector --
            # the largest single saving at H3's shape -- unreachable.
            offload_indices=bool(cfg.get("offload_indices", True)),
            cache_offload=str(cfg.get("cache_offload", "residual")),
            offload_prefetch_ahead=int(cfg.get("offload_prefetch_ahead", 1)),
            use_chunk_selector=bool(cfg.get("use_chunk_selector", False)),
            chunk_selector_size=int(cfg.get("chunk_selector_size", 16384)),
            num_layers=cfg.get("num_layers"),
        )

        def kernel(q, k, v):
            # parallel_config=None on purpose: we are already past the
            # all-to-all, so from this kernel's point of view the tensors are an
            # ordinary full-sequence attention problem. Its own
            # "parallel_config is not None" guard (mcsa_token_h3.py:645) is for
            # the outer, sequence-sharded position and must not fire here.
            return inner(q, k, v, None, 0.0, False, None, False, None,
                         backend=None, parallel_config=None)

        return kernel

    if kind == "dense":
        def kernel(q, k, v):
            return dense(q, k, v)

        return kernel

    raise ValueError(
        f"kind must be 'mcsa_token' or 'dense'; got {kind!r}")


def with_ulysses_sparse(original_dispatch, *, kind, fa3=False, **cfg):
    """Wrap the H3 dispatch so the sparse kernel runs after the all-to-all.

    Without CP this behaves like the ordinary sparse dispatch. With Ulysses CP it
    performs the CP call itself, handing the kernel to diffusers as forward_op.
    With ring CP it raises, because the kernel cannot see a whole KV sequence.
    """
    dense = _make_dense(fa3)
    kernel = _make_kernel(kind, cfg, dense)
    default_scale = 1.0 / math.sqrt(128)

    def forward_op(ctx, query, key, value, attn_mask=None, dropout_p=0.0,
                   is_causal=False, scale=None, enable_gqa=False,
                   return_lse=False, _save_ctx=True, _parallel_config=None):
        # Signature fixed by _templated_context_parallel_attention, which calls
        # this positionally; matched against _native_attention_forward_op
        # (attention_dispatch.py:825). Tensors arrive BTHD, as our kernels want.
        if return_lse:
            raise ValueError(
                "sparse CP forward_op produces no LSE, so it cannot serve ring "
                "attention. Use ulysses_degree with ring_degree=1.")
        _STATS["cp_calls"] += 1
        return kernel(query.contiguous(), key.contiguous(), value.contiguous())

    def backward_op(*a, **kw):
        # Inference only; _templated_context_parallel_attention merely stores
        # this on the autograd Function and never calls it under no_grad.
        raise RuntimeError("sparse CP backend is forward-only")

    def dispatch(query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, scale=None, enable_gqa=False,
                 attention_kwargs=None, *, backend=None, parallel_config=None):
        supported = (
            attn_mask is None
            and not is_causal
            and dropout_p == 0.0
            and not enable_gqa
            and (scale is None or math.isclose(float(scale), default_scale))
            and query.ndim == 4
            and query.shape == key.shape == value.shape
            and query.is_cuda
        )
        if not supported:
            return original_dispatch(query, key, value, attn_mask, dropout_p,
                                     is_causal, scale, enable_gqa,
                                     attention_kwargs, backend=backend,
                                     parallel_config=parallel_config)

        if parallel_config is None:
            _STATS["local_calls"] += 1
            return kernel(query.contiguous(), key.contiguous(),
                          value.contiguous())

        cp = getattr(parallel_config, "context_parallel_config", None)
        if cp is None:
            _STATS["local_calls"] += 1
            return kernel(query.contiguous(), key.contiguous(),
                          value.contiguous())
        if getattr(cp, "ring_degree", 1) > 1:
            raise ValueError(
                f"ring_degree={cp.ring_degree}: ring attention hands each rank "
                f"rotating KV chunks, so a block-sparse kernel never sees a "
                f"whole KV sequence and its routing would be wrong. Use "
                f"ulysses_degree instead (H3 has 56 heads, so 2/4/7/8 divide "
                f"evenly).")

        from diffusers.models.attention_dispatch import (
            _templated_context_parallel_attention)

        # Reuse diffusers' own communication; our kernel is the forward_op, so it
        # executes after the all-to-all on full-sequence head-local tensors.
        return _templated_context_parallel_attention(
            query, key, value, None, dropout_p, is_causal, scale, False, False,
            forward_op=forward_op, backward_op=backward_op,
            _parallel_config=parallel_config,
        )

    return dispatch


__all__ = ["get_cp_stats", "reset_cp_stats", "with_ulysses_sparse"]

"""FlashAttention-3 for the dense parts of a sparse run.

Every sparse run contains dense work: the first ``dense_steps`` denoising
steps, ``dense_layers`` (layer 0 by default), and the calibrate step's full
attention. Left alone those go through diffusers' dispatch, which defaults to
torch SDPA -- so "dense" would mean SDPA, and the reported speedup would be
measured partly against a kernel nobody would deploy.

Routing them through FA3 makes the comparison honest in both directions: the
dense baseline is one worth beating, and the sparse arm's own dense fraction
is not quietly inflating its number.

FA3 is optional. ``fa3_available()`` is checked before the 144 GB load rather
than at the first attention call, so a missing install surfaces immediately
instead of mid-generation -- or, worse, silently reading as an FA3 result.
``--no-fa3`` uses SDPA throughout.
"""

from __future__ import annotations

_FA3_STATE = {"checked": False, "ok": False, "why": ""}


def fa3_available() -> tuple[bool, str]:
    """Whether FlashAttention-3 can actually run here.

    diffusers gates its _flash_3 backend on is_flash_attn_3_available(), which
    needs the `flash_attn_interface` module (attention_dispatch.py:101). Checked
    once and cached; the reason is reported so a silent fall back to SDPA cannot
    be mistaken for an FA3 measurement.
    """
    if _FA3_STATE["checked"]:
        return _FA3_STATE["ok"], _FA3_STATE["why"]
    _FA3_STATE["checked"] = True
    try:
        import flash_attn_interface  # noqa: F401
    except ImportError as exc:
        _FA3_STATE["why"] = (
            f"flash_attn_interface not importable ({exc}). Install "
            "FlashAttention-3 (hopper build) to use FA3 dense."
        )
        return False, _FA3_STATE["why"]
    try:
        import torch
        cap = torch.cuda.get_device_capability(0)
    except Exception as exc:  # no CUDA yet
        _FA3_STATE["why"] = f"cannot read device capability: {exc}"
        return False, _FA3_STATE["why"]
    if cap[0] < 9:
        _FA3_STATE["why"] = f"FA3 needs Hopper or newer, got SM{cap[0]}{cap[1]}"
        return False, _FA3_STATE["why"]
    _FA3_STATE["ok"] = True
    _FA3_STATE["why"] = "ok"
    return True, "ok"


def fa3_dense(q, k, v):
    """Dense attention via FA3, on BTHD tensors.

    FA3's flash_attn_func wants (batch, seqlen, heads, dim) -- which is BTHD
    already, so unlike the SDPA path this needs no transpose at all.
    """
    from flash_attn_interface import flash_attn_func

    out = flash_attn_func(q, k, v)
    # Some builds return (out, softmax_lse); take the output.
    if isinstance(out, tuple):
        out = out[0]
    return out


def make_fa3_dense_dispatch(original_dispatch):
    """Route the plain dense arm through FA3 instead of SDPA."""
    import math

    default_scale = 1.0 / math.sqrt(128)

    def dispatch(query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, scale=None, enable_gqa=False,
                 attention_kwargs=None, *, backend=None, parallel_config=None):
        eligible = (
            parallel_config is None
            and attn_mask is None
            and not is_causal
            and dropout_p == 0.0
            and not enable_gqa
            and (scale is None or math.isclose(float(scale), default_scale))
            and query.ndim == 4
            and query.shape == key.shape == value.shape
            and query.is_cuda
            and query.dtype in (__import__("torch").bfloat16,
                                __import__("torch").float16)
        )
        if eligible:
            return fa3_dense(query.contiguous(), key.contiguous(),
                             value.contiguous()).type_as(query)
        return original_dispatch(query, key, value, attn_mask, dropout_p,
                                 is_causal, scale, enable_gqa,
                                 attention_kwargs, backend=backend,
                                 parallel_config=parallel_config)

    return dispatch


__all__ = ["fa3_available", "fa3_dense", "make_fa3_dense_dispatch"]

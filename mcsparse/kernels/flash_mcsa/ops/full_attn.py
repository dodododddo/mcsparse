"""Select the full-attention implementation for dense steps and layers.

full_attn_impl accepts 'sdpa' (the default), 'fa3' (required when explicitly
selected), or 'auto' (FA3 when available, otherwise SDPA). These paths evaluate
full softmax attention. Calibration separately requires FA3 output and LSE.
"""

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

VALID_IMPLS = ('sdpa', 'fa3', 'auto')

# Cache (function, kind), or (None, None), once per process.
_FA3: Optional[Tuple[Optional[Callable], Optional[str]]] = None


def _probe_fa3() -> Tuple[Optional[Callable], Optional[str]]:
    """Probe FA3 once per process and cache the function reference.

    Import probing happens before the attention hot path.
    """
    global _FA3
    if _FA3 is not None:
        return _FA3
    try:
        from flash_attn_interface import flash_attn_func
        _FA3 = (flash_attn_func, "flash_attn_interface")
    except ImportError:
        _FA3 = (None, None)
    return _FA3


def resolve_impl(impl: str) -> str:
    """Resolve the configured backend to 'sdpa' or 'fa3' during initialization."""
    impl = (impl or 'sdpa').lower()
    if impl not in VALID_IMPLS:
        raise ValueError(
            f"full_attn_impl must be one of {'/'.join(VALID_IMPLS)}, got {impl!r}"
        )
    if impl == 'sdpa':
        return 'sdpa'
    fn, _ = _probe_fa3()
    if impl == 'fa3':
        if fn is None:
            # Explicit FA3 requests fail instead of silently changing the backend.
            raise RuntimeError(
                "full_attn_impl='fa3', but importing flash_attn_interface failed. "
                "Install FA3 from the flash-attention repository's hopper directory, "
                "or use full_attn_impl='auto' to allow fallback to SDPA."
            )
        return 'fa3'
    return 'fa3' if fn is not None else 'sdpa'


def describe(resolved: str) -> str:
    """Return a one-line description of the resolved backend for logging."""
    if resolved == 'fa3':
        _, kind = _probe_fa3()
        return f"full attention = FA3 ({kind})"
    return "full attention = SDPA (torch)"


def full_attention(q: Tensor, k: Tensor, v: Tensor, resolved: str) -> Tensor:
    """Compute full attention in BHLD layout, preserving the query shape.

    Query and key sequence lengths may differ. Both supported backends use
    exact softmax attention. Pass a backend already returned by resolve_impl
    so the hot path does not repeat backend selection.
    """
    if resolved == 'fa3':
        fn, _ = _probe_fa3()
        # FA3 expects BSHD; the internal layout is BHSD.
        out = fn(q.transpose(1, 2).contiguous(),
                 k.transpose(1, 2).contiguous(),
                 v.transpose(1, 2).contiguous(),
                 causal=False)
        if isinstance(out, tuple):
            out = out[0]
        return out.transpose(1, 2).contiguous()

    # Use SDPA's default 1/sqrt(head_dim) scale, matching the native processor.
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=False)


def full_attention_with_lse(q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
    """Return full attention and log-sum-exp from a single FA3 call.

    Calibration uses the output as its result and the LSE to score KV tokens.
    This function requires FA3 and has no fallback.

    The supported FA3 interface returns (output, softmax_lse) when
    return_attn_probs=True. Without that flag it returns a single tensor;
    unpacking that tensor can silently split the batch dimension.

    LSE has shape (B, H, L_q), uses natural logarithms, and already includes
    the attention scale. The token-probability kernel uses exp(scores - lse),
    so neither a log-base conversion nor another scale factor is needed.
    The LSE-returning kernel can have a separate first-call compilation cost.

    Returns:
        output: (B, H, L_q, D), with the same dtype as q.
        lse: (B, H, L_q), float32.
    """
    fn, _ = _probe_fa3()
    if fn is None:
        raise RuntimeError(
            "Calibration requires full attention and LSE from FlashAttention-3, "
            "but importing flash_attn_interface failed. Install FA3 from the "
            "flash-attention repository's hopper directory. Token-sparse attention "
            "cannot run calibration without FA3."
        )
    # FA3 expects BSHD; the internal layout is BHSD.
    out, lse = fn(q.transpose(1, 2).contiguous(),
                  k.transpose(1, 2).contiguous(),
                  v.transpose(1, 2).contiguous(),
                  causal=False, return_attn_probs=True)
    # Restore BHLD output; LSE is already in (B, H, L) order.
    return out.transpose(1, 2).contiguous().to(q.dtype), lse.float()

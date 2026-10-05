"""BF16 token-probability scoring for sparse selection.

Calibration supplies full-attention LSE. This pass computes
exp(Q @ K.T * scale - LSE), averages over each query block, and returns a
score for every (query block, key token) pair for token selection.
"""

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

# Token-level probability kernel (BLKK=1).









@triton.jit
def _token_probs_kernel_bf16(
    Q, K, LSE, OUT,
    stride_qbh, stride_ql, stride_qd,
    stride_kbh, stride_kl, stride_kd,
    stride_lbh, stride_ll,
    stride_obh, stride_om, stride_on,
    L_Q: tl.constexpr, L_K: tl.constexpr, D: tl.constexpr,
    M: tl.constexpr,
    BLKQ: tl.constexpr,
    TILE_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    scale,
):
    """BF16 token-level probability kernel with BLKK=1."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)

    q_base = Q + pid_bh * stride_qbh
    k_base = K + pid_bh * stride_kbh

    q_row_offs = pid_m * BLKQ + tl.arange(0, BLKQ)
    k_row_offs = pid_n * TILE_K + tl.arange(0, TILE_K)
    d_offs = tl.arange(0, BLOCK_D)

    q_mask = (q_row_offs[:, None] < L_Q) & (d_offs[None, :] < D)
    k_mask = (k_row_offs[:, None] < L_K) & (d_offs[None, :] < D)
    q_tile = tl.load(
        q_base + q_row_offs[:, None] * stride_ql + d_offs[None, :] * stride_qd,
        mask=q_mask,
        other=0.0,
    )
    k_tile = tl.load(
        k_base + k_row_offs[:, None] * stride_kl + d_offs[None, :] * stride_kd,
        mask=k_mask,
        other=0.0,
    )

    scores = tl.dot(q_tile, tl.trans(k_tile)) * scale
    scores = tl.where(k_row_offs[None, :] < L_K, scores, -float("inf"))
    scores = tl.where(q_row_offs[:, None] < L_Q, scores, -float("inf"))

    lse_base = LSE + pid_bh * stride_lbh
    lse_vals = tl.load(
        lse_base + q_row_offs * stride_ll,
        mask=q_row_offs < L_Q,
        other=1e12,
    )
    probs = tl.exp(scores - lse_vals[:, None])
    probs = tl.where((q_row_offs < L_Q)[:, None], probs, 0.0)

    actual_blkq = tl.minimum(BLKQ, L_Q - pid_m * BLKQ)
    token_probs = tl.sum(probs, axis=0) / actual_blkq
    token_probs = tl.where(k_row_offs < L_K, token_probs, 0.0)

    out_base = OUT + pid_bh * stride_obh + pid_m * stride_om
    out_offs = pid_n * TILE_K + tl.arange(0, TILE_K)
    tl.store(
        out_base + out_offs * stride_on,
        token_probs.to(OUT.dtype.element_ty),
        mask=out_offs < L_K,
    )


def _compute_token_probs_bf16_triton(q, k, lse, BLKQ, M, scale):
    """Compute token-level probabilities with the BF16 Triton kernel (BLKK=1)."""
    B, H, L_Q, D = q.shape
    L_K = k.shape[2]
    BH = B * H
    BLOCK_D = triton.next_power_of_2(D)
    TILE_K = 64
    N_tiles = (L_K + TILE_K - 1) // TILE_K

    q_flat = q.reshape(BH, L_Q, D).contiguous()
    k_flat = k.reshape(BH, L_K, D).contiguous()
    lse_flat = lse.reshape(BH, L_Q).contiguous()
    out = torch.zeros(BH, M, L_K, device=q.device, dtype=q.dtype)

    grid = (M, N_tiles, BH)
    _token_probs_kernel_bf16[grid](
        q_flat, k_flat, lse_flat, out,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        lse_flat.stride(0), lse_flat.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        L_Q, L_K, D, M,
        BLKQ, TILE_K, BLOCK_D,
        scale,
    )
    return out.reshape(B, H, M, L_K)











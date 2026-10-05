"""Fast PDDP groups similar queries into tile-aligned blocks.

Only Q is reordered. K and V retain their original order, so gather indices
continue to address the original keys. Invert the permutation after attention
to restore the original query order.

At each tree depth, batch nodes and heads together. Compute the centered Gram
matrix using X.T @ X - s @ s.T / n, find its principal direction with power
iteration, sort the projected queries, and split the block interval in two.
The Gram identity avoids allocating a centered sequence-sized copy. Gram
accumulation and power iteration use float32.

Splits fall on block boundaries, so arbitrary sequence lengths are supported;
only the rightmost leaf may be shorter than C. Only permutation indices are
carried between depths. proj_dim optionally discards low-variance directions
for grouping; the returned permutation still indexes the original sequence.

    perm = pca_group_perm(q.reshape(B * H, L, D), C=128)
    q_grouped = apply_perm_bhld(q, perm)
    # Compute attention in grouped query order.
    output = invert_perm_bhld(output_grouped, perm)
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "pca_group_perm",
    "apply_perm",
    "invert_perm",
    "apply_perm_bhld",
    "invert_perm_bhld",
    "leaf_inertia",
    "identity_perm",
]


def apply_perm(x: Tensor, perm: Tensor) -> Tensor:
    """Reorder (BH, N, D) into leaf order using a (BH, N) permutation."""
    return x.gather(1, perm.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def invert_perm(x_permuted: Tensor, perm: Tensor) -> Tensor:
    """Invert apply_perm and restore the original token order."""
    inv = perm.argsort(dim=1)
    return x_permuted.gather(
        1, inv.unsqueeze(-1).expand(-1, -1, x_permuted.shape[-1]))


def apply_perm_bhld(x: Tensor, perm: Tensor) -> Tensor:
    """Apply a (B*H, L) permutation to a (B, H, L, D) tensor."""
    B, H, L, D = x.shape
    return apply_perm(x.reshape(B * H, L, D), perm).reshape(B, H, L, D)


def invert_perm_bhld(x: Tensor, perm: Tensor) -> Tensor:
    """Invert apply_perm_bhld and restore the original query order."""
    B, H, L, D = x.shape
    return invert_perm(x.reshape(B * H, L, D), perm).reshape(B, H, L, D)


def identity_perm(BH: int, N: int, device) -> Tensor:
    """Return the identity permutation for the no-reordering baseline."""
    return torch.arange(N, device=device).unsqueeze(0).expand(BH, N).contiguous()


def leaf_inertia(x: Tensor, perm: Tensor, C: int) -> float:
    """Return the mean squared distance from tokens to their leaf centroids.

    Compare against identity_perm at the same shape and data scale. The final
    partial leaf uses its actual token count without padding.
    """
    BH, N, D = x.shape
    xp = apply_perm(x, perm).float()
    total, count = 0.0, 0
    for start in range(0, N, C):
        blk = xp[:, start:start + C]
        if blk.shape[1] < 2:
            continue
        dev = blk - blk.mean(dim=1, keepdim=True)
        total += dev.pow(2).sum(-1).sum().item()
        count += blk.shape[0] * blk.shape[1]
    return total / max(count, 1)


def _orth(P: Tensor, iters: int = 6) -> Tensor:
    """Orthonormalize columns of P (M, n, k) using Newton-Schulz iteration.

    Batched matrix products approximate (P.T @ P)^(-1/2). The result is used
    only to span a subspace for finding split directions.
    """
    M = torch.bmm(P.transpose(1, 2), P)
    k = M.shape[-1]
    I = torch.eye(k, device=M.device, dtype=M.dtype).expand_as(M).contiguous()
    nrm = M.diagonal(dim1=1, dim2=2).sum(1).clamp(min=1e-12)[:, None, None]
    Y, Z = M / nrm, I.clone()
    for _ in range(iters):
        T = 0.5 * (3.0 * I - torch.bmm(Z, Y))
        Y, Z = torch.bmm(Y, T), torch.bmm(T, Z)
    return torch.bmm(P, Z / nrm.sqrt())


def _split_block_ranges(ranges):
    """Bisect each block interval [lo, hi); keep single-block intervals unchanged."""
    out = []
    for lo, hi in ranges:
        if hi - lo <= 1:
            out.append((lo, hi))
        else:
            mid = lo + (hi - lo) // 2
            out.append((lo, mid))
            out.append((mid, hi))
    return out


def _project_to_top_dims(x: Tensor, proj_dim: int, seed: int) -> Tensor:
    """Project x onto the highest-variance directions of its global covariance.

    This discards low-variance information when proj_dim < D. Columns are
    ordered by variance, so taking the leading columns preserves the highest
    variance available in a contiguous slice.
    """
    BH, N, D = x.shape
    s0 = x.sum(1, dtype=torch.float32)                        # (BH, D)
    G0 = torch.bmm(x.transpose(1, 2), x).float()              # (BH, D, D)
    G0 -= (s0.unsqueeze(2) * s0.unsqueeze(1)) / N
    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed + 1234)
    P = torch.randn(BH, D, proj_dim, device=x.device,
                    dtype=torch.float32, generator=gen)
    for _ in range(2):
        P = _orth(torch.bmm(G0, P))
    val = torch.einsum("hdk,hde,hek->hk", P, G0, P)
    P = P.gather(2, val.argsort(1, descending=True)
                 .unsqueeze(1).expand(-1, D, -1))
    return torch.bmm(x, P.to(x.dtype))


@torch.no_grad()
def pca_group_perm(
    x: Tensor,
    C: int = 128,
    pow_iters: int = 4,
    seed: int = 0,
    proj_dim: int | None = None,
) -> Tensor:
    """Return a per-head permutation from recursive PCA partitioning.

    Args:
        x: (BH, N, D), with batch and heads flattened; heads are independent.
        C: Tokens per leaf, aligned with the sparse kernel query tile.
            N need not be divisible by C or form a power-of-two leaf count.
        pow_iters: Power iterations per node.
        seed: Seed for a private generator; the caller's RNG is unchanged.
        proj_dim: Optional projection onto leading variance directions before
            grouping. None keeps all dimensions. Projection affects grouping
            quality; compare leaf_inertia before enabling it.

    Returns:
        perm: (BH, N) int64. x[b, perm[b, i]] is the i-th reordered token.
            The indices always refer to the original sequence, even when
            proj_dim is enabled.
    """
    BH, N, D = x.shape
    device = x.device
    if N <= C:
        return identity_perm(BH, N, device)

    if proj_dim is not None and proj_dim < D:
        if proj_dim <= 0:
            raise ValueError(f"proj_dim must be positive, got {proj_dim}")
        x = _project_to_top_dims(x, proj_dim, seed)
        D = proj_dim

    K = (N + C - 1) // C            # Ceiling block count; the final block may be partial.
    perm = identity_perm(BH, N, device)

    # Append a zero row for padded positions in variable-length nodes.
    # It contributes nothing to the Gram matrix or projections.
    x_ext = torch.cat([x, x.new_zeros(BH, 1, D)], dim=1)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    ranges = [(0, K)]
    while any(hi - lo > 1 for lo, hi in ranges):
        # Map each block interval to [lo*C, min(hi*C, N)) token positions.
        starts = torch.tensor([lo * C for lo, hi in ranges], device=device)
        ends = torch.tensor([min(hi * C, N) for lo, hi in ranges], device=device)
        sizes = ends - starts
        nodes = len(ranges)
        maxlen = int(sizes.max().item())

        ar = torch.arange(maxlen, device=device)
        pos = starts.unsqueeze(1) + ar.unsqueeze(0)
        valid = ar.unsqueeze(0) < sizes.unsqueeze(1)
        pos_safe = torch.where(valid, pos, torch.zeros_like(pos))

        flat_pos = pos_safe.reshape(1, -1).expand(BH, -1)       # (BH, nodes*maxlen)
        ids = perm.gather(1, flat_pos)
        flat_valid = valid.reshape(1, -1).expand(BH, -1)
        ids = torch.where(flat_valid, ids, torch.full_like(ids, N))  # Map padding to the appended zero row.
        ids = ids.reshape(BH * nodes, maxlen)

        # Keep gathered node data in the input dtype to limit peak memory.
        M = BH * nodes
        Xg = x_ext.gather(
            1, ids.reshape(BH, nodes * maxlen).unsqueeze(-1).expand(-1, -1, D)
        ).reshape(M, maxlen, D)

        vmask = flat_valid.reshape(M, maxlen)
        # Padding is zero; normalize using the actual node lengths.
        cnt = vmask.sum(1, keepdim=True).clamp(min=1).float()   # (M, 1)
        s = Xg.sum(1, dtype=torch.float32)                      # (M, D)

        # Centered Gram identity: Xc.T @ Xc = X.T @ X - s @ s.T / n.
        # Accumulate the batched products in float32.
        G = torch.bmm(Xg.transpose(1, 2), Xg).float()            # (M, D, D)
        G -= (s.unsqueeze(2) * s.unsqueeze(1)) / cnt.unsqueeze(2)

        # Power iteration finds the principal direction in feature space.
        v = torch.randn(M, D, device=device, generator=gen)
        v /= v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        for _ in range(pow_iters):
            v = torch.bmm(G, v.unsqueeze(-1)).squeeze(-1)
            v /= v.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        # A constant mean shift does not affect projection order.
        # Set padded projections to +inf so padding remains in the rightmost leaf.
        a = torch.bmm(Xg, v.to(Xg.dtype).unsqueeze(-1)).squeeze(-1).float()
        a = torch.where(vmask, a, torch.full_like(a, float('inf')))
        order = a.argsort(dim=1)
        ids_sorted = ids.gather(1, order).reshape(BH, nodes * maxlen)

        # Write back only valid positions; padding has no token slot.
        keep = flat_valid[0]
        perm = perm.scatter(1, flat_pos[:, keep], ids_sorted[:, keep])

        ranges = _split_block_ranges(ranges)

    return perm

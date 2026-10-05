"""Token-level sparse attention with BF16 CuTeDSL gather-KV on SM90.

Selection modes are topk, topp, and topk_topp. Each query block shares a KV
selection. topp and topk_topp can produce different KV counts per query block;
gather_kv_lengths tells the kernel how many aligned tiles to evaluate.

This integration uses BF16 selection and attention; no FP8 configuration is
exposed by TokenSparseConfig.
"""
import os
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from copy import deepcopy
import torch
from torch import Tensor

# Optional stage timing, enabled by FLASH_MCSA_TIMING=1.
from .timing import timed as _timed
from einops import rearrange


def _token_sparse_debug_enabled() -> bool:
    return os.getenv("FLASH_TOKEN_SPARSE_DEBUG", "").lower() in ("1", "true", "yes", "on")


def _token_sparse_check_enabled() -> bool:
    """Return whether tensor checks are enabled by FLASH_TOKEN_SPARSE_CHECK.

    These checks report non-finite values, unexpected shapes, and magnitudes
    independently of the verbose debug mode.
    """
    return os.getenv("FLASH_TOKEN_SPARSE_CHECK", "").lower() in ("1", "true", "yes", "on")


def _ck(tag: str, layer, step, **tensors):
    """Check finiteness and magnitude when FLASH_TOKEN_SPARSE_CHECK is enabled.

    Scalar reads synchronize the device, so this diagnostic is disabled by
    default and should only be used for troubleshooting.
    """
    if not _token_sparse_check_enabled():
        return
    msgs = []
    for name, t in tensors.items():
        if t is None:
            msgs.append(f"{name}=None")
            continue
        if not isinstance(t, Tensor):
            msgs.append(f"{name}={t}")
            continue
        f = t.float()
        n_nan = int(torch.isnan(f).sum().item())
        n_inf = int(torch.isinf(f).sum().item())
        absmax = float(f.abs().max().item()) if f.numel() else 0.0
        absmean = float(f.abs().mean().item()) if f.numel() else 0.0
        flag = ""
        if n_nan or n_inf:
            flag = "  <<< NAN/INF"
        elif absmax > 1e4:
            flag = "  <<< unexpected magnitude"
        msgs.append(
            f"{name}[{tuple(t.shape)},{str(t.dtype).replace('torch.','')}] "
            f"absmax={absmax:.4g} absmean={absmean:.4g} "
            f"nan={n_nan} inf={n_inf}{flag}"
        )
    print(f"[TSCHECK] {tag} layer={layer} step={step} | " + " | ".join(msgs),
          flush=True)


def _debug_token_sparse_stats(gi: Tensor, gl: Optional[Tensor], config, orig_L_K: int):
    """Print token-selection statistics when debug logging is enabled."""
    if not _token_sparse_debug_enabled():
        return
    B, H, M, gather_len = gi.shape
    tile_n = config.tile_n
    max_n_blocks = gather_len // tile_n
    if gl is not None:
        lengths_f = gl.float()
        mean_nb = lengths_f.mean().item()
        min_nb = gl.min().item()
        max_nb = gl.max().item()
        mean_tokens = mean_nb * tile_n
        ratio_mean = mean_tokens / max(orig_L_K, 1)
        print(
            f"[TokenSparse] mode={config.selection_mode} topk={config.topk:.4f} "
            f"topp={config.topp} L_K={orig_L_K} gather_len={gather_len} "
            f"n_blocks min/mean/max={min_nb}/{mean_nb:.1f}/{max_nb} "
            f"tokens_mean={mean_tokens:.0f} ratio_mean={ratio_mean:.4f} "
            f"max_possible={max_n_blocks}",
            flush=True,
        )
    else:
        ratio = gather_len / max(orig_L_K, 1)
        print(
            f"[TokenSparse] mode={config.selection_mode} topk={config.topk:.4f} "
            f"L_K={orig_L_K} gather_len={gather_len} "
            f"n_blocks={max_n_blocks} ratio={ratio:.4f}",
            flush=True,
        )


@dataclass
class TokenSparseConfig:
    """Configure token-level sparse attention.

    Args:
        topk: Fraction of KV tokens selected per query group.
        topp: Cumulative probability threshold for topp-based selection.
        selection_mode: 'topk', 'topp', or 'topk_topp'.
        BLKQ: Query tile size, fixed at 128.
        tile_n: KV tile size and count-alignment unit, fixed at 128.
        sink_ranges: Half-open KV ranges always included for every query.
            Sink tokens are added to, not subtracted from, the sparse budget.
        calibrate_steps: Steps that compute selections and calibration data.
        update_steps: Steps that refresh selections using cached LSE.
        full_steps: Steps that use full attention.
        full_layers: Layers that always use full attention.
        share_lse: Whether to reuse LSE across layers.
        cal_downsample: Selection downsampling ratio or (q_ratio, k_ratio).

    This configuration exposes BF16 selection and attention. Residuals retain
    their original dtype. FP8 and residual_dtype switches are not exposed.
    """
    topk: float = 0.1
    topp: Optional[float] = None  # Cumulative probability threshold for topp-based modes
    selection_mode: str = 'topk'  # 'topk' | 'topp' | 'topk_topp'
    BLKQ: int = 128
    tile_n: int = 128
    sink_ranges: Optional[List[Tuple[int, int]]] = None  # Half-open ranges of always-included KV tokens
    calibrate_steps: Optional[List[int]] = None
    update_steps: Optional[List[int]] = None
    full_steps: Optional[List[int]] = None
    full_layers: List[int] = field(default_factory=list)
    share_lse: bool = False
    cal_downsample: Union[int, Tuple[int, int]] = 1  # Integer ratio or (query_ratio, key_ratio)
    adaptive_heads: bool = False  # Allocate head budgets by recall
    adaptive_layers: bool = False  # Adapt budgets by normalized layer depth
    use_chunk_selector: bool = False  # Score tokens in chunks instead of materializing all probabilities
    chunk_selector_size: int = 16384
    offload_indices: bool = False  # Offload cached indices and prefetch before use
    approx_steps: Optional[List[int]] = None  # Steps that select from mean-pooled query scores
    bias_correction: bool = False  # Add the cached dense-minus-sparse residual on reuse steps
    cache_offload: str = 'none'  # 'none' or 'residual'; controls residual offloading only
    # Backend for pure full steps/layers. Calibration always requires FA3
    # to return the full-attention output and LSE together.
    full_attn_impl: str = 'sdpa'
    # Group similar queries; refresh permutations and selected indices together.
    pca_reorder: bool = False
    pca_pow_iters: int = 4
    # Optional low-rank projection for grouping. Dropped dimensions affect quality;
    # compare grouping quality before enabling pca_proj_dim.
    pca_proj_dim: Optional[int] = None
    # Use the centered-key metric for grouping only; attention keeps the original Q.
    # For Ks = K - mean(K), the metric is (Ks.T @ Ks / T)^(1/2).
    pca_whiten: bool = False
    def __post_init__(self):
        assert 0 < self.topk <= 1.0
        assert self.cache_offload in ('none', 'residual'), \
            f"cache_offload must be 'none' or 'residual', got {self.cache_offload}"
        assert self.full_attn_impl in ('sdpa', 'fa3', 'auto'), \
            f"full_attn_impl must be 'sdpa'/'fa3'/'auto', got {self.full_attn_impl}"
        # Resolve once during construction so backend failures surface immediately.
        from .ops.full_attn import resolve_impl as _resolve_full_attn
        self._full_attn_resolved = _resolve_full_attn(self.full_attn_impl)

        assert self.selection_mode in ('topk', 'topp', 'topk_topp'), \
            f"selection_mode must be 'topk', 'topp', or 'topk_topp', got '{self.selection_mode}'"
        if self.selection_mode in ('topp', 'topk_topp'):
            assert self.topp is not None and 0 < self.topp <= 1.0, \
                f"selection_mode='{self.selection_mode}' requires topp in (0, 1], got {self.topp}"
        assert self.BLKQ == 128
        assert self.tile_n == 128
        self._calibrate_set = set(self.calibrate_steps) if self.calibrate_steps else None
        self._update_set = set(self.update_steps) if self.update_steps else None
        self._full_set = set(self.full_steps) if self.full_steps else None
        self._approx_set = set(self.approx_steps) if self.approx_steps else None
        if isinstance(self.cal_downsample, (list, tuple)):
            self._cal_ds_q = int(self.cal_downsample[0])
            self._cal_ds_k = int(self.cal_downsample[1])
        else:
            self._cal_ds_q = int(self.cal_downsample)
            self._cal_ds_k = int(self.cal_downsample)
        assert self._cal_ds_q >= 1 and self._cal_ds_k >= 1, "cal_downsample must contain positive integers"
        assert self.BLKQ % self._cal_ds_q == 0, "BLKQ must be divisible by cal_downsample_q"
        if self.bias_correction:
            assert self._cal_ds_q == 1 and self._cal_ds_k == 1, \
                "bias_correction requires cal_downsample=1 (downsampling is not supported)"
        if self.sink_ranges:
            for start, end in self.sink_ranges:
                assert 0 <= start < end, f"Invalid sink_ranges interval: ({start}, {end})"
        # Cross-layer LSE sharing cannot follow per-layer query permutations.
        # Residuals instead carry a permutation generation and are dropped if stale.
        if self.pca_reorder:
            assert not self.share_lse, \
                "pca_reorder is incompatible with share_lse: LSE is per-query, and each layer " \
                "has its own permutation. Sharing LSE across layers would misalign query rows."

class TokenSparseCache:
    def __init__(self, config):
        self.config = config
        self._step_count = 0
        self._cached_lse = None
        self._cached_indices = None
        self._cached_lengths = None  # (B, H, M), int32 valid tile counts
        # Permutation: (B*H, L_Q), int64. Its lifetime matches the selected indices
        # because those indices were chosen for these exact query groups.
        self._cached_perm = None
        self.layer_t: Optional[float] = None  # Normalized layer depth in [0, 1]
        self._cached_residual = None  # (B, H, L, D), calibration dense-minus-sparse output
        self._cached_residual_cpu = None  # Pinned CPU residual copy
        self._residual_offload_event = None
        # A separate residual stream avoids serializing its D2H with index offloading.
        self._residual_offload_stream = None
        self._residual_prefetch_stream = None
        # The pinned CPU residual is authoritative. Prefetched GPU copies are transient
        # and must be released after consumption to retain the offloading benefit.
        self._prefetched_residual = None
        self._residual_prefetch_event = None
        # Advance on every permutation refresh; residuals belong to that query order.
        self._perm_generation = 0
        # Permutation generation associated with the residual; None without reordering.
        self._residual_perm_gen = None
        self._indices_cpu = None  # Pinned CPU indices
        self._lengths_cpu = None  # Pinned CPU tile counts
        self._prefetched_indices = None  # Temporary prefetched GPU indices
        self._prefetched_lengths = None
        self._prefetch_event = None  # Event marking prefetch completion
        self._prefetch_stream = None  # Lazily created, reusable index-prefetch stream
        self._offload_stream = None  # GPU-to-CPU transfer stream
        self._offload_event = None  # Event marking offload completion
    @property
    def mode(self):
        s = self._step_count
        c = self.config
        if c._full_set and s in c._full_set:
            # Overlapping full/calibration steps without downsampling share a full pass.
            if c._cal_ds_q == 1 and c._cal_ds_k == 1:
                if c._calibrate_set and s in c._calibrate_set:
                    return 'full_calibrate'
            return 'full'
        if c._calibrate_set and s in c._calibrate_set: return 'calibrate'
        if c._approx_set and s in c._approx_set: return 'approx'
        if c._update_set and s in c._update_set: return 'update'
        if not self.has_cache: return 'calibrate'
        return 'reuse'
    @property
    def has_cache(self):
        return self._cached_indices is not None or self._indices_cpu is not None
    @property
    def has_lse(self): return self._cached_lse is not None

    def get_or_make_perm(self, q, recompute, k=None):
        """Return the cached query permutation, recomputing it when required.

        Args:
            q: Full-resolution (B, H, L, D) queries. Call before downsampling:
                pooling and reordering do not commute, and the permutation
                must retain the full query length.
            recompute: Refresh the permutation together with the selection
                so indices refer to the matching query groups.
            k: (B, H, L, D) keys, required by pca_whiten to construct the
                key-derived metric for grouping.
        """
        B, H, L, D = q.shape
        stale = (self._cached_perm is None
                 or tuple(self._cached_perm.shape) != (B * H, L))
        if recompute or stale:
            from .pca_group import pca_group_perm
            q_for_perm = q
            if getattr(self.config, "pca_whiten", False):
                if k is None:
                    raise ValueError(
                        "pca_whiten=True requires k; the whitening matrix is derived from key moments. "
                        "Check that the get_or_make_perm call forwards k.")
                with _timed("pca.whiten"):
                    q_for_perm = _whiten_q_for_perm(q, k)
            with _timed("pca.perm"):
                self._cached_perm = pca_group_perm(
                    q_for_perm.reshape(B * H, L, D), C=self.config.BLKQ,
                    pow_iters=self.config.pca_pow_iters,
                    proj_dim=getattr(self.config, "pca_proj_dim", None),
                )
            # Changing query order invalidates the old residual.
            # The generation is checked again when the residual is consumed.
            self._perm_generation += 1
            if _token_sparse_debug_enabled():
                print(f"[FlashTokenSparse] pca_perm recomputed "
                      f"layer={getattr(self, '_layer_idx', '?')} step={self._step_count} "
                      f"shape={tuple(self._cached_perm.shape)} stale={stale} "
                      f"gen={self._perm_generation}", flush=True)
        return self._cached_perm

    def residual_matches_perm(self):
        """Return whether the residual matches the current query permutation."""
        if not self.config.pca_reorder:
            return True
        return self._residual_perm_gen == self._perm_generation

    def check_seq_len(self, L):
        gi = self._cached_indices if self._cached_indices is not None else self._indices_cpu
        if gi is None: return False
        M = (L + self.config.BLKQ - 1) // self.config.BLKQ
        return gi.shape[2] != M

    def _ensure_offload_done(self):
        """Wait for pending GPU-to-CPU offloading before reading or prefetching it."""
        if self._offload_event is not None:
            self._offload_event.synchronize()
            self._offload_event = None

    def offload_to_cpu(self, gi, gl, device):
        """Asynchronously offload indices to pinned CPU memory alongside attention.

        Args:
            gi: (B, H, M, gather_len) int32 CUDA indices.
            gl: (B, H, M) int32 CUDA tile counts, or None.
            device: CUDA device that owns the source tensors.
        """
        if self._offload_stream is None:
            self._offload_stream = torch.cuda.Stream(device=device)

        if self._indices_cpu is None or self._indices_cpu.shape != gi.shape:
            self._indices_cpu = torch.empty(gi.shape, dtype=gi.dtype, device='cpu', pin_memory=True)
        if gl is not None:
            if self._lengths_cpu is None or self._lengths_cpu.shape != gl.shape:
                self._lengths_cpu = torch.empty(gl.shape, dtype=gl.dtype, device='cpu', pin_memory=True)
        else:
            self._lengths_cpu = None

        # Wait for indices produced on the current stream before starting D2H.
        self._offload_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self._offload_stream):
            self._indices_cpu.copy_(gi, non_blocking=True)
            if gl is not None:
                self._lengths_cpu.copy_(gl, non_blocking=True)
            # The offload stream reads tensors allocated on another stream.
            # Record its use so the allocator cannot recycle the source during D2H.
            gi.record_stream(self._offload_stream)
            if gl is not None:
                gl.record_stream(self._offload_stream)
        self._offload_event = self._offload_stream.record_event()

    def prefetch(self, device):
        """Prefetch CPU indices to the GPU while the preceding layer executes."""
        if self._indices_cpu is None:
            return
        # Wait until the pinned CPU copy is ready before prefetching.
        self._ensure_offload_done()
        # Create the prefetch stream lazily and retain it for reuse.
        if self._prefetch_stream is None:
            self._prefetch_stream = torch.cuda.Stream(device=device)
        stream = self._prefetch_stream
        with torch.cuda.stream(stream):
            self._prefetched_indices = self._indices_cpu.to(device, non_blocking=True)
            self._prefetched_lengths = (
                self._lengths_cpu.to(device, non_blocking=True)
                if self._lengths_cpu is not None else None
            )
        self._prefetch_event = stream.record_event()

    def _get_prefetched_indices(self, device):
        """Return prefetched GPU indices, transferring on demand if needed."""
        if self._prefetched_indices is not None:
            # Wait on the current CUDA stream without blocking the CPU.
            if self._prefetch_event is not None:
                torch.cuda.current_stream(device).wait_event(self._prefetch_event)
                self._prefetch_event = None
            gi = self._prefetched_indices
            gl = self._prefetched_lengths
            # Prefetched tensors were allocated on another stream but are consumed here.
            # Record the current stream before releasing references, or the allocator can
            # recycle indices while the attention kernel is still reading them.
            cur = torch.cuda.current_stream(device)
            gi.record_stream(cur)
            if gl is not None:
                gl.record_stream(cur)
            # Release the temporary GPU copy; a later reuse step will prefetch again.
            self._prefetched_indices = None
            self._prefetched_lengths = None
            return gi, gl
        if self._indices_cpu is not None:
            self._ensure_offload_done()  # Wait for offloading to finish.
            gi = self._indices_cpu.to(device, non_blocking=False)
            gl = self._lengths_cpu.to(device, non_blocking=False) if self._lengths_cpu is not None else None
            return gi, gl
        return self._cached_indices, self._cached_lengths

    @property
    def has_residual(self):
        return (self._cached_residual is not None
                or self._prefetched_residual is not None
                or self._cached_residual_cpu is not None)

    def offload_residual_to_cpu(self, device=None):
        """Offload the residual to pinned CPU memory and release its GPU reference."""
        residual = self._cached_residual
        if residual is None or not residual.is_cuda:
            return False
        if device is None:
            device = residual.device
        if self._residual_offload_stream is None:
            self._residual_offload_stream = torch.cuda.Stream(device=device)
        stream = self._residual_offload_stream
        # Reuse pinned buffers across steps; reallocate if shape or dtype changes.
        if (self._cached_residual_cpu is None
                or self._cached_residual_cpu.shape != residual.shape
                or self._cached_residual_cpu.dtype != residual.dtype):
            self._cached_residual_cpu = torch.empty(
                residual.shape,
                dtype=residual.dtype,
                device='cpu',
                pin_memory=True,
            )
        stream.wait_stream(torch.cuda.current_stream(device))
        # Wait for an earlier H2D read before overwriting its pinned source buffer.
        if self._residual_prefetch_event is not None:
            stream.wait_event(self._residual_prefetch_event)
        with torch.cuda.stream(stream):
            self._cached_residual_cpu.copy_(residual, non_blocking=True)
            residual.record_stream(stream)
        self._residual_offload_event = stream.record_event()
        self._cached_residual = None
        return True

    def prefetch_residual(self, device):
        """Asynchronously prefetch the residual to overlap H2D with earlier layers.

        Return True if a transfer starts. Skip an existing or pending GPU copy.
        """
        def _log(result):
            if _token_sparse_debug_enabled():
                print(f"[FlashTokenSparse] residual_prefetch "
                      f"layer={getattr(self, '_layer_idx', '?')} "
                      f"step={self._step_count} -> {result}", flush=True)

        if self._cached_residual is not None or self._prefetched_residual is not None:
            _log("skip(GPU copy already exists)")
            return False
        if self._cached_residual_cpu is None:
            _log("skip(no CPU copy; residual has not been offloaded)")
            return False
        if device.type != 'cuda':
            _log("skip(non-CUDA device)")
            return False
        if self._residual_prefetch_stream is None:
            self._residual_prefetch_stream = torch.cuda.Stream(device=device)
        stream = self._residual_prefetch_stream
        # Wait for D2H to finish before reading the pinned residual.
        if self._residual_offload_event is not None:
            stream.wait_event(self._residual_offload_event)
        with torch.cuda.stream(stream):
            gpu = self._cached_residual_cpu.to(device, non_blocking=True)
        self._prefetched_residual = gpu
        self._residual_prefetch_event = stream.record_event()
        _log(f"Starting H2D transfer: {gpu.numel()*gpu.element_size()/2**20:.0f} MB")
        return True

    def get_residual(self, device):
        """Return the residual, ready for use on the current CUDA stream.

        The caller must not retain it long-term. Storage preserves its dtype.
        """
        return self._get_residual_raw(device)

    def _get_residual_raw(self, device):
        """Retrieve the residual in its stored dtype."""
        if self._cached_residual is not None:
            return self._cached_residual
        if self._prefetched_residual is not None:
            if self._residual_prefetch_event is not None:
                torch.cuda.current_stream(device).wait_event(self._residual_prefetch_event)
                self._residual_prefetch_event = None
            residual = self._prefetched_residual
            # Record the consuming stream before releasing the prefetched GPU residual.
            residual.record_stream(torch.cuda.current_stream(device))
            # Discard the GPU copy after use; the CPU residual remains authoritative.
            self._prefetched_residual = None
            return residual
        if self._cached_residual_cpu is None:
            return None
        # On a prefetch miss, copy pinned data on the consuming stream.
        # Stream ordering guarantees readiness for the subsequent kernel.
        if self._residual_offload_event is not None:
            torch.cuda.current_stream(device).wait_event(self._residual_offload_event)
        return self._cached_residual_cpu.to(device, non_blocking=True)

    def _store_residual(self, residual):
        """Store the residual in its original dtype.

        Immediately offload it when cache_offload is 'residual'.
        """
        # A new residual invalidates any prefetched copy of the previous one.
        self._prefetched_residual = None
        residual = residual.detach()
        self._cached_residual = residual
        # Associate the residual with the permutation that produced it.
        self._residual_perm_gen = (
            self._perm_generation if self.config.pca_reorder else None
        )
        if self.config.cache_offload == 'residual':
            self.offload_residual_to_cpu(residual.device)

    def drop_stale_residual(self):
        """Drop a residual that belongs to a different query permutation.

        Return True when dropped. Reuse then produces a sparse-only output
        until calibration records a compatible residual.
        """
        if not self.has_residual or self.residual_matches_perm():
            return False
        if self._residual_offload_event is not None:
            self._residual_offload_event.synchronize()
            self._residual_offload_event = None
        self._cached_residual = None
        self._cached_residual_cpu = None
        self._prefetched_residual = None
        self._residual_prefetch_event = None
        self._residual_perm_gen = None
        return True

    def reset(self):
        self._step_count = 0
        self._cached_lse = None
        self._cached_indices = None
        self._cached_lengths = None
        self._cached_perm = None
        self._perm_generation = 0
        # Wait for pending D2H before releasing the source GPU residual.
        if self._residual_offload_event is not None:
            self._residual_offload_event.synchronize()
        self._cached_residual = None
        self._cached_residual_cpu = None
        self._residual_offload_event = None
        self._prefetched_residual = None
        self._residual_prefetch_event = None
        self._residual_perm_gen = None
        if self._offload_event is not None:
            self._offload_event.synchronize()
        self._indices_cpu = None
        self._lengths_cpu = None
        self._prefetched_indices = None
        self._prefetched_lengths = None
        self._prefetch_event = None
        self._offload_event = None
        # Retain reusable streams and the layer_t value assigned by the context.

def _get_fa4_gather_kv():
    from .ops.flash_sparse_attn.cute.interface import flash_attn_gather_kv_func
    return flash_attn_gather_kv_func

# Deduplicate fallback warnings by category and reason on the hot path.
_FALLBACK_WARNED = set()


def _warn_fallback_once(kind: str, reason: str, impact: str):
    """Warn once per fallback category and reason, including its impact.

    Args:
        kind: Fallback category, used together with reason for deduplication.
        reason: Failed requirement or triggering exception.
        impact: Effect on execution, accuracy, or performance.
    """
    key = (kind, reason)
    if key in _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED.add(key)
    import warnings
    warnings.warn(
        f"[FlashTokenSparse] Fallback ({kind}): {reason}\n"
        f"  Impact: {impact}\n"
        f"  This warning appears once per reason. Resolve the cause above to avoid fallback.",
        stacklevel=3,
    )


def _mean_pool_simple(x, ratio):
    """Mean-pool along the sequence axis.

    Maps (B, H, L, D) to (B, H, L // ratio, D), truncating a partial tail.
    """
    if ratio <= 1:
        return x
    B, H, L, D = x.shape
    L_trunc = (L // ratio) * ratio
    x_trunc = x[:, :, :L_trunc, :]
    return x_trunc.reshape(B, H, L_trunc // ratio, ratio, D).mean(dim=3)

def _compute_token_probs(q, k, lse, BLKQ, scale):
    """Return mean attention probabilities per query block and KV token in BF16."""
    B, H, L_Q, D = q.shape
    M = (L_Q + BLKQ - 1) // BLKQ
    from .ops.triton.block_probs import _compute_token_probs_bf16_triton
    return _compute_token_probs_bf16_triton(q, k, lse, BLKQ, M, scale)

# Chunked online top-K avoids materializing the full (B, H, M, L_K) scores.

_CHUNK_SELECTOR_CHUNK_SIZE = 16384


def _chunked_online_topk(q, k, lse, BLKQ, scale, topk_k, tile_n=128, chunk_size=None):
    """Maintain global top-K indices while scoring KV tokens in chunks.

    Args:
        q: (B, H, L_Q, D) queries.
        k: (B, H, L_K, D) keys.
        lse: (B, H, L_Q) log-sum-exp values.
        BLKQ: Query block size.
        scale: Attention scale.
        topk_k: Final selected token count, already aligned to tile_n.
        tile_n: KV alignment unit.
        chunk_size: KV chunk size; None uses _CHUNK_SELECTOR_CHUNK_SIZE.

    Returns:
        topk_indices: (B, H, M, topk_k), int32. Chunking avoids materializing
        the complete query-block-by-key probability tensor.
    """
    from .ops.triton.block_probs import _token_probs_kernel_bf16

    if chunk_size is None:
        chunk_size = _CHUNK_SELECTOR_CHUNK_SIZE
    B, H, L_Q, D = q.shape
    L_K = k.shape[2]
    M = (L_Q + BLKQ - 1) // BLKQ
    device = q.device

    topk_k_aligned = ((topk_k + tile_n - 1) // tile_n) * tile_n
    topk_k_aligned = min(topk_k_aligned, L_K)

    global_topk_vals = torch.full((B, H, M, topk_k_aligned), -65504.0,
                                  device=device, dtype=torch.bfloat16)
    global_topk_idxs = torch.zeros((B, H, M, topk_k_aligned),
                                   device=device, dtype=torch.int32)

    BH = B * H
    BLOCK_D = 128
    TILE_K = 64

    q_flat = q.reshape(BH, L_Q, D).contiguous()
    lse_flat = lse.reshape(BH, L_Q).contiguous()

    n_chunks = (L_K + chunk_size - 1) // chunk_size

    combined_len = topk_k_aligned + chunk_size
    combined_vals = torch.empty((B, H, M, combined_len), device=device, dtype=torch.bfloat16)
    combined_idxs = torch.empty((B, H, M, combined_len), device=device, dtype=torch.int32)

    for chunk_idx in range(n_chunks):
        k_start = chunk_idx * chunk_size
        k_end = min(k_start + chunk_size, L_K)
        chunk_len = k_end - k_start
        cur_combined_len = topk_k_aligned + chunk_len

        k_chunk = k[:, :, k_start:k_end, :].contiguous()

        k_flat_chunk = k_chunk.reshape(BH, chunk_len, D).contiguous()
        N_tiles_chunk = (chunk_len + TILE_K - 1) // TILE_K
        chunk_probs = torch.zeros(BH, M, chunk_len, device=device, dtype=q.dtype)

        grid = (M, N_tiles_chunk, BH)
        _token_probs_kernel_bf16[grid](
            q_flat, k_flat_chunk, lse_flat, chunk_probs,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            k_flat_chunk.stride(0), k_flat_chunk.stride(1), k_flat_chunk.stride(2),
            lse_flat.stride(0), lse_flat.stride(1),
            chunk_probs.stride(0), chunk_probs.stride(1), chunk_probs.stride(2),
            L_Q, chunk_len, D, M,
            BLKQ, TILE_K, BLOCK_D,
            scale,
        )
        chunk_probs = chunk_probs.reshape(B, H, M, chunk_len)

        # Fill the preallocated merge buffer instead of allocating through cat.
        combined_vals[:, :, :, :topk_k_aligned].copy_(global_topk_vals)
        combined_vals[:, :, :, topk_k_aligned:cur_combined_len].copy_(chunk_probs)

        combined_idxs[:, :, :, :topk_k_aligned].copy_(global_topk_idxs)
        chunk_indices = torch.arange(k_start, k_end, device=device, dtype=torch.int32)
        combined_idxs[:, :, :, topk_k_aligned:cur_combined_len] = chunk_indices

        cur_slice_vals = combined_vals[:, :, :, :cur_combined_len]
        _, topk_pos = torch.topk(cur_slice_vals.float(), topk_k_aligned, dim=-1, sorted=False)
        global_topk_vals.copy_(cur_slice_vals.gather(-1, topk_pos).to(torch.bfloat16))
        global_topk_idxs.copy_(combined_idxs[:, :, :, :cur_combined_len].gather(-1, topk_pos))

    return global_topk_idxs


# Chunk query blocks to bound peak memory. Recall is independent per row,
# so chunking changes memory and launch costs without changing the result.
_ADAPTIVE_RECALL_ELEM_BUDGET = 1 << 26


def _compute_adaptive_topk_token(token_probs, base_topk, layer_t=None):
    """Allocate per-head top-K ratios from retained attention mass.

    Heads with higher recall receive less budget; lower-recall heads receive
    more. Recall is accumulated in query-block chunks without sorting all keys.

    Args:
        token_probs: (B, H, M, L_K_ds) token probabilities.
        base_topk: Base fraction of selected tokens.
        layer_t: Optional normalized layer depth in [0, 1].

    Returns:
        per_head_topk: (B, H) selection fractions.
    """
    B, H, M, L_K_ds = token_probs.shape
    k = max(1, int(base_topk * L_K_ds))
    k = min(k, L_K_ds)

    # Materialize only the current query-block chunk's top-K values.
    m_chunk = _ADAPTIVE_RECALL_ELEM_BUDGET // max(1, B * H * L_K_ds)
    m_chunk = max(1, min(M, m_chunk))

    recall_parts = []
    for m0 in range(0, M, m_chunk):
        chunk = token_probs[:, :, m0:m0 + m_chunk]  # View only; no copy.
        # Only the top-K sum is needed; internal ordering is irrelevant.
        topk_sum = torch.topk(chunk, k, dim=-1, sorted=False).values.sum(dim=-1)
        total_sum = chunk.sum(dim=-1).clamp(min=1e-8)
        recall_parts.append(topk_sum / total_sum)  # (B, H, m_chunk)
    recall_per_head = torch.cat(recall_parts, dim=-1).mean(dim=-1)  # (B, H)

    # Higher-recall heads receive less budget.
    rank_order = recall_per_head.argsort(dim=-1)
    ranks = torch.zeros_like(rank_order)
    ranks.scatter_(-1, rank_order,
                   torch.arange(H, device=token_probs.device).expand(B, H))

    t = ranks.float() / max(H - 1, 1)
    if layer_t is not None:
        per_head_topk = base_topk * (1.0 + 0.5 * layer_t - 0.5 * t)
    else:
        per_head_topk = base_topk * (1.5 - t)
    per_head_topk = per_head_topk.clamp(min=1.0 / L_K_ds, max=1.0)
    return per_head_topk  # (B, H)

    # Higher-recall heads receive less budget.
    rank_order = recall_per_head.argsort(dim=-1)
    ranks = torch.zeros_like(rank_order)
    ranks.scatter_(-1, rank_order,
                   torch.arange(H, device=token_probs.device).expand(B, H))

    t = ranks.float() / max(H - 1, 1)
    if layer_t is not None:
        per_head_topk = base_topk * (1.0 + 0.5 * layer_t - 0.5 * t)
    else:
        per_head_topk = base_topk * (1.5 - t)
    per_head_topk = per_head_topk.clamp(min=1.0 / L_K_ds, max=1.0)
    return per_head_topk  # (B, H)


def _select_topk_tokens(token_probs, topk_ratio, tile_n=128, ds_k=1, orig_L_K=None,
                        selection_mode='topk', topp=None, sink_ranges=None):
    """Select KV tokens independently per head and expand downsampled indices.

    Modes:
        topk: Keep the highest-probability tokens within topk_ratio.
        topp: Keep the probability prefix below topp, with at least one token.
        topk_topp: Use topp with topk_ratio as a lower bound on the count.

    Sink ranges are included for every query in addition to the sparse budget.
    They are excluded from probability ranking. Counts are aligned during
    selection using real tokens, not duplicate padding. Variable-length rows
    are padded only beyond their valid tile count, where the kernel skips them.

    Args:
        token_probs: (B, H, M, L_K_ds) probabilities.
        topk_ratio: Scalar or per-head selection fraction.
        tile_n: KV tile-alignment unit.
        ds_k: Key downsampling ratio.
        orig_L_K: Original key length, required when ds_k > 1.
        selection_mode: 'topk', 'topp', or 'topk_topp'.
        topp: Cumulative probability threshold for topp-based modes.
        sink_ranges: Half-open original-token ranges; None or [] disables them.

    Returns:
        gather_indices: (B, H, M, gather_len), int32 original-key indices.
        gather_kv_lengths: (B, H, M), int32 valid tile counts, or None when
            all query blocks use the same length.
    """
    B, H, M, L_K_ds = token_probs.shape
    device = token_probs.device

    # Always include sink tokens in addition to the sparse budget.
    sink_indices = None  # (n_sink,) or None
    n_sink = 0
    if sink_ranges:
        sink_mask = torch.zeros(L_K_ds, dtype=torch.bool, device=device)
        for start, end in sink_ranges:
            ds_start = start // ds_k if ds_k > 1 else start
            ds_end = (end + ds_k - 1) // ds_k if ds_k > 1 else end
            ds_end = min(ds_end, L_K_ds)
            if ds_start < ds_end:
                sink_mask[ds_start:ds_end] = True
        n_sink = sink_mask.sum().item()
        if n_sink > 0:
            sink_indices = sink_mask.nonzero(as_tuple=True)[0]  # (n_sink,)
            # Exclude sink tokens from probability ranking to avoid double selection.
            token_probs = token_probs.clone()
            token_probs[:, :, :, sink_mask] = 0.0

    # topk_ratio is scalar or (B, H) for adaptive head budgets.
    is_adaptive = isinstance(topk_ratio, Tensor)

    # Choose the downsampled alignment unit so expansion lands on tile_n.
    align_unit_ds = max(1, tile_n // ds_k) if ds_k > 1 else tile_n

    def _align_up(n):
        """Round up to align_unit_ds and clamp to L_K_ds for scalars or tensors.

        Alignment happens during selection, so every valid tile refers to
        selected tokens rather than duplicated padding that changes softmax.
        """
        if isinstance(n, Tensor):
            n = n.clamp(min=1)
            n = ((n + align_unit_ds - 1) // align_unit_ds) * align_unit_ds
            return n.clamp(max=L_K_ds)
        n = max(1, int(n))
        n = ((n + align_unit_ds - 1) // align_unit_ds) * align_unit_ds
        return min(n, L_K_ds)

    if selection_mode == 'topk':
        # Align the budget during selection using real tokens, avoiding duplicate
        # padding that would change the softmax distribution.
        if is_adaptive:
            # Align each adaptive head budget during selection.
            per_head_k_ds = _align_up((topk_ratio * L_K_ds).long())  # (B, H)
            actual_topk_ds = int(per_head_k_ds.max().item())
            # Keep descending order so truncating per-head prefixes preserves top-K.
            selected_indices_ds = torch.topk(token_probs, actual_topk_ds, dim=-1, sorted=True).indices
            per_pos_counts = per_head_k_ds.unsqueeze(-1).expand(B, H, M).long()  # (B, H, M)
        else:
            per_head_k_ds = _align_up(int(topk_ratio * L_K_ds))
            actual_topk_ds = per_head_k_ds
            # (B, H, M, actual_topk_ds)
            selected_indices_ds = torch.topk(token_probs, actual_topk_ds, dim=-1, sorted=False).indices
            per_pos_counts = None

    elif selection_mode == 'topp':
        assert topp is not None, "topp mode requires topp"
        # Probabilities are already normalized. Sort once without estimating a cutoff.
        sorted_probs, sorted_indices = torch.sort(token_probs, dim=-1, descending=True)
        # Accumulate the probability prefix in float32.
        cumsum_before = torch.cumsum(sorted_probs.float(), dim=-1) - sorted_probs.float()
        sorted_mask = cumsum_before < topp
        # Round each selected prefix up with the next real tokens.
        # The mass may slightly exceed topp, but no duplicated padding enters softmax.
        per_pos_counts = _align_up(sorted_mask.sum(dim=-1))  # (B, H, M)
        actual_topk_ds = int(per_pos_counts.max().item())
        selected_indices_ds = sorted_indices[:, :, :, :actual_topk_ds]

    elif selection_mode == 'topk_topp':
        # Use top-K as a minimum budget after the topp selection.
        assert topp is not None, "topk_topp mode requires topp"
        if is_adaptive:
            min_k_ds_t = (topk_ratio * L_K_ds).clamp(min=1, max=L_K_ds).long()  # (B, H)
        else:
            min_k_ds = max(1, int(topk_ratio * L_K_ds))
            min_k_ds = min(min_k_ds, L_K_ds)
        # Probabilities are already normalized; sort once.
        sorted_probs, sorted_indices = torch.sort(token_probs, dim=-1, descending=True)
        cumsum_before = torch.cumsum(sorted_probs.float(), dim=-1) - sorted_probs.float()
        sorted_mask = cumsum_before < topp
        topp_counts = sorted_mask.sum(dim=-1)  # (B, H, M)
        if is_adaptive:
            per_pos_counts = torch.clamp(topp_counts, min=0)
            min_k_expanded = min_k_ds_t.unsqueeze(-1).expand(B, H, M)  # (B, H, M)
            per_pos_counts = torch.max(per_pos_counts, min_k_expanded)  # (B, H, M)
        else:
            per_pos_counts = torch.clamp(topp_counts, min=min_k_ds)  # (B, H, M)
        per_pos_counts = _align_up(per_pos_counts)
        actual_topk_ds = int(per_pos_counts.max().item())
        selected_indices_ds = sorted_indices[:, :, :, :actual_topk_ds]

    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")

    if ds_k > 1:
        # Expand each downsampled key index to its ds_k original token positions.
        assert orig_L_K is not None, "orig_L_K is required when ds_k > 1"
        base = selected_indices_ds.unsqueeze(-1).long() * ds_k
        offsets = torch.arange(ds_k, device=device, dtype=torch.long)
        expanded = (base + offsets).reshape(B, H, M, -1)
        expanded = expanded.clamp(max=orig_L_K - 1)
        actual_topk = expanded.shape[-1]
        if per_pos_counts is not None:
            per_pos_counts = per_pos_counts * ds_k
        if sink_indices is not None:
            sink_base = sink_indices.unsqueeze(-1).long() * ds_k
            sink_expanded = (sink_base + offsets).reshape(-1)
            sink_expanded = sink_expanded[sink_expanded < orig_L_K]
            n_sink = sink_expanded.shape[0]
            sink_indices = sink_expanded
    else:
        expanded = selected_indices_ds.long()
        actual_topk = expanded.shape[-1]

    # Prepend sink indices; they are additional to the sparse budget.
    if sink_indices is not None and n_sink > 0:
        sink_block = sink_indices.long().view(1, 1, 1, -1).expand(B, H, M, -1)
        expanded = torch.cat([sink_block, expanded], dim=-1)
        actual_topk = expanded.shape[-1]
        # An arbitrary sink count can break tile alignment. Realign the total using
        # real trailing candidates from the already-aligned candidate pool.
        if per_pos_counts is not None:
            per_pos_counts = per_pos_counts + n_sink
        else:
            per_pos_counts = torch.full((B, H, M), actual_topk, device=device, dtype=torch.long)
        per_pos_counts = ((per_pos_counts + tile_n - 1) // tile_n) * tile_n
        per_pos_counts = per_pos_counts.clamp(max=actual_topk)

    L_K_orig = orig_L_K if orig_L_K is not None else L_K_ds
    max_n_blocks = (L_K_orig + tile_n - 1) // tile_n  # Maximum aligned KV tile count in the original sequence

    if per_pos_counts is not None:
        # Variable-length rows are already aligned during selection.
        # Clamp to the largest valid whole-tile count without adding unselected slots.
        cap = min(max_n_blocks * tile_n, (expanded.shape[-1] // tile_n) * tile_n)
        cap = max(cap, tile_n)  # At least one tile
        per_pos_counts = per_pos_counts.clamp(min=tile_n, max=cap)
        gather_kv_lengths = (per_pos_counts // tile_n).to(torch.int32)  # (B, H, M)
        gl = int(per_pos_counts.max().item())

        # Zero-padding beyond each row's valid tile count is never read by the kernel.
        if gl > actual_topk:
            pad = torch.zeros(B, H, M, gl - actual_topk, device=device, dtype=expanded.dtype)
            expanded = torch.cat([expanded, pad], dim=-1)
        elif gl < actual_topk:
            expanded = expanded[:, :, :, :gl]

        return expanded.to(torch.int32).contiguous(), gather_kv_lengths.contiguous()
    else:
        # Fixed-length rows already have an aligned selection; no padding is needed.
        gl = min(actual_topk, max_n_blocks * tile_n)
        if actual_topk % tile_n != 0:
            # If expansion breaks alignment, round down to a whole tile rather than
            # duplicate tokens and change the attention distribution.
            gl = max(tile_n, (gl // tile_n) * tile_n)
        if gl < actual_topk:
            expanded = expanded[:, :, :, :gl]

        return expanded.to(torch.int32).contiguous(), None


@torch.no_grad()
def _whiten_q_for_perm(q, k):
    """Transform Q for grouping in the key-derived attention metric.

    The transformed queries are used only to compute the permutation;
    attention still uses the original Q. With centered keys Ks, distances in
    Q @ (Ks.T @ Ks / L)^(1/2) match differences between key-score vectors.

    Compute the Gram matrix in float32, symmetrize it before eigendecomposition,
    and clamp small negative eigenvalues to zero before taking square roots.
    """
    L = q.shape[2]
    kf = k.float()
    ks = kf - kf.mean(dim=2, keepdim=True)                  # Center keys over the sequence axis.
    g = ks.transpose(-1, -2) @ ks / L                       # (B, H, D, D)
    g = 0.5 * (g + g.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(g)
    r = evals.clamp(min=0).sqrt()
    r_mat = (evecs * r.unsqueeze(-2)) @ evecs.transpose(-1, -2)
    return (q.float() @ r_mat).to(q.dtype)


def _apply_perm_bhld(x, perm):
    """Apply a (B*H, L) permutation to a (B, H, L, D) tensor."""
    from .pca_group import apply_perm_bhld
    return apply_perm_bhld(x, perm)


def _invert_perm_bhld(x, perm):
    """Invert the query permutation and restore the original token order."""
    from .pca_group import invert_perm_bhld
    return invert_perm_bhld(x, perm)


# Refresh permutations with selections so query groups and indices stay paired.
# Do not refresh on update: its cached per-query LSE uses the calibration order.
_SELECTION_RECOMPUTE_MODES = ('calibrate', 'full_calibrate', 'approx')


def _ck_vs_dense(tag, layer, step, q, k, v, out):
    """Compare an output with dense attention when dense checks are enabled.

    Reports relative error, cosine similarity, and PSNR. Enable with
    FLASH_TOKEN_SPARSE_CHECK_DENSE=1; this adds a full-attention call per check.
    """
    if os.getenv("FLASH_TOKEN_SPARSE_CHECK_DENSE", "").lower() not in ("1", "true", "yes", "on"):
        return
    with torch.no_grad():
        sc = 1.0 / math.sqrt(q.shape[-1])
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sc).to(torch.float32)
        o = out.float()
        diff = (ref - o)
        rel = (diff.norm() / ref.norm().clamp(min=1e-12)).item()
        cos = torch.nn.functional.cosine_similarity(
            ref.flatten(), o.flatten(), dim=0).item()
        mse = diff.pow(2).mean().item()
        peak = (ref.max() - ref.min()).clamp(min=1e-12).item()
        psnr = 20 * math.log10(peak / math.sqrt(mse)) if mse > 0 else float('inf')
        flag = ""
        if rel > 0.5 or cos < 0.9:
            flag = "  <<< large deviation from dense attention"
        print(f"[TSDENSE] {tag} layer={layer} step={step} | "
              f"rel_err={rel:.4f} cosine={cos:.4f} PSNR={psnr:.2f}dB{flag}", flush=True)


class TokenSparseBackend:
    def __init__(self): self._fn = None
    def _gkv(self):
        if self._fn is None: self._fn = _get_fa4_gather_kv()
        return self._fn
    def forward(self, q, k, v, cache, mode, external_lse=None):
        q,k,v = q.contiguous(), k.contiguous(), v.contiguous()
        if cache.check_seq_len(q.shape[2]): mode = 'calibrate'
        _li = getattr(cache, '_layer_idx', '?')
        _st = cache._step_count
        if _token_sparse_check_enabled():
            print(f"[TSCHECK] forward.enter layer={_li} step={_st} mode={mode} "
                  f"pca={cache.config.pca_reorder} bias={cache.config.bias_correction} "
                  f"B={q.shape[0]} H={q.shape[1]} L={q.shape[2]} D={q.shape[3]} "
                  f"has_cache={cache.has_cache} has_res={cache.has_residual} "
                  f"perm_gen={cache._perm_generation} res_tag={cache._residual_perm_gen}",
                  flush=True)
        if mode == 'full':
            # Full attention does not use gathered indices and needs no query reordering.
            from .ops.full_attn import full_attention as _full_attn
            with _timed("full.attn"):
                return _full_attn(q, k, v, cache.config._full_attn_resolved).to(q.dtype)

        if not cache.config.pca_reorder:
            out = self._dispatch(q, k, v, cache, mode, external_lse)
            _ck("forward.exit(no-pca)", _li, _st, out=out)
            _ck_vs_dense(f"exit mode={mode}", _li, _st, q, k, v, out)
            return out

        # Permute only Q; gather indices still address the original K/V order.
        # Invert the output permutation before returning.
        perm = cache.get_or_make_perm(
            q, recompute=mode in _SELECTION_RECOMPUTE_MODES, k=k)
        # Duplicate or missing permutation entries cannot be inverted correctly.
        if _token_sparse_check_enabled():
            _L = q.shape[2]
            _u = int(perm.unique().numel())
            _ok = (_u == _L and int(perm.min().item()) == 0
                   and int(perm.max().item()) == _L - 1)
            print(f"[TSCHECK] forward.perm layer={_li} step={_st} | "
                  f"perm[{tuple(perm.shape)}] unique={_u}/{_L} "
                  f"min={int(perm.min().item())} max={int(perm.max().item())} "
                  f"{'OK' if _ok else '<<< invalid permutation'}", flush=True)
        with _timed("pca.apply_perm"):
            q_perm = _apply_perm_bhld(q, perm).contiguous()
        out = self._dispatch(q_perm, k, v, cache, mode, external_lse)
        with _timed("pca.invert_perm"):
            out = _invert_perm_bhld(out, perm).contiguous()
        _ck("forward.exit(pca)", _li, _st, out=out)
        _ck_vs_dense(f"exit mode={mode}", _li, _st, q, k, v, out)
        return out

    def _dispatch(self, q, k, v, cache, mode, external_lse=None):
        if mode == 'full_calibrate': return self._full_cal(q,k,v,cache)
        if mode == 'calibrate': return self._cal(q,k,v,cache,external_lse)
        if mode == 'reuse':
            _li = getattr(cache, '_layer_idx', '?')
            _st = cache._step_count
            if cache.config.offload_indices:
                gi, gl = cache._get_prefetched_indices(q.device)
                out = self._sp(q, k, v, gi, gl, cache=cache)
            else:
                gi, gl = cache._cached_indices, cache._cached_lengths
                out = self._sp(q, k, v, gi, gl, cache=cache)
            # Cached indices must stay in range and each valid row needs a nonzero tile count.
            if _token_sparse_check_enabled():
                _lk = k.shape[2]
                _bad = (int((gi >= _lk).sum().item()) + int((gi < 0).sum().item())
                        if gi is not None else -1)
                print(f"[TSCHECK] reuse.indices layer={_li} step={_st} | "
                      f"gi={'None <<< missing indices' if gi is None else f'[{tuple(gi.shape)}] out_of_bounds={_bad}'} "
                      f"L_K={_lk} | "
                      f"gl={'None(fixed length)' if gl is None else f'min={int(gl.min().item())} max={int(gl.max().item())}'}",
                      flush=True)
            _ck("reuse.sparse_out", _li, _st, out=out)
            # Apply a residual only if it matches the current query permutation.
            # If stale, return sparse-only output until the next calibration.
            if cache.config.bias_correction and cache.drop_stale_residual():
                if _token_sparse_debug_enabled():
                    print(f"[FlashTokenSparse] bias_correction: dropped residual "
                          f"(perm changed) layer={getattr(cache, '_layer_idx', '?')} "
                          f"step={cache._step_count}", flush=True)
            if cache.config.bias_correction and cache.has_residual:
                residual = cache.get_residual(out.device)
                # Keep the reduction and synchronizing scalar read inside debug mode.
                if _token_sparse_debug_enabled():
                    res_norm = residual.norm().item()
                    if res_norm < 1e-6:
                        import warnings
                        warnings.warn(
                            f"[FlashTokenSparse] bias_correction: residual norm={res_norm:.2e} "
                            f"≈0, bias may not be effective (layer={getattr(cache, '_layer_idx', '?')})",
                            stacklevel=2,
                        )
                    print(f"[FlashTokenSparse] bias_correction: applied residual "
                          f"norm={res_norm:.6f} layer={getattr(cache, '_layer_idx', '?')}",
                          flush=True)
                # Use an out-of-place sum; do not assume ownership of the sparse output buffer.
                _before = out
                out = out + residual
                _ck("reuse.after_residual", _li, _st,
                    sparse=_before, residual=residual, out=out)
            elif cache.config.bias_correction and not cache.has_residual:
                if _token_sparse_debug_enabled():
                    print(f"[FlashTokenSparse] bias_correction: WARNING residual is None! "
                          f"layer={getattr(cache, '_layer_idx', '?')} step={cache._step_count}",
                          flush=True)
            return out
        if mode == 'update': return self._upd(q,k,v,cache)
        if mode == 'approx': return self._approx(q,k,v,cache)
        raise ValueError(f"bad mode {mode}")
    def _lse(self, q, k):
        """Compute natural-log LSE of shape (B, H, L), including the attention scale.

        This materializes the full score matrix and is intended for the
        downsampled _cal path without bias correction. Bias correction uses
        _full_cal, which obtains output and LSE together from FA3.
        """
        D = q.shape[-1]
        sc = 1.0 / math.sqrt(D)
        return torch.logsumexp(
            torch.matmul(q.float(), k.float().transpose(-1, -2)) * sc,
            dim=-1).float()
    def _full_cal(self, q, k, v, cache):
        """Combine full attention with calibration using one FA3 output/LSE pass.

        Used without downsampling when full and calibration steps overlap.
        Cache token selections and, when enabled, the dense-minus-sparse
        residual. Return the full-attention output.
        """
        cfg = cache.config
        B, H, L, D = q.shape
        sc = 1.0 / math.sqrt(D)
        orig_L_K = k.shape[2]

        # FA3 returns output and LSE together. Calibration has no SDPA fallback.
        from .ops.full_attn import full_attention_with_lse
        with _timed("cal.full_attn"):
            output, lse = full_attention_with_lse(q, k, v)

        # Check the LSE shape before passing it to token-probability scoring.
        assert lse.shape == (B, H, L), (
            f"_full_cal: expected LSE shape {(B, H, L)}, got {tuple(lse.shape)}. "
            f"The full-attention kernel may not have returned LSE, causing incorrect unpacking."
        )
        assert output.shape == q.shape, (
            f"_full_cal: expected full-attention output shape {tuple(q.shape)}, "
            f"got {tuple(output.shape)}"
        )

        _li = getattr(cache, '_layer_idx', '?')
        _st = cache._step_count
        _ck("full_cal.in", _li, _st, q=q, k=k, v=v)
        _ck("full_cal.fullattn", _li, _st, output=output, lse=lse)

        # Check use_chunk_selector first, matching _cal. The chunked path uses a fixed
        # budget and bypasses _resolve_topk, so adaptive_heads does not apply there.
        if cfg.use_chunk_selector:
            topk_k = max(1, int(cfg.topk * orig_L_K))
            topk_k = ((topk_k + cfg.tile_n - 1) // cfg.tile_n) * cfg.tile_n
            topk_k = min(topk_k, orig_L_K)
            with _timed("cal.selection"):
                gi = _chunked_online_topk(q, k, lse, cfg.BLKQ, sc, topk_k,
                                          tile_n=cfg.tile_n,
                                          chunk_size=cfg.chunk_selector_size)
            gl = None
        else:
            with _timed("cal.token_probs"):
                tp = _compute_token_probs(q, k, lse, cfg.BLKQ, sc)
            _ck("full_cal.token_probs", _li, _st, tp=tp)
            topk_ratio = self._resolve_topk(tp, cache)
            _ck("full_cal.topk_ratio", _li, _st, topk_ratio=topk_ratio)
            gi, gl = _select_topk_tokens(tp, topk_ratio, cfg.tile_n, ds_k=1, orig_L_K=orig_L_K,
                                         selection_mode=cfg.selection_mode, topp=cfg.topp,
                                         sink_ranges=cfg.sink_ranges)
            # Release token probabilities before attention and residual computation.
            del tp
        if _token_sparse_check_enabled():
            _bad = int((gi >= orig_L_K).sum().item()) + int((gi < 0).sum().item())
            print(f"[TSCHECK] full_cal.select layer={_li} step={_st} "
                  f"chunk={cfg.use_chunk_selector} | "
                  f"gi[{tuple(gi.shape)}] max={int(gi.max().item())} "
                  f"min={int(gi.min().item())} orig_L_K={orig_L_K} "
                  f"out_of_bounds={_bad}{'  <<< index out of bounds' if _bad else ''} | "
                  f"gl={'None(fixed length)' if gl is None else f'[{tuple(gl.shape)}] min={int(gl.min().item())} max={int(gl.max().item())}'}"
                  f"{'  <<< gl contains a zero tile count' if gl is not None and int(gl.min().item()) == 0 else ''}",
                  flush=True)

        _debug_token_sparse_stats(gi, gl, cfg, orig_L_K)
        cache._cached_lse = lse
        if cache.config.offload_indices:
            cache.offload_to_cpu(gi, gl, gi.device)
            cache._cached_indices = None; cache._cached_lengths = None
        else:
            cache._cached_indices = gi; cache._cached_lengths = gl

        if cfg.bias_correction:
            sparse_out = self._sp(q, k, v, gi, gl, cache=cache)
            # Compute the difference before casting; using out= would force the dtype
            # of the sparse output buffer.
            _residual = (output - sparse_out).to(q.dtype)
            _ck("full_cal.residual", _li, _st,
                output=output, sparse_out=sparse_out, residual=_residual)
            cache._store_residual(_residual)
            if _token_sparse_debug_enabled():
                residual = cache.get_residual(q.device)
                r_norm = residual.norm().item()
                print(f"[FlashTokenSparse] bias_correction: cached residual norm={r_norm:.6f} "
                      f"layer={cache._layer_idx}", flush=True)

        return output

    def _resolve_topk(self, tp, cache):
        """Resolve a scalar or (B, H) top-K ratio from head and layer adaptation."""
        cfg = cache.config
        H = tp.shape[1]
        has_layer = cfg.adaptive_layers and cache.layer_t is not None
        has_head = cfg.adaptive_heads and H > 1
        if has_head:
            return _compute_adaptive_topk_token(
                tp, cfg.topk, layer_t=cache.layer_t if has_layer else None
            )
        elif has_layer:
            return cfg.topk * (0.5 + cache.layer_t)
        else:
            return cfg.topk

    def _cal(self, q, k, v, cache, ext_lse):
        cfg = cache.config; sc = 1.0/math.sqrt(q.shape[-1])
        ds_q, ds_k = cfg._cal_ds_q, cfg._cal_ds_k
        orig_L_Q = q.shape[2]
        orig_L_K = k.shape[2]

        # Bias correction shares the full output/LSE calibration path.
        if cfg.bias_correction:
            return self._full_cal(q, k, v, cache)

        q_cal = _mean_pool_simple(q, ds_q) if ds_q > 1 else q
        k_cal_lse = _mean_pool_simple(k, ds_q) if ds_q > 1 else k  # LSE keys use the query downsampling ratio.
        lse = ext_lse if ext_lse is not None else self._lse(q_cal, k_cal_lse)
        k_cal_tp = _mean_pool_simple(k, ds_k) if ds_k > 1 else k

        if cfg.use_chunk_selector:
            L_K_ds = k_cal_tp.shape[2]
            topk_k = max(1, int(cfg.topk * L_K_ds))
            topk_k = ((topk_k + cfg.tile_n - 1) // cfg.tile_n) * cfg.tile_n
            topk_k = min(topk_k, L_K_ds)
            with _timed("cal.selection"):
                gi = _chunked_online_topk(q_cal, k_cal_tp, lse, cfg.BLKQ, sc, topk_k, tile_n=cfg.tile_n, chunk_size=cfg.chunk_selector_size)
            if ds_k > 1:
                base = gi.unsqueeze(-1).long() * ds_k
                offsets = torch.arange(ds_k, device=gi.device, dtype=torch.long)
                gi = (base + offsets).reshape(gi.shape[0], gi.shape[1], gi.shape[2], -1)
                gi = gi.clamp(max=orig_L_K - 1).to(torch.int32).contiguous()
            gl = None
        else:
            # Keep BLKQ unchanged; downsampling reduces the query-block count instead.
            with _timed("cal.token_probs"):
                tp = _compute_token_probs(q_cal, k_cal_tp, lse, cfg.BLKQ, sc)
            topk_ratio = self._resolve_topk(tp, cache)
            gi, gl = _select_topk_tokens(tp, topk_ratio, cfg.tile_n, ds_k=ds_k, orig_L_K=orig_L_K,
                                     selection_mode=cfg.selection_mode, topp=cfg.topp,
                                     sink_ranges=cfg.sink_ranges)
            # Release selection probabilities before launching sparse attention.
            del tp

        # Expand query-block indices from M_ds back to the original block count.
        if ds_q > 1:
            gi, gl = self._upsample_indices(gi, gl, orig_L_Q, cfg.BLKQ, ds_q)
        _debug_token_sparse_stats(gi, gl, cfg, orig_L_K)
        cache._cached_lse = lse
        # Offload indices asynchronously alongside the sparse kernel.
        if cache.config.offload_indices:
            cache.offload_to_cpu(gi, gl, gi.device)
            cache._cached_indices = None; cache._cached_lengths = None
        else:
            cache._cached_indices = gi; cache._cached_lengths = gl
        return self._sp(q, k, v, gi, gl, cache=cache)
    def _upd(self, q, k, v, cache):
        cfg = cache.config; sc = 1.0/math.sqrt(q.shape[-1])
        ds_q, ds_k = cfg._cal_ds_q, cfg._cal_ds_k
        orig_L_Q = q.shape[2]
        orig_L_K = k.shape[2]
        q_cal = _mean_pool_simple(q, ds_q) if ds_q > 1 else q
        k_cal_tp = _mean_pool_simple(k, ds_k) if ds_k > 1 else k

        if cfg.use_chunk_selector:
            L_K_ds = k_cal_tp.shape[2]
            topk_k = max(1, int(cfg.topk * L_K_ds))
            topk_k = ((topk_k + cfg.tile_n - 1) // cfg.tile_n) * cfg.tile_n
            topk_k = min(topk_k, L_K_ds)
            with _timed("upd.selection"):
                gi = _chunked_online_topk(q_cal, k_cal_tp, cache._cached_lse, cfg.BLKQ, sc, topk_k, tile_n=cfg.tile_n, chunk_size=cfg.chunk_selector_size)
            if ds_k > 1:
                base = gi.unsqueeze(-1).long() * ds_k
                offsets = torch.arange(ds_k, device=gi.device, dtype=torch.long)
                gi = (base + offsets).reshape(gi.shape[0], gi.shape[1], gi.shape[2], -1)
                gi = gi.clamp(max=orig_L_K - 1).to(torch.int32).contiguous()
            gl = None
        else:
            # Keep BLKQ unchanged; downsampling reduces the query-block count instead.
            with _timed("upd.token_probs"):
                tp = _compute_token_probs(q_cal, k_cal_tp, cache._cached_lse, cfg.BLKQ, sc)
            topk_ratio = self._resolve_topk(tp, cache)
            gi, gl = _select_topk_tokens(tp, topk_ratio, cfg.tile_n, ds_k=ds_k, orig_L_K=orig_L_K,
                                     selection_mode=cfg.selection_mode, topp=cfg.topp,
                                     sink_ranges=cfg.sink_ranges)
            # Release selection probabilities before attention and residual computation.
            del tp

        # Expand query-block indices to the original block count.
        if ds_q > 1:
            gi, gl = self._upsample_indices(gi, gl, orig_L_Q, cfg.BLKQ, ds_q)
        _debug_token_sparse_stats(gi, gl, cfg, orig_L_K)
        # Offload indices asynchronously alongside the sparse kernel.
        if cache.config.offload_indices:
            cache.offload_to_cpu(gi, gl, gi.device)
            cache._cached_indices = None; cache._cached_lengths = None
        else:
            cache._cached_indices = gi; cache._cached_lengths = gl
        sparse_out = self._sp(q, k, v, gi, gl, cache=cache)
        # Cache the dense-minus-sparse residual and return the full output.
        if cfg.bias_correction:
            sc_full = 1.0 / math.sqrt(q.shape[-1])
            full_out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, scale=sc_full).to(q.dtype)
            # Do not reuse the sparse buffer through out=; cast after subtraction.
            cache._store_residual((full_out - sparse_out).to(q.dtype))
            return full_out
        return sparse_out

    def _approx(self, q, k, v, cache, chunk_size=16384):
        """Select tokens from mean-pooled query scores without cached LSE.

        Pool each query block, score key chunks, merge into the global top-K,
        run attention using the gathered indices, and cache them for reuse.
        chunk_size bounds the temporary query-block-by-key score tensor.
        """
        cfg = cache.config
        B, H, L, D = q.shape
        BLKQ = cfg.BLKQ
        tile_n = cfg.tile_n
        L_K = k.shape[2]

        M = (L + BLKQ - 1) // BLKQ
        L_pad = M * BLKQ
        if L_pad > L:
            q_padded = torch.nn.functional.pad(q, (0, 0, 0, L_pad - L))
        else:
            q_padded = q
        q_mean = q_padded.reshape(B, H, M, BLKQ, D).mean(dim=3)  # (B, H, M, D)

        topk_k = max(1, int(cfg.topk * L_K))
        topk_k = ((topk_k + tile_n - 1) // tile_n) * tile_n
        topk_k = min(topk_k, L_K)

        device = q.device
        n_chunks = (L_K + chunk_size - 1) // chunk_size

        global_topk_vals = torch.full((B, H, M, topk_k), float('-inf'),
                                      device=device, dtype=torch.float32)
        global_topk_idxs = torch.zeros((B, H, M, topk_k),
                                       device=device, dtype=torch.int32)

        # Reuse the merge buffer to avoid allocating a new concatenation each chunk.
        max_chunk_len = min(chunk_size, L_K)
        combined_len = topk_k + max_chunk_len
        combined_vals = torch.empty((B, H, M, combined_len), device=device, dtype=torch.float32)
        combined_idxs = torch.empty((B, H, M, combined_len), device=device, dtype=torch.int32)

        for chunk_idx in range(n_chunks):
            k_start = chunk_idx * chunk_size
            k_end = min(k_start + chunk_size, L_K)
            chunk_len = k_end - k_start
            cur_combined_len = topk_k + chunk_len

            k_chunk = k[:, :, k_start:k_end, :]  # (B, H, chunk_len, D)
            chunk_scores = torch.matmul(q_mean, k_chunk.transpose(-1, -2))  # (B, H, M, chunk_len)

            combined_vals[:, :, :, :topk_k].copy_(global_topk_vals)
            combined_vals[:, :, :, topk_k:cur_combined_len].copy_(chunk_scores.float())

            combined_idxs[:, :, :, :topk_k].copy_(global_topk_idxs)
            chunk_indices = torch.arange(k_start, k_end, device=device, dtype=torch.int32)
            combined_idxs[:, :, :, topk_k:cur_combined_len] = chunk_indices

            cur_slice = combined_vals[:, :, :, :cur_combined_len]
            _, topk_pos = torch.topk(cur_slice, topk_k, dim=-1, sorted=False)
            global_topk_vals.copy_(cur_slice.gather(-1, topk_pos))
            global_topk_idxs.copy_(combined_idxs[:, :, :, :cur_combined_len].gather(-1, topk_pos))

        # Sort selected indices as required by the gather kernel.
        gi, _ = torch.sort(global_topk_idxs, dim=-1)
        gi = gi.to(torch.int32).contiguous()

        return self._sp(q, k, v, gi, None, cache=cache)

    @staticmethod
    def _upsample_indices(gi, gl, orig_L_Q, BLKQ, ds_q):
        """Expand downsampled query-block indices to the original block count.

        Each original block i uses downsampled block i // ds_q.

        Args:
            gi: (B, H, M_ds, gather_len), int32 indices.
            gl: (B, H, M_ds), int32 tile counts, or None.
            orig_L_Q: Original query length.
            BLKQ: Original query block size.
            ds_q: Query downsampling ratio.

        Returns:
            (gi_up, gl_up), expanded and truncated to the original block count.
        """
        M_orig = (orig_L_Q + BLKQ - 1) // BLKQ
        M_ds = gi.shape[2]
        if M_ds == M_orig:
            return gi, gl
        # Original query block i uses downsampled block i // ds_q.
        # Repeat and truncate to the original query-block count.
        gi_up = gi.repeat_interleave(ds_q, dim=2)[:, :, :M_orig, :]
        gl_up = gl.repeat_interleave(ds_q, dim=2)[:, :, :M_orig] if gl is not None else None
        return gi_up.contiguous(), gl_up.contiguous() if gl_up is not None else None

    def _sp(self, q, k, v, gi, gl=None, cache=None):
        """Run token-sparse attention, converting BHLD inputs to BLHD for the kernel.

        gi: (B, H, M, gather_len), int32 indices.
        gl: (B, H, M), int32 per-block KV tile counts, or None.
        cache: Layer and step metadata used by diagnostics.
        """
        B,H,L,D = q.shape
        # Convert BHLD to BLHD and time layout copies separately from the kernel.
        with _timed("sparse.layout"):
            q_bshd = q.transpose(1, 2).contiguous()
            k_bshd = k.transpose(1, 2).contiguous()
            v_bshd = v.transpose(1, 2).contiguous()
        with _timed("sparse.kernel"):
            o, _ = self._gkv()(q_bshd, k_bshd, v_bshd, gi,
                               softmax_scale=1.0/math.sqrt(D), return_lse=False,
                               gather_kv_lengths=gl)
        # Check raw kernel output before any residual correction is applied.
        if _token_sparse_check_enabled():
            _li = getattr(cache, '_layer_idx', '?') if cache is not None else '?'
            _st = cache._step_count if cache is not None else '?'
            _ck("_sp.kernel_out", _li, _st, o=o)
        # (B, L, H, D) -> (B, H, L, D)
        return o.transpose(1, 2).contiguous()

class TokenSparseContext:
    """Manage token-sparse attention state and cache lifetimes across layers."""

    def __init__(self, config):
        self.config = config
        self._be = TokenSparseBackend()
        self._fl = set(config.full_layers)
        self._cfl = set(self._fl)
        # Serial CFG passes use separate cache slots despite sharing layer indices.
        self._cache_slots = {"_default": {}}  # slot_name -> {layer_idx: TokenSparseCache}
        self._active_slot = "_default"
        self._caches = self._cache_slots["_default"]  # Active cache slot
        self._nl = None
        self._sc = 0
        self._slse = None
        self._slse_per_slot = {}  # slot_name -> Optional[Tensor]
        self._pm = []
        self._current_topk = None
        self._current_selection_mode = None
        # Release inactive transformer caches in multi-transformer execution.
        self._forward_called_this_step = False
        # Release a sibling context when this context becomes active.
        self._siblings = []
        self._density_records = []
        self._latest_density = {}
        # Share converted rotary embeddings across layers in the same step.
        self._cos_sin_cache = None

    def step(self):
        """Advance the step counter after each denoising step."""
        # An unused transformer can release its cached residuals for this step.
        if not self._forward_called_this_step:
            self.release_residuals()
        self._forward_called_this_step = False

        self._sc += 1
        self._cfl = set(self._fl)
        self._slse = None
        self._slse_per_slot.clear()
        self._current_topk = None
        self._current_selection_mode = None
        for slot_caches in self._cache_slots.values():
            for c in slot_caches.values():
                c._step_count = self._sc
        if _token_sparse_debug_enabled():
            print(f"[FlashTokenSparse] step -> {self._sc}", flush=True)
        # Prefetch the first sparse layer's indices for the next step.
        if self.config.offload_indices:
            self._prefetch_first_sparse_layer()
        # Prefetch its residual too, avoiding a synchronous first-layer transfer.
        if self._residual_offload_enabled():
            self._prefetch_first_residual_layer()

    def _prefetch_first_residual_layer(self):
        """Prefetch the first sparse layer residual after advancing the step."""
        if self._nl is None:
            return
        for li in range(self._nl):
            if li in self._cfl:  # Dense layers need no sparse prefetch.
                continue
            c = self._caches.get(li)
            if c is None:
                continue
            if c.mode == 'reuse':
                # CPU tensor metadata cannot identify the destination GPU; use the active device.
                c.prefetch_residual(torch.device('cuda'))
            break

    def reset(self):
        """Reset all cached state before processing a new sample."""
        self._slse = None
        self._slse_per_slot.clear()
        self._cfl = set(self._fl)
        self._sc = 0
        self._current_topk = None
        self._current_selection_mode = None
        self._forward_called_this_step = False
        self._density_records.clear()
        self._latest_density.clear()
        # Clear rotary caches between samples: shapes and Python object IDs may change.
        if self._cos_sin_cache is not None:
            self._cos_sin_cache.clear()
        for slot_caches in self._cache_slots.values():
            for c in slot_caches.values():
                c.reset()
        self._active_slot = "_default"
        self._caches = self._cache_slots["_default"]
        if _token_sparse_debug_enabled():
            total = sum(len(s) for s in self._cache_slots.values())
            print(f"[FlashTokenSparse] reset slots={len(self._cache_slots)} caches={total}", flush=True)

    def release_residuals(self):
        """Release GPU indices, LSE, and residuals for every layer and cache slot.

        Inactive transformers can release memory this way. A later forward
        recalibrates automatically when the required cache is absent.
        """
        for slot_caches in self._cache_slots.values():
            for c in slot_caches.values():
                c._cached_lse = None
                c._cached_indices = None
                c._cached_lengths = None
                c._cached_perm = None
                # Wait for pending D2H before releasing the source GPU residual.
                if c._residual_offload_event is not None:
                    c._residual_offload_event.synchronize()
                    c._residual_offload_event = None
                c._cached_residual = None
                c._prefetched_residual = None
                c._residual_prefetch_event = None
                c._residual_perm_gen = None
                c._indices_cpu = None
                c._lengths_cpu = None
                c._prefetched_indices = None
                c._prefetched_lengths = None
                # Release pinned CPU copies too; they cannot be paged out by the OS.
                c._cached_residual_cpu = None
        self._slse_per_slot.clear()

    @contextmanager
    def cache_context(self, name: str):
        """Activate a named cache slot for serial classifier-free-guidance passes.

        Separate conditional and unconditional indices, LSE, and residuals so
        one calibration does not overwrite the other. Restore the previous
        slot on exit. name identifies the slot, such as 'cond' or 'uncond'.
        """
        prev_slot = self._active_slot
        prev_caches = self._caches
        prev_slse = self._slse

        self._slse_per_slot[prev_slot] = prev_slse

        if name not in self._cache_slots:
            self._cache_slots[name] = {}
        self._active_slot = name
        self._caches = self._cache_slots[name]
        self._slse = self._slse_per_slot.get(name, None)

        if _token_sparse_debug_enabled():
            print(f"[FlashTokenSparse] cache_context enter slot={name} "
                  f"caches={len(self._caches)}", flush=True)
        try:
            yield
        finally:
            self._slse_per_slot[name] = self._slse
            self._active_slot = prev_slot
            self._caches = prev_caches
            self._slse = prev_slse
            if _token_sparse_debug_enabled():
                print(f"[FlashTokenSparse] cache_context exit slot={name} "
                      f"restored_slot={prev_slot}", flush=True)

    def _hook_cache_context(self, model):
        """Hook a model's cache_context to switch cache slots for serial CFG passes.

        Models without cache_context are left unchanged.
        """
        original_cache_context = getattr(model, 'cache_context', None)
        if original_cache_context is None:
            return

        ctx = self

        @contextmanager
        def patched_cache_context(name: str):
            with original_cache_context(name):
                with ctx.cache_context(name):
                    yield

        model.cache_context = patched_cache_context
        if not hasattr(self, '_hooked_cache_contexts'):
            self._hooked_cache_contexts = []
        self._hooked_cache_contexts.append((model, 'cache_context', original_cache_context))

    def set_step_override(self, **overrides):
        """Override full_layers, topk, or selection_mode for the current step.

        Overrides are cleared when the step advances.
        """
        if 'full_layers' in overrides:
            self._cfl = set(overrides['full_layers'])
        if 'topk' in overrides:
            self._current_topk = overrides['topk']
        if 'selection_mode' in overrides:
            self._current_selection_mode = overrides['selection_mode']

    def is_full(self, li):
        """Return whether this layer uses full attention at the current step."""
        if getattr(self.config, '_dense_only', False):
            return True
        if self.config._full_set and self._sc in self.config._full_set:
            return True
        return li in self._cfl

    def is_full_step(self):
        """Return whether the current step uses full attention globally."""
        return self.config._full_set is not None and self._sc in self.config._full_set

    def _should_forward_full_step_to_sparse_path(self):
        """Route overlapping full/calibration steps without downsampling to calibration."""
        return (
            self.config._calibrate_set is not None
            and self._sc in self.config._calibrate_set
            and self.is_full_step()
            and self.config._cal_ds_q == 1
            and self.config._cal_ds_k == 1
        )

    def _gc(self, li):
        if li not in self._caches:
            c = TokenSparseCache(self.config); c._step_count = self._sc
            if self.config.adaptive_layers and self._nl is not None and self._nl > 1:
                c.layer_t = li / (self._nl - 1)
            c._layer_idx = li
            self._caches[li] = c
        return self._caches[li]

    def forward(self, li, q, k, v):
        if not self._forward_called_this_step:
            self._forward_called_this_step = True
            for sibling in self._siblings:
                sibling.release_residuals()
        c = self._gc(li); m = c.mode
        if _token_sparse_debug_enabled():
            print(f"[FlashTokenSparse] forward step={self._sc} slot={self._active_slot} "
                  f"layer={li} mode={m} "
                  f"q={tuple(q.shape)} has_cache={c.has_cache} has_lse={c.has_lse}", flush=True)

        # Start the next layer's prefetch before this layer computes, so H2D can
        # overlap the current sparse-attention kernel.
        if self.config.offload_indices:
            self._prefetch_next_sparse_layer(li, q.device)
        if self._residual_offload_enabled():
            self._prefetch_next_residual_layers(li, q.device)
        elif _token_sparse_debug_enabled() and self.config.bias_correction:
            # GPU-resident residuals need no H2D prefetch when cache_offload is disabled.
            print(f"[FlashTokenSparse] residual_prefetch disabled: "
                  f"cache_offload={self.config.cache_offload!r} "
                  f"(must be 'residual' for H2D prefetch)", flush=True)

        o = self._be.forward(q, k, v, c, m,
                             external_lse=self._slse if self.config.share_lse else None)
        if self.config.share_lse and m in ('calibrate', 'full_calibrate') and c.has_lse and self._slse is None:
            self._slse = c._cached_lse
        self._record_density(li, m, c)
        return o

    def _residual_offload_enabled(self):
        return (self.config.cache_offload == 'residual'
                and self.config.bias_correction
                and torch.cuda.is_available())

    def _prefetch_next_residual_layers(self, current_li, device):
        """Prefetch residuals for the next offload_prefetch_ahead sparse layers."""
        dbg = _token_sparse_debug_enabled()
        if self._nl is None or device.type != 'cuda':
            if dbg:
                print(f"[FlashTokenSparse] residual_prefetch skip: "
                      f"_nl={self._nl} device={device.type} (_nl=None means the automatic "
                      f"patch_* setup was not used; the layer count and next layer are unknown)", flush=True)
            return
        ahead = getattr(self.config, 'offload_prefetch_ahead', 1)
        if ahead <= 0:
            if dbg:
                print(f"[FlashTokenSparse] residual_prefetch skip: ahead={ahead}",
                      flush=True)
            return
        found = 0
        for next_li in range(current_li + 1, self._nl):
            if next_li in self._cfl:  # Dense layers need no residual.
                continue
            next_c = self._caches.get(next_li)
            if next_c is None:
                continue
            if next_c.mode == 'reuse':
                next_c.prefetch_residual(device)
            elif dbg:
                print(f"[FlashTokenSparse] residual_prefetch skip: "
                      f"layer={next_li} mode={next_c.mode} (not reuse)", flush=True)
            found += 1
            if found >= ahead:
                break
        if found == 0 and dbg:
            print(f"[FlashTokenSparse] residual_prefetch skip: "
                  f"No sparse layer to prefetch after layer={current_li}", flush=True)

    def _prefetch_next_sparse_layer(self, current_li, device):
        """Prefetch indices for the next sparse layer that will reuse its cache."""
        if self._nl is None:
            return
        for next_li in range(current_li + 1, self._nl):
            if next_li in self._cfl:  # Skip dense layers.
                continue
            next_c = self._caches.get(next_li)
            if next_c is None:
                continue
            if next_c.mode == 'reuse' and next_c._indices_cpu is not None:
                next_c.prefetch(device)
            break  # Prefetch only the immediately following sparse layer.

    def _prefetch_first_sparse_layer(self):
        """Prefetch indices for the first sparse layer after advancing the step."""
        if self._nl is None:
            return
        for li in range(self._nl):
            if li in self._cfl:  # Skip dense layers.
                continue
            c = self._caches.get(li)
            if c is None:
                continue
            if c.mode == 'reuse' and c._indices_cpu is not None:
                # CPU index tensors do not identify the destination GPU; use the current device.
                device = torch.device('cuda')
                c.prefetch(device)
            break

    def _record_density(self, layer_idx, mode, cache):
        """Record the realized density of the most recent sparse forward for a layer."""
        gi = cache._cached_indices if cache._cached_indices is not None else cache._indices_cpu
        if gi is None:
            return
        B, H, M, gather_len = gi.shape
        gl = cache._cached_lengths if cache._cached_lengths is not None else cache._lengths_cpu
        if gl is not None:
            mean_selected = gl.float().mean().item()
        else:
            mean_selected = float(gather_len)
        orig_L_K = cache._last_orig_L_K if hasattr(cache, '_last_orig_L_K') else gather_len
        density = mean_selected / max(orig_L_K, 1)
        record = {
            'step': self._sc,
            'layer': layer_idx,
            'mode': mode,
            'density': density,
            'mean_selected_tokens': mean_selected,
            'total_tokens': orig_L_K,
        }
        self._density_records.append(record)
        self._latest_density[layer_idx] = record

    @property
    def latest_density(self):
        """Return the most recent sparse-forward density for each layer."""
        return dict(self._latest_density)

    def get_density(self, *, latest=False, reset=False):
        """Return aggregate realized attention-density statistics."""
        records = dict(self._latest_density) if latest else list(self._density_records)
        if reset:
            self._density_records.clear()
            self._latest_density.clear()
        return records

    def __getitem__(self, li):
        def _call(q, k, v):
            return self.forward(li, q, k, v)
        return _call

    @property
    def num_layers(self):
        return self._nl

    @property
    def full_layers(self):
        return sorted(self._fl)

    @full_layers.setter
    def full_layers(self, layers):
        self._fl = set(layers)


    def unpatch(self):
        """Restore the original attention and cache_context methods."""
        for b, a, o in self._pm:
            if a == "processor" and hasattr(b, "set_processor"):
                b.set_processor(o)
            else:
                setattr(b, a, o)
        self._pm.clear()
        # Restore cache_context as well to avoid stacking hooks on repeated patching.
        if hasattr(self, '_hooked_cache_contexts'):
            for obj, attr, original in self._hooked_cache_contexts:
                setattr(obj, attr, original)
            self._hooked_cache_contexts.clear()


__all__ = [
    "TokenSparseConfig",
    "TokenSparseContext",
    "TokenSparseCache",
]

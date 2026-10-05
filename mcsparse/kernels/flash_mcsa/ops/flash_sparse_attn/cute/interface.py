import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32

import mcsparse.kernels.flash_mcsa.ops.flash_sparse_attn.cute.utils as utils
from mcsparse.kernels.flash_mcsa.ops.flash_sparse_attn.cute.cute_dsl_utils import to_cute_tensor
from mcsparse.kernels.flash_mcsa.ops.flash_sparse_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90
from mcsparse.kernels.flash_mcsa.ops.flash_sparse_attn.cute.flash_fwd_sm90_fp8 import (
    FlashAttentionForwardSm90Fp8,
)
from mcsparse.kernels.flash_mcsa.ops.flash_sparse_attn.cute.utils import AuxData


@dataclass(frozen=True)
class FwdConfig:
    m_block_size: int
    n_block_size: int
    mma_pv_is_rs: bool
    intra_wg_overlap: bool


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}


def maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


# Validating gather_kv_lengths bounds needs .item(), i.e. a device sync on every
# call. Off by default; set FLASH_MCSA_FP8_CHECK_LENGTHS=1 when debugging.
_FP8_CHECK_LENGTHS = os.environ.get("FLASH_MCSA_FP8_CHECK_LENGTHS", "0") == "1"


@lru_cache(None)
def _tile_size_fwd_sm90(
    head_dim: int,
    head_dim_v: int,
) -> FwdConfig:
    hdim_multiple_of = 16
    head_dim_padded = math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of
    head_dim_v_padded = math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of
    if head_dim_padded <= 64:
        return FwdConfig(128, 128, False, False)
    if head_dim_padded == 96:
        return FwdConfig(128, 128, False, False)
    if head_dim_padded == 128:
        return FwdConfig(128, 128, head_dim_v_padded <= 128, False)
    if head_dim_padded == 160:
        # head_dim 144 runs here: 144 satisfies neither the fp8 MMA-K
        # granularity (32) nor the smem swizzle atoms (d % 32), so callers
        # zero-pad 144 -> 160.
        # mma_pv_is_rs=True is required, not just preferred: at 160 the bf16
        # smem budget is Q 40K + (K+V) 2 stages 160K = 200K, and the SS path's
        # extra sP (tile_m*tile_n*2 = 32K) would total 232K, over the 227K
        # limit. RS keeps P in registers. num_stages must stay 2 for the same
        # reason (the call site below hardcodes it); the fp8 entry point runs
        # 3 stages because fp8 halves every per-stage buffer.
        return FwdConfig(128, 128, True, False)
    if head_dim_padded == 192:
        return FwdConfig(128, 128, False, False)
    if head_dim_padded == 256:
        return FwdConfig(128, 64, False, False)
    raise ValueError(f"Unsupported head_dim={head_dim} for minimal SM90 gather-KV forward")


def _get_device_arch() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _validate_head_dims(head_dim: int, head_dim_v: int, alignment: int) -> None:
    if head_dim % alignment != 0:
        raise ValueError(f"head_dim must be a multiple of {alignment}, got {head_dim}")
    if head_dim_v % alignment != 0:
        raise ValueError(f"head_dim_v must be a multiple of {alignment}, got {head_dim_v}")
    if head_dim > 256:
        raise ValueError(f"head_dim > 256 is unsupported, got {head_dim}")
    if head_dim_v > 512:
        raise ValueError(f"head_dim_v > 512 is unsupported, got {head_dim_v}")


def _validate_tensor(
    tensor: torch.Tensor,
    name: str,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor.shape != shape:
        raise ValueError(f"{name}.shape must be {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name}.dtype must be {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name}.device must be {device}, got {tensor.device}")


def flash_attn_gather_kv_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gather_kv_indices: torch.Tensor,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    tile_mn: Optional[Tuple[int, int]] = None,
    num_threads: int = 384,
    gather_kv_lengths: Optional[torch.Tensor] = None,
):
    """Forward-only SM90 token-level sparse gather-KV attention.

    Supported subset:
    - Hopper/SM90 only.
    - MHA layout: q=(B, Sq, H, D), k=(B, Sk, H, D), v=(B, Sk, H, Dv).
    - Non-causal, non-local; no varlen, paged KV, or backward pass.
    - gather_kv_indices: (B, H, ceil(Sq / tile_m), gather_len), int32,
      contiguous in the last dimension. gather_len must be divisible by
      tile_n, which is currently fixed at 128.
    - Optional gather_kv_lengths: (B, H, ceil(Sq / tile_m)), int32.
      Each entry counts valid KV tiles for a query block. Token counts must
      already be tile-aligned; iterations beyond the count are skipped.
      None uses gather_len / tile_n iterations for every query block.
    - Head dimension 144 is zero-padded to 160 for shared-memory swizzling;
      the output is sliced back to 144.
    """
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    if q.ndim == 4 and q.shape[-1] == 144:
        # Zero-padding adds no QK contribution; slice padded output channels away.
        # Compute softmax_scale from the original head dimension before padding.
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        qp, kp, vp = [torch.nn.functional.pad(t, (0, 16)) for t in (q, k, v)]
        out_padded = torch.nn.functional.pad(out, (0, 16)) if out is not None else None
        o, l = flash_attn_gather_kv_func(
            qp, kp, vp, gather_kv_indices,
            softmax_scale=softmax_scale,
            return_lse=return_lse,
            out=out_padded,
            lse=lse,
            tile_mn=tile_mn,
            num_threads=num_threads,
            gather_kv_lengths=gather_kv_lengths,
        )
        o = o[..., :144]
        if out is not None:
            out.copy_(o)
            o = out
        return o, l
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    arch = _get_device_arch()
    if arch // 10 != 9:
        raise RuntimeError(f"minimal gather-KV forward requires SM90/Hopper, got sm_{arch}")
    if q.requires_grad or k.requires_grad or v.requires_grad:
        raise RuntimeError("This minimal package is forward-only; detach inputs or use torch.inference_mode().")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"q/k/v must be fp16 or bf16, got {q.dtype}")
    if not (q.dtype == k.dtype == v.dtype):
        raise TypeError("q, k and v must have the same dtype")
    if not all(t.is_cuda for t in (q, k, v, gather_kv_indices)):
        raise RuntimeError("q, k, v and gather_kv_indices must be CUDA tensors")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must be 4D tensors")

    batch_size, seqlen_q, num_head, head_dim = q.shape
    batch_k, seqlen_k, num_head_kv, head_dim_k = k.shape
    batch_v, seqlen_v, num_head_kv_v, head_dim_v = v.shape
    if batch_k != batch_size or batch_v != batch_size:
        raise ValueError("q, k and v must have the same batch size")
    if seqlen_v != seqlen_k:
        raise ValueError("k and v must have the same sequence length")
    if num_head_kv_v != num_head_kv:
        raise ValueError("k and v must have the same number of KV heads")
    if head_dim_k != head_dim:
        raise ValueError("k head_dim must match q head_dim")
    if num_head % num_head_kv != 0:
        raise ValueError("num_head must be divisible by num_head_kv")
    if num_head != num_head_kv:
        raise NotImplementedError("head-independent gather_kv_indices currently supports MHA only (num_head == num_head_kv)")

    alignment = 16 // q.element_size()
    _validate_head_dims(head_dim, head_dim_v, alignment)
    dtype = torch2cute_dtype_map[q.dtype]
    qhead_per_kvhead = num_head // num_head_kv
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    fwd_cfg = _tile_size_fwd_sm90(head_dim, head_dim_v)
    if tile_mn is not None:
        fwd_cfg = FwdConfig(tile_mn[0], tile_mn[1], fwd_cfg.mma_pv_is_rs, fwd_cfg.intra_wg_overlap)
    tile_m, tile_n = fwd_cfg.m_block_size, fwd_cfg.n_block_size
    if tile_n != 128:
        raise ValueError("minimal gather-KV forward currently requires tile_n == 128")

    gather_len = gather_kv_indices.shape[-1]
    num_m_blocks = math.ceil(seqlen_q / tile_m)
    if gather_kv_indices.dtype != torch.int32:
        raise TypeError("gather_kv_indices must be int32")
    if gather_kv_indices.stride(-1) != 1:
        raise ValueError("gather_kv_indices must be contiguous in the last dimension")
    if gather_len % tile_n != 0:
        raise ValueError(f"gather_len must be divisible by {tile_n}, got {gather_len}")
    if gather_len > math.ceil(seqlen_k / tile_n) * tile_n:
        raise ValueError(
            f"gather_len must be <= ceil_align(seqlen_k, {tile_n})={math.ceil(seqlen_k / tile_n) * tile_n}, "
            f"got gather_len={gather_len}, seqlen_k={seqlen_k}"
        )
    expected_gather_shape = (batch_size, num_head, num_m_blocks, gather_len)
    if tuple(gather_kv_indices.shape) != expected_gather_shape:
        raise ValueError(
            f"gather_kv_indices must have shape {expected_gather_shape}, "
            f"got {tuple(gather_kv_indices.shape)}"
        )

    has_gather_lengths = gather_kv_lengths is not None
    if has_gather_lengths:
        if gather_kv_lengths.dtype != torch.int32:
            raise TypeError("gather_kv_lengths must be int32")
        expected_lengths_shape = (batch_size, num_head, num_m_blocks)
        if tuple(gather_kv_lengths.shape) != expected_lengths_shape:
            raise ValueError(
                f"gather_kv_lengths must have shape {expected_lengths_shape}, "
                f"got {tuple(gather_kv_lengths.shape)}"
            )
        if not gather_kv_lengths.is_contiguous():
            gather_kv_lengths = gather_kv_lengths.contiguous()

    device = q.device
    out_dtype = q.dtype
    out_shape = (batch_size, seqlen_q, num_head, head_dim_v)
    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=device)
    else:
        _validate_tensor(out, "out", out_shape, out_dtype, device)
    lse_shape = (batch_size, num_head, seqlen_q)
    if lse is None:
        lse = torch.empty(lse_shape, dtype=torch.float32, device=device) if return_lse else None
    else:
        _validate_tensor(lse, "lse", lse_shape, torch.float32, device)

    q_tensor, k_tensor, v_tensor, o_tensor = [to_cute_tensor(t) for t in (q, k, v, out)]
    lse_tensor = to_cute_tensor(lse, assumed_align=4)
    gather_tensor = to_cute_tensor(gather_kv_indices)
    gather_lengths_tensor = to_cute_tensor(gather_kv_lengths) if has_gather_lengths else None
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        tile_m,
        tile_n,
        num_threads,
        fwd_cfg.mma_pv_is_rs,
        fwd_cfg.intra_wg_overlap,
        gather_len,
        lse is None,
        arch,
        has_gather_lengths,
    )
    if compile_key not in flash_attn_gather_kv_func.compile_cache:
        fa_fwd = FlashAttentionForwardSm90(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            is_causal=False,
            is_local=False,
            pack_gqa=qhead_per_kvhead > 1,
            tile_m=tile_m,
            tile_n=tile_n,
            num_stages=2,
            num_threads=num_threads,
            Q_in_regs=False,
            intra_wg_overlap=fwd_cfg.intra_wg_overlap,
            mma_pv_is_rs=fwd_cfg.mma_pv_is_rs,
            mask_mod=None,
            score_mod=None,
            has_aux_tensors=False,
            q_subtile_factor=1,
            paged_kv_non_tma=False,
            gather_kv=True,
        )
        flash_attn_gather_kv_func.compile_cache[compile_key] = cute.compile(
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            None,  # mCuSeqlensQ
            None,  # mCuSeqlensK
            None,  # mSeqUsedQ
            None,  # mSeqUsedK
            None,  # mPageTable
            gather_tensor,  # mGatherKVIndices
            gather_lengths_tensor,  # mGatherKVLengths
            None,  # window_size_left
            None,  # window_size_right
            None,  # learnable_sink
            None,  # blocksparse_tensors
            AuxData(),
            current_stream,
            options="--enable-tvm-ffi",
        )

    flash_attn_gather_kv_func.compile_cache[compile_key](
        q.detach(),
        k.detach(),
        v.detach(),
        out.detach(),
        lse,
        softmax_scale,
        None,  # mCuSeqlensQ
        None,  # mCuSeqlensK
        None,  # mSeqUsedQ
        None,  # mSeqUsedK
        None,  # mPageTable
        gather_kv_indices,  # mGatherKVIndices
        gather_kv_lengths,  # mGatherKVLengths
        None,  # window_size_left
        None,  # window_size_right
        None,  # learnable_sink
        None,  # blocksparse_tensors
        AuxData(),
    )
    return (out, lse) if return_lse else (out, None)


flash_attn_gather_kv_func.compile_cache = {}


def flash_attn_gather_kv_fp8_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gather_kv_indices: torch.Tensor,
    scale_qk_log2: torch.Tensor,
    scale_v: torch.Tensor,
    gather_kv_lengths: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
):
    """Forward-only SM90 FP8 (e4m3) token-level sparse gather-KV attention.

    Pre-quantized entry point — the caller owns quantization. See
    flash_mcsa.ops.fp8_quant for the helpers, and note the required order:
    quantize on the real head_dim, then pad_hdim_fp8(144 -> 160), then
    permute_v_dcols(v).

    - q=(B, Sq, H, D), k=(B, Sk, H, D), v=(B, Sk, H, Dv), all float8_e4m3fn.
      V's d-columns MUST be pre-permuted (ops.fp8_quant.permute_v_dcols) to
      compensate the kernel's in-smem transpose; passing raw V yields silently
      wrong output.
    - scale_qk_log2=(B, H, ceil(Sq/128)) fp32 = softmax_scale*log2(e)*s_q*s_k
    - scale_v=(B, H) fp32, folded into the final O normalization
    - out is bfloat16 (not fp8); lse is fp32 and always written by the kernel
    - D must be 128 or 160; MHA only; gather_len % 128 == 0; Sq % 128 == 0
    """
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    arch = _get_device_arch()
    if arch // 10 != 9:
        raise RuntimeError(f"fp8 gather-KV forward requires SM90/Hopper, got sm_{arch}")
    if q.requires_grad or k.requires_grad or v.requires_grad:
        raise RuntimeError("fp8 gather-KV attention is forward-only")
    if not (q.dtype == k.dtype == v.dtype == torch.float8_e4m3fn):
        raise TypeError(f"q/k/v must all be float8_e4m3fn, got {q.dtype}/{k.dtype}/{v.dtype}")
    if not all(t.is_cuda for t in (q, k, v, gather_kv_indices, scale_qk_log2, scale_v)):
        raise RuntimeError("all inputs must be CUDA tensors")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must be 4D tensors")

    batch_size, seqlen_q, num_head, head_dim = q.shape
    batch_k, seqlen_k, num_head_kv, head_dim_k = k.shape
    batch_v, seqlen_v, num_head_kv_v, head_dim_v = v.shape
    if batch_k != batch_size or batch_v != batch_size:
        raise ValueError("q, k and v must have the same batch size")
    if seqlen_v != seqlen_k:
        raise ValueError("k and v must have the same sequence length")
    if num_head_kv_v != num_head_kv:
        raise ValueError("k and v must have the same number of KV heads")
    if head_dim_k != head_dim:
        raise ValueError("k head_dim must match q head_dim")
    if num_head != num_head_kv:
        raise NotImplementedError("the fp8 gather-KV forward is MHA-only (num_head == num_head_kv)")
    if head_dim not in (128, 160) or head_dim_v != head_dim:
        raise ValueError(
            f"fp8 gather-KV forward supports head_dim 128 or 160 (144 zero-padded to 160) "
            f"with head_dim == head_dim_v, got D={head_dim}, Dv={head_dim_v}"
        )
    # The producer derives the V row pointer from the K row pointer plus a
    # constant byte offset, so K and V must share a layout.
    if k.stride() != v.stride():
        raise ValueError("the fused K+V gather requires identical K/V strides")

    fwd_cfg = _tile_size_fwd_sm90(head_dim, head_dim_v)
    tile_m, tile_n = fwd_cfg.m_block_size, fwd_cfg.n_block_size
    if tile_m != 128 or tile_n != 128:
        raise ValueError("the fp8 gather-KV forward requires tile_m == tile_n == 128")
    num_m_blocks = math.ceil(seqlen_q / tile_m)
    # A partial last m_block is fine: the tile scheduler launches it and the
    # epilogue predicates its O/LSE stores. Q's TMA load needs no guard (TMA
    # clamps out-of-bounds reads). Only pay for the predication when needed.
    check_seqlen_q_oob = (seqlen_q % tile_m) != 0

    gather_len = gather_kv_indices.shape[-1]
    if gather_kv_indices.dtype != torch.int32:
        raise TypeError("gather_kv_indices must be int32")
    if gather_kv_indices.stride(-1) != 1:
        raise ValueError("gather_kv_indices must be contiguous in the last dimension")
    if gather_len % tile_n != 0:
        raise ValueError(f"gather_len must be divisible by {tile_n}, got {gather_len}")
    if gather_len > math.ceil(seqlen_k / tile_n) * tile_n:
        raise ValueError(
            f"gather_len must be <= ceil_align(seqlen_k, {tile_n}), "
            f"got gather_len={gather_len}, seqlen_k={seqlen_k}"
        )
    expected_gather_shape = (batch_size, num_head, num_m_blocks, gather_len)
    if tuple(gather_kv_indices.shape) != expected_gather_shape:
        raise ValueError(
            f"gather_kv_indices must have shape {expected_gather_shape}, "
            f"got {tuple(gather_kv_indices.shape)}"
        )

    expected_scale_qk_shape = (batch_size, num_head, num_m_blocks)
    if tuple(scale_qk_log2.shape) != expected_scale_qk_shape:
        raise ValueError(
            f"scale_qk_log2 must have shape {expected_scale_qk_shape}, "
            f"got {tuple(scale_qk_log2.shape)}"
        )
    if tuple(scale_v.shape) != (batch_size, num_head):
        raise ValueError(
            f"scale_v must have shape {(batch_size, num_head)}, got {tuple(scale_v.shape)}"
        )
    if scale_qk_log2.dtype != torch.float32 or scale_v.dtype != torch.float32:
        raise TypeError("scale_qk_log2 and scale_v must be float32")
    scale_qk_log2 = scale_qk_log2.contiguous()
    scale_v = scale_v.contiguous()

    has_gather_lengths = gather_kv_lengths is not None
    if has_gather_lengths:
        if gather_kv_lengths.dtype != torch.int32:
            raise TypeError("gather_kv_lengths must be int32")
        if tuple(gather_kv_lengths.shape) != expected_scale_qk_shape:
            raise ValueError(
                f"gather_kv_lengths must have shape {expected_scale_qk_shape}, "
                f"got {tuple(gather_kv_lengths.shape)}"
            )
        if not gather_kv_lengths.is_contiguous():
            gather_kv_lengths = gather_kv_lengths.contiguous()
        # A zero would walk the producer's two-stage transpose flush into an
        # unacquired pipeline stage, and a value past gather_len // tile_n would
        # read indices outside the tile. The selection code clamps to [1, max]
        # by construction, so this is only a debug guard: checking it needs
        # .item(), i.e. a device sync on every call.
        if _FP8_CHECK_LENGTHS:
            lo = int(gather_kv_lengths.min().item())
            hi = int(gather_kv_lengths.max().item())
            if lo < 1:
                raise ValueError(f"gather_kv_lengths must be >= 1 everywhere, got min={lo}")
            if hi > gather_len // tile_n:
                raise ValueError(
                    f"gather_kv_lengths must be <= gather_len // {tile_n} = "
                    f"{gather_len // tile_n}, got max={hi}"
                )

    device = q.device
    out_shape = (batch_size, seqlen_q, num_head, head_dim_v)
    if out is None:
        out = torch.empty(out_shape, dtype=torch.bfloat16, device=device)
    else:
        _validate_tensor(out, "out", out_shape, torch.bfloat16, device)
    # The fp8 epilogue always writes LSE, so it is allocated even when the
    # caller does not ask for it.
    lse_shape = (batch_size, num_head, seqlen_q)
    if lse is None:
        lse = torch.empty(lse_shape, dtype=torch.float32, device=device)
    else:
        _validate_tensor(lse, "lse", lse_shape, torch.float32, device)

    # to_cute_tensor already maps float8_e4m3fn -> cutlass.Float8E4M3FN (it
    # exports the uint8 byte view because torch 2.9 DLPack lacks fp8), but at
    # tvm-ffi CALL time the tensors must be passed as .view(torch.uint8).
    q_tensor, k_tensor, v_tensor = [to_cute_tensor(t) for t in (q, k, v)]
    o_tensor = to_cute_tensor(out)
    lse_tensor = to_cute_tensor(lse, assumed_align=4)
    gather_tensor = to_cute_tensor(gather_kv_indices, assumed_align=4)
    scale_qk_tensor = to_cute_tensor(scale_qk_log2, assumed_align=4)
    scale_v_tensor = to_cute_tensor(scale_v, assumed_align=4)
    gather_lengths_tensor = (
        to_cute_tensor(gather_kv_lengths, assumed_align=4) if has_gather_lengths else None
    )
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compile_key = (
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        gather_len,
        arch,
        has_gather_lengths,
        check_seqlen_q_oob,
    )
    if compile_key not in flash_attn_gather_kv_fp8_func.compile_cache:
        fa_fwd = FlashAttentionForwardSm90Fp8(
            cutlass.Float8E4M3FN,
            head_dim,
            head_dim_v,
            1,  # qhead_per_kvhead
            is_causal=False,
            is_local=False,
            pack_gqa=False,
            tile_m=tile_m,
            tile_n=tile_n,
            # 3 stages is required by the two-stage-lag V transpose. At D=160
            # this is ~225KB of smem, inside the 227KB limit only because fp8
            # halves the per-stage footprint.
            num_stages=3,
            num_threads=384,
            Q_in_regs=False,
            mask_mod=None,
            score_mod=None,
            has_aux_tensors=False,
            q_subtile_factor=1,
            check_seqlen_q_oob=check_seqlen_q_oob,
        )
        flash_attn_gather_kv_fp8_func.compile_cache[compile_key] = cute.compile(
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            scale_qk_tensor,
            scale_v_tensor,
            gather_tensor,
            gather_lengths_tensor,
            current_stream,
            options="--enable-tvm-ffi",
        )

    flash_attn_gather_kv_fp8_func.compile_cache[compile_key](
        q.view(torch.uint8),
        k.view(torch.uint8),
        v.view(torch.uint8),
        out.detach(),
        lse,
        scale_qk_log2,
        scale_v,
        gather_kv_indices,
        gather_kv_lengths,
    )
    return (out, lse) if return_lse else (out, None)


flash_attn_gather_kv_fp8_func.compile_cache = {}


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gather_kv_indices: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    return_lse: bool = False,
    **kwargs,
):
    if gather_kv_indices is None:
        raise NotImplementedError("minimal_gather_kv_sm90 only provides gather_kv_indices forward")
    if causal:
        raise NotImplementedError("minimal_gather_kv_sm90 does not support causal attention")
    if softcap not in (None, 0.0):
        raise NotImplementedError("minimal_gather_kv_sm90 does not support softcap")
    if kwargs:
        unsupported = ", ".join(sorted(kwargs.keys()))
        raise NotImplementedError(f"Unsupported arguments in minimal_gather_kv_sm90: {unsupported}")
    return flash_attn_gather_kv_func(
        q,
        k,
        v,
        gather_kv_indices,
        softmax_scale=softmax_scale,
        return_lse=return_lse,
    )

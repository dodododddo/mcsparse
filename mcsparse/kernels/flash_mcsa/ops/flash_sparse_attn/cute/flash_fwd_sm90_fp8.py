# SM90 FP8 (e4m3) forward for token-level topk sparse attention with gathered KV.
#
# Ported from ../topk-sparse-attn-sm90/sparse_fwd_sm90_fp8.py and re-based onto
# this tree's FlashAttentionForwardSm90 (which folds gather-KV into the base via
# the gather_kv flag rather than a subclass), with variable-length support added.
#
# Differences vs the bf16 gather-KV path:
# - Q/K/V are fp8e4m3 in gmem and smem; O is bf16; both GEMMs run fp8 WGMMA
#   with fp32 accumulate. SM90 fp8 WGMMA needs K-major operands on both sides.
# - QK^T: natural layouts are already K-major (d contiguous). PV: V must be
#   K-major (token contiguous), so the producer transposes each gathered V tile
#   in smem (LDSM.T + prmt + STSM, see fp8_transpose_sm90.py). The transpose's
#   fixed d/n permutations are compensated outside the kernel: d-columns of V
#   are pre-permuted at quantization (ops.fp8_quant.permute_v_dcols) and the V
#   gather fetches tokens in CBLK-permuted order (load_index_topk_vperm).
# - P is staged through smem (sP, fp8 K-major) for the PV GEMM: SS avoids the
#   fp8 A-operand register-layout permutes FA3 uses for its RS path, and was
#   measured 3-8% faster at every config.
# - Scales: mScaleQK (B, H, Sq/128) fp32 = softmax_scale*log2e*s_q*s_k is read
#   per tile and becomes the softmax exp2 scale (dequantizes the logits);
#   mScaleV (B, H) is folded into the final O normalization. Softmax runs with
#   max_offset=8 (P in [0, 256]) to avoid e4m3 underflow on flat rows; the LSE
#   subtracts it back.
# - Epilogue is always registers->gmem (STG); no sO buffer, no O TMA.
# - mGatherKVLengths (B, H, Sq/128) optionally caps n_block per Q block, for
#   selection_mode='topp'/'topk_topp'. Read in both producer and consumer.

import math
from typing import Callable, Optional
from functools import partial
from dataclasses import dataclass
from types import SimpleNamespace

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from quack import copy_utils, layout_utils, sm90_utils
from quack.cute_dsl_utils import ParamsBase

from mcsparse.kernels.flash_mcsa.ops.flash_sparse_attn.cute.cute_dsl_utils import assume_tensor_aligned
import mcsparse.kernels.flash_mcsa.ops.flash_sparse_attn.cute.utils as utils
from mcsparse.kernels.flash_mcsa.ops.flash_sparse_attn.cute.mask import AttentionMask
from mcsparse.kernels.flash_mcsa.ops.flash_sparse_attn.cute.seqlen_info import SeqlenInfoQK
from mcsparse.kernels.flash_mcsa.ops.flash_sparse_attn.cute.block_info import BlockInfo
import mcsparse.kernels.flash_mcsa.ops.flash_sparse_attn.cute.pipeline as pipeline_custom
from mcsparse.kernels.flash_mcsa.ops.flash_sparse_attn.cute.softmax import Softmax
from mcsparse.kernels.flash_mcsa.ops.flash_sparse_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    StaticPersistentTileScheduler,
)
from mcsparse.kernels.flash_mcsa.ops.flash_sparse_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90
from mcsparse.kernels.flash_mcsa.ops.flash_sparse_attn.cute.topk_gather_kv import CpasyncGatherKVManagerSm90
from mcsparse.kernels.flash_mcsa.ops.flash_sparse_attn.cute.fp8_transpose_sm90 import SmemTransposeFp8

from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

import operator

LN2 = math.log(2.0)


@dsl_user_op
def cvt_e4m3x2_f32(a, b, *, loc=None, ip=None) -> cutlass.Int16:
    """Pack two fp32 into two e4m3 bytes (cvt.rn.satfinite.e4m3x2.f32)."""
    return cutlass.Int16(
        llvm.inline_asm(
            T.i16(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;",
            "=h,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def cvt_fp8(src: cute.Tensor, dst: cute.Tensor):
    """Vectorized fp32 -> e4m3 fragment convert (the generic .to() emits one
    cvt per element; this packs pairs, mirroring utils.cvt_f16)."""
    assert cute.size(src.shape) % 2 == 0
    dst16 = cute.recast_tensor(dst, cutlass.Int16)
    assert cute.size(dst16.shape) * 2 == cute.size(src.shape)
    for i in cutlass.range_constexpr(cute.size(dst16)):
        dst16[i] = cvt_e4m3x2_f32(src[2 * i], src[2 * i + 1])


@cute.jit
def reshape_acc_to_frgA_fp8(acc: cute.Tensor) -> cute.Tensor:
    """FA3 convert_layout_acc_Aregs fp8 branch (hopper/utils.h:167-176):
    ((2,2,N/8), MMA_M, MMA_N) -> ((4,2,2), MMA_M, N/32*MMA_N).
    The leading Layout<4>:1 pretends modes (0,0)/(0,1) are contiguous; only
    correct together with permute_aregs_fp8's register shuffles."""
    al = acc.layout
    l = cute.logical_divide(al, ((None, None, 4), None, None))
    st02 = l.stride[0][2][0]
    new_layout = cute.make_layout(
        ((4, 2, 2), l.shape[1], (l.shape[0][2][1], l.shape[2])),
        stride=(
            (1, st02, 2 * st02),
            l.stride[1],
            (l.stride[0][2][1], l.stride[2]),
        ),
    )
    return cute.make_tensor(acc.iterator, new_layout)


@cute.jit
def permute_aregs_fp8(frag: cute.Tensor):
    """FA3 permute_Aregs_fp8 (hopper/utils.h:516-547): quad shuffles + byte
    permutes fixing the fp8-downcast accumulator bytes into the RS operand-A
    register layout of the fp8 wgmma."""
    tidx, _, _ = cute.arch.thread_idx()
    quad_idx = tidx % 4
    lane_03 = quad_idx == 0 or quad_idx == 3
    selector_upper = 0x5410 if lane_03 else 0x1054
    selector_lower = 0x7632 if lane_03 else 0x3276
    # upper_map[quad] = {0, 3, 1, 2}: low bit = q0^q1, high bit = q0
    q0 = quad_idx % 2
    q1 = quad_idx // 2
    upper_lane = 2 * q0 + (q0 ^ q1)
    frag32 = cute.recast_tensor(frag, Int32)
    for j in cutlass.range_constexpr(cute.size(frag32) // 2):
        upper = frag32[2 * j]
        lower = frag32[2 * j + 1]
        upper0 = upper if lane_03 else lower
        lower0 = lower if lane_03 else upper
        upper0 = utils.shuffle_sync(upper0, upper_lane, width=4)
        lower0 = utils.shuffle_sync(lower0, upper_lane ^ 1, width=4)
        frag32[2 * j] = Int32(cute.arch.prmt(upper0, lower0, selector_upper))
        frag32[2 * j + 1] = Int32(cute.arch.prmt(upper0, lower0, selector_lower))


@dataclass
class Fp8Softmax(Softmax):
    """Softmax with a power-of-two bias on exp2 so P uses e4m3 range [0, 2^off].

    row_scale (O rescale between blocks) is offset-invariant; row_sum and acc_O
    both carry 2^offset so the final normalization needs no correction; only
    the LSE subtracts offset*ln2.
    """

    max_offset: cutlass.Constexpr[int] = 8

    @staticmethod
    def create(
        scale_log2: Float32,
        num_rows: cutlass.Constexpr[int],
        arch: cutlass.Constexpr[int] = 80,
        softmax_scale: Float32 | None = None,
        max_offset: cutlass.Constexpr[int] = 8,
    ):
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return Fp8Softmax(
            scale_log2, num_rows, row_max, row_sum, arch, softmax_scale, max_offset
        )

    @cute.jit
    def online_softmax(
        self,
        acc_S: cute.Tensor,
        is_first: cutlass.Constexpr[bool] = False,
        check_inf: cutlass.Constexpr[bool] = True,
    ) -> cute.Tensor:
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        row_scale = cute.make_fragment_like(self.row_max, Float32)
        row_max = self.row_max
        row_sum = self.row_sum
        scale_log2 = self.scale_log2
        arch = self.arch
        offset = Float32(self.max_offset)

        for r in cutlass.range(cute.size(row_max), unroll_full=True):
            acc_S_row = acc_S_mn[r, None].load()
            row_max_cur = utils.fmax_reduce(
                acc_S_row,
                init_val=row_max[r] if cutlass.const_expr(not is_first) else None,
                arch=arch,
            )
            row_max_cur = cute.arch.warp_reduction_max(row_max_cur, threads_in_group=4)
            row_max_prev = row_max[r]
            row_max[r] = row_max_cur
            if cutlass.const_expr(check_inf):
                row_max_cur = 0.0 if row_max_cur == -Float32.inf else row_max_cur
            row_max_cur_scaled = row_max_cur * scale_log2 - offset
            acc_S_row_exp = cute.math.exp2(
                acc_S_row * scale_log2 - row_max_cur_scaled, fastmath=True
            )
            if cutlass.const_expr(is_first):
                acc_S_row_sum = utils.fadd_reduce(acc_S_row_exp, init_val=None, arch=arch)
                row_scale[r] = 1.0
            else:
                row_scale[r] = cute.math.exp2(
                    (row_max_prev - row_max_cur) * scale_log2, fastmath=True
                )
                acc_S_row_sum = utils.fadd_reduce(
                    acc_S_row_exp, init_val=row_sum[r] * row_scale[r], arch=arch
                )
            row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None].store(acc_S_row_exp)
        return row_scale
        # NOTE: a per-row `if row_scale[r] != 1.0` skip inside rescale_O was
        # tried (bit-exact, cuts 64 FMAs/block on the common no-new-max path)
        # and measured 10-15% SLOWER: the runtime branch breaks the compiler's
        # scheduling of the unrolled multiply chain. Keep unconditional rescale.

    @cute.jit
    def finalize(self, final_scale: Float32 = 1.0, sink_val=None) -> cute.Tensor:
        assert sink_val is None
        row_sum = self.row_sum
        row_max = self.row_max
        scale_log2 = self.scale_log2
        row_sum.store(utils.warp_reduce(row_sum.load(), operator.add, width=4))
        row_scale = cute.make_fragment_like(row_max, Float32)
        offset = Float32(self.max_offset)
        for r in cutlass.range(cute.size(row_sum), unroll_full=True):
            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
            row_scale[r] = (
                cute.arch.rcp_approx(row_sum[r] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            ) * final_scale
            row_sum_cur = row_sum[r]
            # LSE over the dequantized logits: exp2 carried an extra 2^offset.
            row_sum[r] = (
                (row_max[r] * scale_log2 + cute.math.log2(row_sum_cur, fastmath=True) - offset)
                * LN2
                if not acc_O_mn_row_is_zero_or_nan
                else -Float32.inf
            )
        return row_scale


class FlashAttentionForwardSm90Fp8(FlashAttentionForwardSm90):
    """FP8 (e4m3) gather-KV forward. Self-contained: overrides the whole
    producer/consumer path rather than reusing the base's TMA/bf16 pipeline."""

    def __init__(self, *args, persistent: bool = False, check_seqlen_q_oob: bool = False, **kwargs):
        # gather_kv=True forces use_tma_KV=False (cp.async KV producer)
        kwargs.setdefault("gather_kv", True)
        kwargs.setdefault("intra_wg_overlap", True)
        # SS (P through smem) measured 3-8% faster than FA3's RS register path.
        kwargs.setdefault("mma_pv_is_rs", False)
        super().__init__(*args, **kwargs)
        self.persistent = persistent
        # seqlen_q % tile_m != 0: the tile scheduler launches a partial last
        # m_block, so epilogue_fp8 must predicate its O/LSE stores. Costs a
        # per-row branch in the epilogue, so it is opt-in from the host.
        self.check_seqlen_q_oob = check_seqlen_q_oob
        assert self.dtype in (cutlass.Float8E4M3FN,), "fp8 kernel needs e4m3 dtype"
        assert self.intra_wg_overlap, "only the intra-wg-overlap consumer is ported"
        assert self.qhead_per_kvhead == 1 and not self.pack_gqa, "fp8 path is MHA-only"
        # The gathered domain is dense and unordered: causal/local masking is
        # meaningless here, and the consumer runs with no mask_fn at all.
        assert not self.is_causal and not self.is_local
        # No hdim predication: 144 is zero-padded to 160 host-side, so the
        # kernel always sees an exact multiple of the swizzle/MMA-K granularity.
        assert not self.check_hdim_oob and not self.check_hdim_v_oob
        # 64x32 transpose blocks need hdim_v % 64; the 32x64 path (tile_n % 64)
        # covers hdim_v % 32 (e.g. the 160-padded 144 head dim)
        assert self.tile_hdimv % 32 == 0 and (
            self.tile_hdimv % 64 == 0 or self.tile_n % 64 == 0
        ), "V transpose needs hdim_v % 64 == 0 or (hdim_v % 32 and tile_n % 64)"
        assert self.tile_hdim % 32 == 0, "fp8 QK MMA contracts in K steps of 32"
        self.dtype_out = cutlass.BFloat16
        self.transposer = SmemTransposeFp8(self.tile_hdimv, self.tile_n)

    def _check_type(self, *args, **kwargs):
        pass  # fp8 in / bf16 out validated host-side in interface.py

    def _get_tiled_mma(self):
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.tile_n),
        )
        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,  # B (V^T) K-major: token dim contiguous
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.tile_hdimv),
            a_source=warpgroup.OperandSource.RMEM
            if self.mma_pv_is_rs
            else warpgroup.OperandSource.SMEM,
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sVstage_struct, sVt_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)], self.buffer_align_bytes
            ]
            for layout in (
                self.sQ_layout,
                self.sK_layout,
                self.sVstage_layout,
                self.sVt_layout,
            )
        ]
        cosize_sP = cute.cosize(self.sP_layout) if const_expr(self.sP_layout is not None) else 0
        sP_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cosize_sP], self.buffer_align_bytes
        ]
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_VT_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        @cute.struct
        class SharedStorageFp8:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_VT: mbar_ptr_VT_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sVstage: sVstage_struct
            sVt: sVt_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct

        return SharedStorageFp8

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) fp8
        mK: cute.Tensor,  # (b, s_k, h, d) fp8
        mV: cute.Tensor,  # (b, s_k, h, dv) fp8, d-columns pre-permuted
        mO: cute.Tensor,  # (b, s_q, h, dv) bf16
        mLSE: cute.Tensor,  # (b, h, s_q) fp32
        mScaleQK: cute.Tensor,  # (b, h, s_q // tile_m) fp32
        mScaleV: cute.Tensor,  # (b, h) fp32
        mIndexTopk: cute.Tensor,  # (b, h, s_q // tile_m, topk) int32
        mGatherKVLengths: Optional[cute.Tensor] = None,  # (b, h, s_q // tile_m) int32
        stream: cuda.CUstream = None,
    ):
        self.varlen_q = False
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        QO_layout_transpose = [1, 3, 2, 0]
        mQ, mO = [layout_utils.select(t, QO_layout_transpose) for t in (mQ, mO)]
        KV_layout_transpose = [1, 3, 2, 0]
        mK, mV = [layout_utils.select(t, KV_layout_transpose) for t in (mK, mV)]
        mLSE = layout_utils.select(mLSE, [2, 1, 0])
        # (b, h, n_mblocks, topk) -> (topk, n_mblocks, h, b)
        mIndexTopk = layout_utils.select(mIndexTopk, [3, 2, 1, 0])
        # (b, h, n_mblocks) -> (n_mblocks, h, b), matching the base's handling
        mGatherKVLengths = (
            layout_utils.select(mGatherKVLengths, [2, 1, 0])
            if const_expr(mGatherKVLengths is not None)
            else None
        )
        # scales keep (b, h, ...) mode order; read as per-tile scalars

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size
        self.num_threads_per_warp_group = 128
        self.num_wg_mma = self.num_mma_threads // self.num_threads_per_warp_group
        assert self.num_wg_mma in [1, 2]
        self.num_threads = self.num_threads_per_warp_group * (self.num_wg_mma + 1)
        self.num_producer_threads = 128  # gather + transpose need the full WG
        self.num_Q_load_threads = self.num_threads_per_warp_group
        self.num_epilogue_threads = self.num_mma_threads
        self.num_mma_regs, self.num_producer_regs = 224, 56
        self.use_block_sparsity = False
        self.use_scheduler_barrier = (
            (self.num_wg_mma >= 2 and self.tile_hdim <= 128)
            if const_expr(self.intra_wg_overlap)
            else (self.num_wg_mma == 2)
        )
        self.use_tma_Q = True
        self.use_tma_O = False
        self.rescale_O_before_gemm = False
        self._setup_attributes()
        self.sQ_layout, self.sK_layout, self.sVstage_layout, self.sVt_layout = [
            sm90_utils.make_smem_layout(self.dtype, LayoutEnum.ROW_MAJOR, shape, stage)
            for shape, stage in [
                ((self.tile_m, self.tile_hdim), None),
                ((self.tile_n, self.tile_hdim), self.num_stages),
                ((self.tile_n, self.tile_hdimv), self.num_stages),
                ((self.tile_hdimv, self.tile_n), self.num_stages),
            ]
        ]
        self.sP_layout = None
        if const_expr(not self.mma_pv_is_rs):
            self.sP_layout = sm90_utils.make_smem_layout(
                self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_n)
            )

        SharedStorage = self._get_shared_storage_cls()

        gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
        self.tma_copy_bytes = {
            "Q": cute.size_in_bytes(mQ.element_type, cute.select(self.sQ_layout, mode=[0, 1]))
        }
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_Q,
            mQ,
            self.sQ_layout,
            (self.tile_m, self.tile_hdim),
        )

        TileScheduler = (
            StaticPersistentTileScheduler if const_expr(self.persistent) else SingleTileScheduler
        )
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
            1,
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            mCuSeqlensQ=None,
            mSeqUsedQ=None,
            qhead_per_kvhead_packgqa=1,
            element_size=self.dtype.width // 8,
            is_persistent=self.persistent,
            lpt=False,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        fastdiv_mods = utils.compute_fastdiv_mods(mQ, mK, 1, False, None, None)

        s2r_tiled_copy, r2s_tiled_copy = self.transposer.make_tiled_copies()

        self.kernel(
            tma_tensor_Q,
            mK,
            mV,
            mO,
            mLSE,
            mScaleQK,
            mScaleV,
            mIndexTopk,
            mGatherKVLengths,
            tma_atom_Q,
            self.sQ_layout,
            self.sK_layout,
            self.sVstage_layout,
            self.sVt_layout,
            self.sP_layout,
            s2r_tiled_copy,
            r2s_tiled_copy,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            fastdiv_mods,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mScaleQK: cute.Tensor,
        mScaleV: cute.Tensor,
        mIndexTopk: cute.Tensor,
        mGatherKVLengths: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sVstage_layout: cute.ComposedLayout,
        sVt_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        s2r_tiled_copy: cute.TiledCopy,
        r2s_tiled_copy: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        fastdiv_mods=None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        tma_warp = ThreadCooperativeGroup(1)
        load_threads = ThreadCooperativeGroup(self.num_threads_per_warp_group)
        mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)
        mma_threads = ThreadCooperativeGroup(self.num_mma_threads)
        pipeline_q = pipeline_custom.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_Q.data_ptr(),
            num_stages=1,
            producer_group=tma_warp,
            consumer_group=mma_warps,
            tx_count=self.tma_copy_bytes["Q"],
            defer_sync=True,
        )
        pipeline_k = pipeline_custom.PipelineCpAsync.create(
            barrier_storage=storage.mbar_ptr_K.data_ptr(),
            num_stages=self.num_stages,
            producer_group=load_threads,
            consumer_group=mma_warps,
            defer_sync=True,
            elect_one_release=True,
            syncwarp_before_release=False,
        )
        # V staging: cp.async gather, produced and consumed by the producer WG
        pipeline_vt = pipeline_custom.PipelineCpAsync.create(
            barrier_storage=storage.mbar_ptr_VT.data_ptr(),
            num_stages=self.num_stages,
            producer_group=load_threads,
            consumer_group=load_threads,
            defer_sync=True,
            elect_one_release=False,
            syncwarp_before_release=False,
        )
        # transposed V: plain async pipeline, producer WG -> MMA warpgroups
        pipeline_v = pipeline.PipelineAsync.create(
            barrier_storage=storage.mbar_ptr_V.data_ptr(),
            num_stages=self.num_stages,
            producer_group=load_threads,
            consumer_group=mma_threads,
            defer_sync=True,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sVstage = storage.sVstage.get_tensor(sVstage_layout.outer, swizzle=sVstage_layout.inner)
        sVt = storage.sVt.get_tensor(sVt_layout.outer, swizzle=sVt_layout.inner)
        sP = None
        if const_expr(sP_layout is not None):
            sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)

        block_info = BlockInfo(
            self.tile_m, self.tile_n, False, False, False, None, None,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mIndexTopk.shape[0],
            mCuSeqlensQ=None, mCuSeqlensK=None, mSeqUsedQ=None, mSeqUsedK=None,
            mCuTotalMBlocks=None, mCuBlockIdxOffsets=None,
        )
        AttentionMaskCls = partial(
            AttentionMask, self.tile_m, self.tile_n,
            window_size_left=None, window_size_right=None,
            qhead_per_kvhead_packgqa=1,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if warp_idx < 4:  # Producer
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            self.load(
                mQ, mK, mV, mIndexTopk, mGatherKVLengths,
                sQ, sK, sVstage, sVt,
                tma_atom_Q,
                s2r_tiled_copy, r2s_tiled_copy,
                pipeline_k, pipeline_vt, pipeline_v, pipeline_q,
                block_info, SeqlenInfoCls, TileSchedulerCls,
            )
        else:  # Consumer
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_qk, tiled_mma_pv,
                mO, mLSE, mScaleQK, mScaleV, mGatherKVLengths,
                sQ, sK, sVt, sP,
                pipeline_k, pipeline_v, pipeline_q,
                tidx,
                block_info, SeqlenInfoCls, AttentionMaskCls, TileSchedulerCls,
                fastdiv_mods,
            )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mIndexTopk: cute.Tensor,
        mGatherKVLengths: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sVstage: cute.Tensor,
        sVt: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        s2r_tiled_copy: cute.TiledCopy,
        r2s_tiled_copy: cute.TiledCopy,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_vt: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tidx, _, _ = cute.arch.thread_idx()

        q_producer_phase = Int32(1)
        st_k = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
        st_vtp = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
        st_vtc = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages)
        st_vp = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
            load_Q, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
            )

            mK_cur = mK[None, None, head_idx, batch_idx]
            mV_cur = mV[None, None, head_idx, batch_idx]
            mIdx_cur = mIndexTopk[None, m_block, head_idx, batch_idx]
            gather_mgr = CpasyncGatherKVManagerSm90.create(
                mIdx_cur,
                tidx,
                # Real K length: SeqlenInfo carries seqlen_k == topk (the
                # gathered domain), but indices address the full K tensor.
                mK.shape[0],
                self.tile_n,
                self.tile_hdim,
                self.tile_hdimv,
                self.num_threads_per_warp_group,
                mK.element_type,
            )
            vIdx = cute.make_rmem_tensor((gather_mgr.topk_indices_per_thread,), Int32)

            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
            # Variable length (selection_mode topp/topk_topp): this Q block only
            # has mGatherKVLengths[...] valid tile_n-blocks of gathered KV.
            if const_expr(mGatherKVLengths is not None):
                n_block_max = mGatherKVLengths[m_block, head_idx, batch_idx]

            if warp_idx_in_wg == 0:
                pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
            q_producer_phase ^= 1

            gather_mgr.load_index_topk(n_block_max - 1)
            gather_mgr.load_index_topk_vperm(n_block_max - 1, vIdx)
            n_blocks = n_block_max - n_block_min
            for i in cutlass.range(n_blocks, unroll=1):
                n_block = n_block_max - 1 - i
                pipeline_k.producer_acquire(st_k)
                gather_mgr.load_X(mK_cur, sK[None, None, st_k.index], "K")
                cute.arch.cp_async_commit_group()
                pipeline_k.producer_commit(st_k)
                st_k.advance()

                pipeline_vt.producer_acquire(st_vtp)
                gather_mgr.load_X(
                    mV_cur, sVstage[None, None, st_vtp.index], "V", idx_regs=vIdx
                )
                cute.arch.cp_async_commit_group()
                pipeline_vt.producer_commit(st_vtp)
                st_vtp.advance()

                # prefetch next block's indices while the copies are in flight
                if n_block > n_block_min:
                    gather_mgr.load_index_topk(n_block - 1)
                    gather_mgr.load_index_topk_vperm(n_block - 1, vIdx)

                # transpose with a TWO-stage lag: V(i-2)'s cp.async has had two
                # full gather iterations to land, so the consumer_wait inside
                # rarely stalls (one-stage lag left ~a memory latency exposed).
                # Requires num_stages >= 3 (i-2, i-1, i live simultaneously).
                if i > 1:
                    self._transpose_stage(
                        pipeline_vt, pipeline_v, st_vtc, st_vp,
                        sVstage, sVt, s2r_tiled_copy, r2s_tiled_copy, tidx,
                    )
                    st_vtc.advance()
                    st_vp.advance()

            # Flush the last pending stages of this tile. The in-loop `i > 1`
            # ran max(0, n_blocks - 2) transposes, so this adds 2 (or 1 when
            # n_blocks == 1) for a total of exactly n_blocks. The trip count is
            # constexpr with a runtime predicate, so a dynamic n_blocks (from
            # mGatherKVLengths) is fine; n_blocks >= 1 is asserted host-side.
            for fi in cutlass.range_constexpr(2):
                if n_blocks > 1 or fi == 0:
                    self._transpose_stage(
                        pipeline_vt, pipeline_v, st_vtc, st_vp,
                        sVstage, sVt, s2r_tiled_copy, r2s_tiled_copy, tidx,
                    )
                    st_vtc.advance()
                    st_vp.advance()

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        pipeline_k.producer_tail(st_k)
        pipeline_v.producer_tail(st_vp)

    @cute.jit
    def _transpose_stage(
        self,
        pipeline_vt: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        st_vtc,
        st_vp,
        sVstage: cute.Tensor,
        sVt: cute.Tensor,
        s2r_tiled_copy: cute.TiledCopy,
        r2s_tiled_copy: cute.TiledCopy,
        tidx: Int32,
    ):
        pipeline_vt.consumer_wait(st_vtc, pipeline_vt.consumer_try_wait(st_vtc))
        pipeline_v.producer_acquire(st_vp)
        sVt_src = layout_utils.transpose_view(sVstage[None, None, st_vtc.index])
        self.transposer.transpose(
            s2r_tiled_copy, r2s_tiled_copy,
            sVt_src, sVt[None, None, st_vp.index], tidx,
        )
        cute.arch.fence_view_async_shared()
        pipeline_v.producer_commit(st_vp)
        pipeline_vt.consumer_release(st_vtc)

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mScaleQK: cute.Tensor,
        mScaleV: cute.Tensor,
        mGatherKVLengths: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sVt: cute.Tensor,
        sP: Optional[cute.Tensor],
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        tidx: Int32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        fastdiv_mods=None,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        warp_group_thread_layout = cute.make_layout(
            self.num_wg_mma, stride=self.num_threads_per_warp_group
        )
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx))
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim), sQ, sK
        )
        mma_qk_fn = partial(
            sm90_utils.gemm_zero_init, tiled_mma_qk, (self.tile_m, self.tile_n), tSrQ, tSrK
        )
        acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
            wg_mma_pv, (self.tile_m, self.tile_hdimv, self.tile_n), sP, sVt
        )
        mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)

        smem_copy_params = SimpleNamespace(smem_thr_copy_P=None, tPsP=None)
        if const_expr(not self.mma_pv_is_rs):
            smem_copy_atom_P = utils.get_smem_store_atom(
                self.arch.major * 10 + self.arch.minor, self.dtype
            )
            smem_thr_copy_P = cute.make_tiled_copy_C(smem_copy_atom_P, tiled_mma_qk).get_slice(
                tidx
            )
            tPsP = smem_thr_copy_P.partition_D(sP)
            smem_copy_params = SimpleNamespace(smem_thr_copy_P=smem_thr_copy_P, tPsP=tPsP)

        self.mma_init()

        q_consumer_phase = Int32(0)
        kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            scale_qk_log2 = Float32(mScaleQK[batch_idx, head_idx, m_block])
            scale_v = Float32(mScaleV[batch_idx, head_idx])
            softmax = Fp8Softmax.create(
                scale_qk_log2,
                num_rows=acc_O.shape[0][0] * acc_O.shape[1],
                max_offset=8,
            )

            mma_one_n_block = partial(
                self.mma_one_n_block_intrawg_overlap_fp8,
                mma_qk_fn=mma_qk_fn,
                pipeline_k=pipeline_k,
                pipeline_v=pipeline_v,
                acc_O=acc_O,
                tOrP=tOrP,
                smem_copy_params=smem_copy_params,
                check_inf=True,
                seqlen=seqlen,
                softmax=softmax,
            )
            process_first_half_block = partial(
                self.first_half_block_overlap_fp8,
                mma_qk_fn=mma_qk_fn,
                pipeline_k=pipeline_k,
                tOrP=tOrP,
                smem_copy_params=smem_copy_params,
                softmax=softmax,
            )
            process_last_half_block = partial(
                self.last_half_block_overlap_fp8,
                pipeline_v=pipeline_v,
                mma_pv_fn=mma_pv_fn,
            )

            # no mask_fn: topk % tile_n == 0 with all-valid indices means the
            # gathered domain never has partial tiles (asserted host-side)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
            # Must match the producer's bound exactly, or the pipelines desync.
            if const_expr(mGatherKVLengths is not None):
                n_block_max = mGatherKVLengths[m_block, head_idx, batch_idx]
            pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)
            O_should_accumulate = False

            kv_consumer_state = process_first_half_block(
                n_block=n_block_max - 1,
                kv_consumer_state=kv_consumer_state,
                is_first_block=True,
            )
            n_block_max -= 1
            for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=n_block_max - 1 - n_tile,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                )
                O_should_accumulate = True
            pipeline_q.consumer_release_w_index(0)
            kv_consumer_state = process_last_half_block(
                kv_consumer_state=kv_consumer_state,
                zero_init=not O_should_accumulate,
            )
            q_consumer_phase ^= 1

            row_scale = softmax.finalize(final_scale=scale_v)
            softmax.rescale_O(acc_O, row_scale)

            self.epilogue_fp8(
                acc_O, softmax.row_sum, mO, mLSE, seqlen,
                tiled_mma_pv, tidx, m_block, head_idx, batch_idx,
            )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def first_half_block_overlap_fp8(
        self,
        n_block: Int32,
        mma_qk_fn: Callable,
        kv_consumer_state,
        pipeline_k,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Fp8Softmax,
        is_first_block: bool = False,
    ):
        pipeline_k.consumer_wait(kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state))
        acc_S = mma_qk_fn(B_idx=kv_consumer_state.index, wg_wait=0)
        pipeline_k.consumer_release(kv_consumer_state)
        # no masking anywhere: topk % tile_n == 0 and all indices are valid, so
        # the gathered domain has no partial tiles (asserted host-side)
        softmax.online_softmax(acc_S, is_first=is_first_block)
        self._stage_P(acc_S, tOrP, smem_copy_params)
        return kv_consumer_state

    @cute.jit
    def _stage_P(self, acc_S, tOrP, smem_copy_params):
        """Convert P to e4m3 and hand it to the PV mma: RS keeps it in
        registers (fake-4 view + permute_aregs_fp8), SS stages through sP."""
        if const_expr(self.mma_pv_is_rs):
            tOrP_acc = reshape_acc_to_frgA_fp8(acc_S)
            cvt_fp8(tOrP_acc, tOrP)
            permute_aregs_fp8(tOrP)
        else:
            tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
            tOrP_cur = cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
            cvt_fp8(tOrP_acc, tOrP_cur)
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()

    @cute.jit
    def last_half_block_overlap_fp8(
        self,
        kv_consumer_state,
        pipeline_v,
        mma_pv_fn: Callable,
        zero_init: bool,
    ):
        pipeline_v.consumer_wait(kv_consumer_state, pipeline_v.consumer_try_wait(kv_consumer_state))
        mma_pv_fn(B_idx=kv_consumer_state.index, zero_init=zero_init, wg_wait=0)
        pipeline_v.consumer_release(kv_consumer_state)
        kv_consumer_state.advance()
        return kv_consumer_state

    @cute.jit
    def mma_one_n_block_intrawg_overlap_fp8(
        self,
        smem_pipe_read,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        pipeline_k,
        pipeline_v,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Fp8Softmax,
        seqlen,
        mask_fn: Optional[Callable] = None,
        check_inf: cutlass.Constexpr = True,
    ):
        smem_pipe_read_v = smem_pipe_read.clone()
        smem_pipe_read.advance()
        pipeline_k.consumer_wait(smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read))
        self.warp_scheduler_barrier_sync()
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        pipeline_v.consumer_wait(smem_pipe_read_v, pipeline_v.consumer_try_wait(smem_pipe_read_v))
        mma_pv_fn(B_idx=smem_pipe_read_v.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(1)
        pipeline_k.consumer_release(smem_pipe_read)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(acc_S, check_inf=check_inf)
        warpgroup.wait_group(0)
        pipeline_v.consumer_release(smem_pipe_read_v)
        self._stage_P(acc_S, tOrP, smem_copy_params)
        softmax.rescale_O(acc_O, row_scale)
        return smem_pipe_read

    @cute.jit
    def epilogue_fp8(
        self,
        acc_O: cute.Tensor,
        lse: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        seqlen,
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        # STG epilogue: O goes straight from registers to gmem in bf16.
        rO = cute.make_fragment_like(acc_O, self.dtype_out)
        rO.store(acc_O.load().to(self.dtype_out))
        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        thr_mma = tiled_mma.get_slice(tidx)
        mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2)[None, head_idx]
        gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
        gLSE_expanded_layout = cute.append(
            gLSE.layout, cute.make_layout((self.tile_hdimv,), stride=(0,))
        )
        gLSE_expanded = cute.make_tensor(gLSE.iterator, gLSE_expanded_layout)
        taccOgLSE = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(gLSE_expanded))
        taccOcO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))
        # When seqlen_q is not a multiple of tile_m the last m_block is partial:
        # the tile scheduler still launches it (ceil_div), so both the LSE and O
        # stores must drop the rows past seqlen_q or they write out of bounds.
        # Predication follows the bf16 base (flash_fwd.py epilogue): compare
        # thread 0's row coords against seqlen_q - m_block*tile_m, so the bound
        # is warp-uniform. check_seqlen_q_oob is a host-side constexpr, so the
        # aligned case compiles to exactly the unpredicated code as before.
        t0accOcO = None
        row_limit = None
        if const_expr(self.check_seqlen_q_oob):
            t0accOcO = layout_utils.reshape_acc_to_mn(thr_mma.get_slice(0).partition_C(cO))
            row_limit = seqlen.seqlen_q - m_block * self.tile_m - taccOcO[0][0]
        if taccOcO[0][1] == 0:
            for m in cutlass.range(cute.size(taccOgLSE.shape[1]), unroll_full=True):
                if const_expr(self.check_seqlen_q_oob):
                    if t0accOcO[m, 0][0] < row_limit:
                        taccOgLSE[m, 0] = lse[m]
                else:
                    taccOgLSE[m, 0] = lse[m]
        mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
        gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
        taccOgO = thr_mma.partition_C(gO)
        if const_expr(self.check_seqlen_q_oob):
            # rO/taccOgO are acc_O-shaped, so rows are mode 0 after
            # reshape_acc_to_mn and a row slice is [m, None] — the convention
            # Softmax.rescale_O relies on (it asserts
            # size(row_scale) == size(acc_O_mn, mode=[0]) then indexes
            # acc_O_mn[r, None]). The LSE tensor above is a different,
            # broadcast-expanded shape, so it keeps the base's own shape[1] loop.
            # autovec_copy on the whole tile would store the rows past seqlen_q.
            rO_mn = layout_utils.reshape_acc_to_mn(rO)
            tO_mn = layout_utils.reshape_acc_to_mn(taccOgO)
            assert cute.size(rO_mn, mode=[0]) == cute.size(tO_mn, mode=[0])
            for m in cutlass.range(cute.size(rO_mn, mode=[0]), unroll_full=True):
                if t0accOcO[m, 0][0] < row_limit:
                    cute.autovec_copy(rO_mn[m, None], tO_mn[m, None])
        else:
            cute.autovec_copy(rO, taccOgO)

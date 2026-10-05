# In-smem fp8 (8-bit) transpose for SM90, ported from FA3's Transpose_V
# (hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:170-206, 702-740).
#
# SM90 fp8 WGMMA requires the B operand K-major; ldmatrix/stmatrix have no
# 8-bit variants before sm_100, so the transpose runs at 16-bit granularity
# (LDSM.x4.trans) and fixes up the byte pairs with prmt (0x6420 / 0x7531).
# STSM uses FA3's NON-permuting value layout (their commented alternative at
# ML:201-203): the resulting fixed d/n permutations are compensated outside the
# kernel instead — V's d-columns are pre-permuted at quantization
# (ops.fp8_quant.permute_v_dcols) and the V gather fetches tokens in
# CBLK-permuted order (CpasyncGatherKVManagerSm90.load_index_topk_vperm).
#
# Layout contract (all fp8, per stage):
#   sVstage: (tile_n, hdim_v) row-major (d contiguous)  -- gather destination
#   sVt:     (hdim_v, tile_n) row-major (n contiguous)  -- WGMMA B operand (K-major)
# transpose() reads transpose_view(sVstage) == FA3's sVt (d, n) d-contiguous
# and writes sVt == FA3's sV (d, n) n-contiguous.

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute.nvgpu import warp


class SmemTransposeFp8:
    """Builds the tiled copies once (host side) and exposes a per-stage
    transpose to be run by exactly 128 threads (one warpgroup)."""

    def __init__(self, hdim_v: int, tile_n: int):
        # FA3 has two tilings (ML:184-194): 64x32 (d, n) blocks when
        # hdim_v % 64 == 0, else 32x64 blocks which need tile_n % 64 == 0
        # (used for the padded-160 head dim).
        assert hdim_v % 32 == 0
        assert tile_n % 32 == 0
        self.hdim_mult_64 = hdim_v % 64 == 0
        assert self.hdim_mult_64 or tile_n % 64 == 0
        self.hdim_v = hdim_v
        self.tile_n = tile_n
        self.ldsm_divide = (64, 8) if self.hdim_mult_64 else (32, 8)
        self.stsm_divide = (8, 16)

    @cute.jit
    def make_tiled_copies(self):
        # LDSM.T side: FA3 S2RTiledCopyVt (fp8-element units, over (d, n) view)
        s2r_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            cutlass.Float8E4M3FN,
        )
        if const_expr(self.hdim_mult_64):
            ldsm_thr = cute.make_layout((32, 4), stride=(4, 1))
        else:
            ldsm_thr = cute.make_layout((16, 4, 1, 2), stride=(4, 1, 0, 64))
        ldsm_val = cute.make_layout((2, 2, 1, 4), stride=(1, 2, 16, 4))
        # STSM side: FA3 R2STiledCopyV, non-permuting variant
        r2s_atom = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4),
            cutlass.Float8E4M3FN,
        )
        if const_expr(self.hdim_mult_64):
            stsm_thr = cute.make_layout((8, 4, 4, 1), stride=(4, 1, 32, 0))
        else:
            stsm_thr = cute.make_layout((8, 4, 2, 2), stride=(4, 1, 32, 64))
        stsm_val = cute.make_layout((1, 4, 2, 2), stride=(0, 1, 4, 8))
        s2r = cute.make_tiled_copy_tv(s2r_atom, ldsm_thr, ldsm_val)
        r2s = cute.make_tiled_copy_tv(r2s_atom, stsm_thr, stsm_val)
        return s2r, r2s

    @cute.jit
    def transpose(
        self,
        s2r_tiled_copy: cute.TiledCopy,
        r2s_tiled_copy: cute.TiledCopy,
        sVt_src: cute.Tensor,  # (hdim_v, tile_n) view, d contiguous (transpose_view of staging)
        sV_dst: cute.Tensor,  # (hdim_v, tile_n), n contiguous
        tidx: Int32,
    ):
        s2r_thr = s2r_tiled_copy.get_slice(tidx)
        r2s_thr = r2s_tiled_copy.get_slice(tidx)
        tTranssVt = s2r_thr.partition_S(cute.flat_divide(sVt_src, self.ldsm_divide))
        tTranssV = r2s_thr.partition_D(cute.flat_divide(sV_dst, self.stsm_divide))
        # Both partitions cover (64, 32)-elem tiles per iteration; flatten the
        # iteration modes and loop. FA3 groups pairs for ILP; plain loop first.
        tTranssVt_g = cute.group_modes(tTranssVt, 1, cute.rank(tTranssVt))
        tTranssV_g = cute.group_modes(tTranssV, 1, cute.rank(tTranssV))
        for i in cutlass.range(cute.size(tTranssVt_g, mode=[1]), unroll_full=True):
            tTransrV = cute.make_fragment_like(tTranssV_g[None, 0])
            cute.copy(s2r_tiled_copy, tTranssVt_g[None, i], tTransrV)
            rV32 = cute.recast_tensor(tTransrV, Int32)
            for j in cutlass.range_constexpr(cute.size(rV32) // 2):
                upper = rV32[2 * j]
                lower = rV32[2 * j + 1]
                rV32[2 * j] = Int32(cute.arch.prmt(upper, lower, 0x6420))
                rV32[2 * j + 1] = Int32(cute.arch.prmt(upper, lower, 0x7531))
            cute.copy(r2s_tiled_copy, tTransrV, tTranssV_g[None, i])

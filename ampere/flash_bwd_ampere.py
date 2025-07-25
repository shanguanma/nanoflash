# reference from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_bwd.py
import math
from types import SimpleNamespace
from typing import Type, Callable, Optional
from functools import partial

from dataclasses import dataclass

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp
import cutlass.utils.ampere_helpers as sm80_utils_basic
from cutlass import Float32,Int32 
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import nvvm, llvm
from cutlass.cute.runtime import from_dlpack

class SeqlenInfo:
    def __init__(
        self,
        batch_idx: cutlass.Int32,
        seqlen_q_static: cutlass.Int32,
        seqlen_k_static: cutlass.Int32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
    ):
        self.offset_q = 0 if cutlass.const_expr(mCuSeqlensQ is None) else mCuSeqlensQ[batch_idx]
        self.offset_k = 0 if cutlass.const_expr(mCuSeqlensK is None) else mCuSeqlensK[batch_idx]
        if cutlass.const_expr(mSeqUsedQ is not None):
            self.seqlen_q = mSeqUsedQ[batch_idx]
        else:
            self.seqlen_q = (
                seqlen_q_static
                if cutlass.const_expr(mCuSeqlensQ is None)
                else mCuSeqlensQ[batch_idx + 1] - self.offset_q
            )
        if cutlass.const_expr(mSeqUsedK is not None):
            self.seqlen_k = mSeqUsedK[batch_idx]
        else:
            self.seqlen_k = (
                seqlen_k_static
                if cutlass.const_expr(mCuSeqlensK is None)
                else mCuSeqlensK[batch_idx + 1] - self.offset_k
            )
        self.has_cu_seqlens_q: int = mCuSeqlensQ is not None
        self.has_cu_seqlens_k: int = mCuSeqlensK is not None

        

@cute.jit
def shuffle_sync(
    value: cute.Numeric,
    offset: cute.typing.Int,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.Numeric:
    assert value.width % 32 == 0, "value type must be a multiple of 32 bits"
    # 1 -> 0b11111, 2 -> 0b11110, 4 -> 0b11100, 8 -> 0b11000, 16 -> 0b10000, 32 -> 0b00000
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp
    val = cute.make_fragment(1, type(value))
    val[0] = value
    val_i32 = cute.recast_tensor(val, cutlass.Int32)
    for i in cutlass.range_constexpr(cute.size(val_i32)):
        val_i32[i] = cute.arch.shuffle_sync(val_i32[i], offset, mask_and_clamp=mask_and_clamp)
    return val[0]


@dataclass(frozen=True)
class AttentionMask:
    m_block_size: cutlass.Constexpr[int]
    n_block_size: cutlass.Constexpr[int]
    seqlen_q: cutlass.Int32
    seqlen_k: cutlass.Int32
    window_size_left: Optional[cutlass.Int32] = None
    window_size_right: Optional[cutlass.Int32] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1  # only pass in if we're doing PackGQA

    @cute.jit
    def apply_mask(
        self,
        acc_S: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
    ) -> None:
        assert not (mask_causal and mask_local), "mask_causal and mask_local cannot be both True"
        acc_S_mn = make_acc_tensor_mn_view(acc_S)
        cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
        tScS_mn = make_acc_tensor_mn_view(thr_mma.partition_C(cS))
        # We use t0ScS as these indices are known at compile time. We then must subtract the
        # column limit by the thread column offset.
        t0ScS_mn = make_acc_tensor_mn_view(thr_mma.get_slice(0).partition_C(cS))
        thr_col_offset = tScS_mn[0][1]
        seqlenk_col_limit = self.seqlen_k - n_block * self.n_block_size - thr_col_offset
        if cutlass.const_expr(not mask_causal and not mask_local):
            if cutlass.const_expr(mask_seqlen):
                # traverse column index.
                for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                    # if t0ScS_mn[0, c][1] >= seqlenk_col_limit:
                    #     acc_S_mn[None, c].fill(-cutlass.Float32.inf)
                    oob = t0ScS_mn[0, c][1] >= seqlenk_col_limit
                    for r in cutlass.range_constexpr(cute.size(tScS_mn.shape[0])):
                        acc_S_mn[r, c] = -cutlass.Float32.inf if oob else acc_S_mn[r, c]
        else:  # Causal or local
            # If PackGQA, we split the work of compute divmod among threads in the same row
            threads_per_row = thr_mma.tv_layout_C.shape[0][0]
            if cutlass.const_expr(self.qhead_per_kvhead_packgqa != 1):
                assert cute.arch.WARP_SIZE % threads_per_row == 0, (
                    "threads_per_row must divide WARP_SIZE"
                )
                assert cute.size(acc_S_mn.shape[0]) <= threads_per_row
                tidx = thr_mma.thr_idx
                mma_m_idx = (
                    m_block * self.m_block_size + tScS_mn[tidx % threads_per_row, 0][0]
                ) // self.qhead_per_kvhead_packgqa
            causal_row_offset = (
                1 + self.seqlen_k - n_block * self.n_block_size - self.seqlen_q - thr_col_offset
            )
            if cutlass.const_expr(mask_causal):
                for r in cutlass.range_constexpr(cute.size(tScS_mn.shape[0])):
                    # get the column index limit based on current row. Only consider the row index, so the column index sets to 0.
                    if cutlass.const_expr(self.qhead_per_kvhead_packgqa == 1):
                        row_idx = tScS_mn[r, 0][0] + m_block * self.m_block_size
                    else:
                        row_idx = shuffle_sync(
                            mma_m_idx, r % threads_per_row, width=threads_per_row
                        )
                    col_limit_right = row_idx + causal_row_offset
                    if cutlass.const_expr(mask_seqlen):
                        col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                    # traverse column index.
                    for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                        # only consider the column index, so the row index sets to 0.
                        # if t0ScS_mn[0, c][1] >= col_limit_right:
                            # acc_S_mn[r, c] = -cutlass.Float32.inf
                        acc_S_mn[r, c] = -cutlass.Float32.inf if t0ScS_mn[0, c][1] >= col_limit_right else acc_S_mn[r, c]
            else:  # Local
                local_row_offset_right = (
                    causal_row_offset + self.window_size_right
                    if cutlass.const_expr(self.window_size_right is not None)
                    else None
                )
                local_row_offset_left = (
                    causal_row_offset - 1 - self.window_size_left
                    if cutlass.const_expr(self.window_size_left is not None)
                    else None
                )
                for r in cutlass.range_constexpr(cute.size(tScS_mn.shape[0])):
                    if cutlass.const_expr(self.qhead_per_kvhead_packgqa == 1):
                        row_idx = tScS_mn[r, 0][0] + m_block * self.m_block_size
                    else:
                        row_idx = shuffle_sync(
                            mma_m_idx, r % threads_per_row, width=threads_per_row
                        )
                    if cutlass.const_expr(self.window_size_right is not None):
                        col_limit_right = row_idx + local_row_offset_right
                        if cutlass.const_expr(mask_seqlen):
                            col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                    else:
                        col_limit_right = self.n_block_size
                    col_limit_left = (
                        row_idx + local_row_offset_left if cutlass.const_expr(self.window_size_left is not None) else 0
                    )
                    # if cute.arch.thread_idx()[0] == 128: cute.printf("n_block = {}, r = {}, row_idx = {}, causal_row_offset = {}, col_limit_right = {}, col_limit_left = {}", n_block, r, row_idx, causal_row_offset, col_limit_right, col_limit_left)
                    # traverse column index.
                    for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                        col_idx = t0ScS_mn[0, c][1]
                        # only consider the column index, so the row index sets to 0.
                        if col_idx >= col_limit_right or col_idx < col_limit_left:
                            acc_S_mn[r, c] = -cutlass.Float32.inf

@cute.jit
def convert_layout_acc_frgA(acc_layout: cute.Layout) -> cute.Layout:
    # For back to back gemm, convert layout of acc0 to gemm 1 accept layout.
    # For Sm80, as the mma instruction shape is 16x8x16, we need to convert from (4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
    # For Sm90, FP16/BF16, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((2, 2, 2), MMA_M, (N / 16, MMA_N))
    # TODO: Sm90 FP8
    if cutlass.const_expr(cute.rank(acc_layout.shape[0]) == 3):  # Sm90
        l = cute.logical_divide(
            acc_layout, ((None, None, 2), None, None)
        )  # ((2, 2, (2, N / 16)), MMA_M, MMA_N)
        rA_mma_view = cute.make_layout(
            (
                (l.shape[0][0], l.shape[0][1], l.shape[0][2][0]),
                l.shape[1],
                (l.shape[0][2][1], l.shape[2]),
            ),
            stride=(
                (l.stride[0][0], l.stride[0][1], l.stride[0][2][0]),
                l.stride[1],
                (l.stride[0][2][1], l.stride[2]),
            ),
        )
    else:  # Sm80
        # (4, MMA_M, MMA_N) -> (4, MMA_M, (2, MMA_N / 2))
        l = cute.logical_divide(acc_layout, (None, None, 2))
        rA_mma_view = cute.make_layout(
            (
                (l.shape[0], l.shape[2][0]),
                l.shape[1],
                l.shape[2][1],
            ),
            stride=(
                (l.stride[0], l.stride[2][0]),
                l.stride[1],
                l.stride[2][1],
            ),
        )
    return rA_mma_view

@dsl_user_op
def atomic_add_fp32(a: float | Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    # gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    # # cache_hint = cutlass.Int64(0x12F0000000000000)
    # llvm.inline_asm(
    #     None,
    #     [gmem_ptr_i64, Float32(a).ir_value(loc=loc, ip=ip)],
    #     # [gmem_ptr_i64, Float32(a).ir_value(loc=loc, ip=ip), cache_hint.ir_value()],
    #     "red.global.add.f32 [$0], $1;",
    #     # "red.global.add.L2::cache_hint.f32 [$0], $1, 0x12F0000000000000;",
    #     # "red.global.add.L2::cache_hint.f32 [$0], $1, $2;",
    #     "l,f",
    #     # "l,f,l",
    #     has_side_effects=True,
    #     is_align_stack=False,
    #     asm_dialect=llvm.AsmDialect.AD_ATT,
    # )
    nvvm.atomicrmw(
        res=T.f32(), op=nvvm.AtomicOpKind.FADD, ptr=gmem_ptr.llvm_ptr, a=Float32(a).ir_value()
    )


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


def mma_make_fragment_A(
    smem: cute.Tensor, thr_mma: cute.core.ThrMma, swapAB: cutlass.Constexpr[bool] = False
) -> cute.Tensor:
    if cutlass.const_expr(swapAB):
        return mma_make_fragment_B(smem, thr_mma)
    else:
        return thr_mma.make_fragment_A(thr_mma.partition_A(smem))


def mma_make_fragment_B(
    smem: cute.Tensor, thr_mma: cute.core.ThrMma, swapAB: cutlass.Constexpr[bool] = False
) -> cute.Tensor:
    if cutlass.const_expr(swapAB):
        return mma_make_fragment_A(smem, thr_mma)
    else:
        return thr_mma.make_fragment_B(thr_mma.partition_B(smem))

def convert_layout_acc_mn(acc_layout: cute.Layout) -> cute.Layout:
    """
    For Sm80, convert ((2, 2), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, MMA_N), ...).
    For Sm90, convert ((2, 2, V), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, V, MMA_N), ...).
    """
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    acc_layout_mn = cute.make_layout(
        (
            (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),  # MMA_M
            (
                acc_layout_col_major.shape[0][0],
                *acc_layout_col_major.shape[0][2:],
                acc_layout_col_major.shape[2],
            ),  # MMA_N
            *acc_layout_col_major.shape[3:],
        ),
        stride=(
            (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),  # MMA_M
            (
                acc_layout_col_major.stride[0][0],
                *acc_layout_col_major.stride[0][2:],
                acc_layout_col_major.stride[2],
            ),  # MMA_N
            *acc_layout_col_major.stride[3:],
        ),
    )
    return cute.composition(acc_layout, acc_layout_mn)


def make_acc_tensor_mn_view(acc: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout))

def make_tiled_copy_A(
    copy_atom: cute.CopyAtom, tiled_mma: cute.TiledMma, swapAB: cutlass.Constexpr[bool] = False
) -> cute.TiledCopy:
    if cutlass.const_expr(swapAB):
        return make_tiled_copy_B(copy_atom, tiled_mma)
    else:
        return cute.make_tiled_copy(
            copy_atom,
            layout_tv=tiled_mma.tv_layout_A_tiled,
            tiler_mn=(tiled_mma.get_tile_size(0), tiled_mma.get_tile_size(2)),
        )


def make_tiled_copy_B(
    copy_atom: cute.CopyAtom, tiled_mma: cute.TiledMma, swapAB: cutlass.Constexpr[bool] = False
) -> cute.TiledCopy:
    if cutlass.const_expr(swapAB):
        return make_tiled_copy_A(copy_atom, tiled_mma)
    else:
        return cute.make_tiled_copy(
            copy_atom,
            layout_tv=tiled_mma.tv_layout_B_tiled,
            tiler_mn=(tiled_mma.get_tile_size(1), tiled_mma.get_tile_size(2)),
        )

def make_tiled_copy_C(copy_atom: cute.CopyAtom, tiled_mma: cute.TiledMma) -> cute.TiledCopy:
    return cute.make_tiled_copy(
        copy_atom,
        layout_tv=tiled_mma.tv_layout_C_tiled,
        tiler_mn=(tiled_mma.get_tile_size(0), tiled_mma.get_tile_size(1)),
    )

@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
    # Only compute predicates for the "k" dimension. For the mn dimension, we will use "if"
    tApA = cute.make_fragment(
        cute.make_layout(
            (cute.size(tAcA, mode=[0, 1]), cute.size(tAcA, mode=[1]), cute.size(tAcA, mode=[2])),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
    return tApA


# copy from flash_fwd_ampere.py
def get_smem_layout_atom(dtype: Type[cutlass.Numeric], k_dim: int) -> cute.ComposedLayout:
    dtype_byte = cutlass.const_expr(dtype.width // 8)
    bytes_per_row = cutlass.const_expr(k_dim * dtype_byte)
    smem_k_block_size = cutlass.const_expr(
        128
        if bytes_per_row % 128 == 0
        else (64 if bytes_per_row % 64 == 0 else (32 if bytes_per_row % 32 == 0 else 16))
    ) // dtype_byte
    swizzle_bits = (
        4
        if smem_k_block_size == 128
        else (3 if smem_k_block_size == 64 else (2 if smem_k_block_size == 32 else 1))
    )
    swizzle_base = 2 if dtype_byte == 4 else (3 if dtype_byte == 2 else 4)
    return cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, swizzle_base, swizzle_base),
        0,
        cute.make_ordered_layout((8 if cutlass.const_expr(k_dim % 32 == 0) else 16, smem_k_block_size), order=(1, 0)),
    )
# copy from flash_fwd_ampere.py
@cute.jit
def gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsA: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_A: cute.TiledCopy,
    smem_thr_copy_B: cute.TiledCopy,
    hook_fn: Optional[Callable] = None,
    A_in_regs: cutlass.Constexpr[bool] = False,
    B_in_regs: cutlass.Constexpr[bool] = False,
    swap_AB: cutlass.Constexpr[bool] = False,
) -> None:
    if cutlass.const_expr(swap_AB):
        gemm(
            tiled_mma,
            acc,
            tCrB,
            tCrA,
            tCsB,
            tCsA,
            smem_thr_copy_B,
            smem_thr_copy_A,
            hook_fn,
            A_in_regs=B_in_regs,
            B_in_regs=A_in_regs,
            swap_AB=False,
        )
    else:
        tCrA_copy_view = smem_thr_copy_A.retile(tCrA)
        tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
        if cutlass.const_expr(not A_in_regs):
            cute.copy(smem_thr_copy_A, tCsA[None, None, 0], tCrA_copy_view[None, None, 0])
        if cutlass.const_expr(not B_in_regs):
            cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])
        for k in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
            if k < cute.size(tCsA.shape[2]) - 1:
                if cutlass.const_expr(not A_in_regs):
                    cute.copy(
                        smem_thr_copy_A, tCsA[None, None, k + 1], tCrA_copy_view[None, None, k + 1]
                    )
                if cutlass.const_expr(not B_in_regs):
                    cute.copy(
                        smem_thr_copy_B, tCsB[None, None, k + 1], tCrB_copy_view[None, None, k + 1]
                    )
            cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
            if cutlass.const_expr(k == 0 and hook_fn is not None):
                hook_fn()


@cute.jit
def gemm_rs(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_B: cute.TiledCopy,
    hook_fn: Optional[Callable] = None,
) -> None:
    tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
    cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        if cutlass.const_expr(k < cute.size(tCrA.shape[2]) - 1):
            cute.copy(smem_thr_copy_B, tCsB[None, None, k + 1], tCrB_copy_view[None, None, k + 1])
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
        if cutlass.const_expr(k == 0 and hook_fn is not None):
            hook_fn()


@cute.jit
def warp_reduce(
    val: cute.TensorSSA | cute.Numeric,
    op: Callable,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.TensorSSA | cute.Numeric:
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_fragment(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


class FlashAttentionBackwardPreprocess:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        m_block_size: int = 128,
        num_threads: int = 128,
    ):
        """
        All contiguous dimensions must be at least 16 bytes aligned which indicates the head dimension
        should be a multiple of 8.

        :param head_dim: head dimension
        :type head_dim: int
        :param m_block_size: m block size
        :type m_block_size: int
        :param num_threads: number of threads
        :type num_threads: int
        """
        self.dtype = dtype
        self.m_block_size = m_block_size
        # padding head_dim to a multiple of 32 as k_block_size
        hdim_multiple_of = 32
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.num_threads = num_threads

    @staticmethod
    def can_implement(dtype, head_dim, m_block_size, num_threads) -> bool:
        """Check if the kernel can be implemented with the given parameters.

        :param dtype: data type
        :type dtype: cutlass.Numeric
        :param head_dim: head dimension
        :type head_dim: int
        :param m_block_size: m block size
        :type m_block_size: int
        :param num_threads: number of threads
        :type num_threads: int

        :return: True if the kernel can be implemented, False otherwise
        :rtype: bool
        """
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if num_threads < m_block_size:  # For multiplying lse with log2
            return False
        return True

    def _setup_attributes(self):
        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        # Thread layouts for copies
        # We want kBlockKGmem to be a power of 2 so that when we do the summing,
        # it's just between threads in the same warp
        gmem_k_block_size = (
            128
            if self.head_dim_padded % 128 == 0
            else (
                64
                if self.head_dim_padded % 64 == 0
                else (32 if self.head_dim_padded % 32 == 0 else 16)
            )
        )
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype.width
        # atom_universal_copy: universal copy atom for O & dO load
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # tOdO_layout: thread layout for O & dO load
        self.gmem_threads_per_row = gmem_k_block_size // async_copy_elems
        assert self.num_threads % self.gmem_threads_per_row == 0
        tOdO_layout = cute.make_ordered_layout(
            (self.num_threads // self.gmem_threads_per_row, self.gmem_threads_per_row),
            order=(1, 0),
        )
        # Value layouts for copies
        vOdO_layout = cute.make_layout((1, async_copy_elems))
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy, tOdO_layout, vOdO_layout
        )
        self.gmem_tiled_copy_dO = cute.make_tiled_copy_tv(
            atom_universal_copy, tOdO_layout, vOdO_layout
        )

        async_copy_elems_accum = universal_copy_bits // cutlass.Float32.width
        atom_universal_copy_accum = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float32,
            num_bits_per_copy=universal_copy_bits,
        )
        assert (
            self.m_block_size * self.head_dim_padded // async_copy_elems_accum
        ) % self.num_threads == 0
        self.gmem_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            atom_universal_copy_accum,
            cute.make_layout(self.num_threads),
            cute.make_layout(async_copy_elems_accum),
        )

    @cute.jit
    def __call__(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mdPsum: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mLSElog2: Optional[cute.Tensor],
        mdQaccum: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        # Get the data type and check if it is fp16 or bf16
        if cutlass.const_expr(not (mO.element_type == mdO.element_type)):
            raise TypeError("All tensors must have the same data type")
        if cutlass.const_expr(not mO.element_type in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if cutlass.const_expr(not mdPsum.element_type in [cutlass.Float32]):
            raise TypeError("dPsum tensor must be Float32")
        if cutlass.const_expr(mdQaccum is not None):
            if cutlass.const_expr(not mdQaccum.element_type in [cutlass.Float32]):
                raise TypeError("dQaccum tensor must be Float32")
        if cutlass.const_expr(mLSE is not None):
            assert mLSElog2 is not None, "If mLSE is provided, mLSElog2 must also be provided"
            if cutlass.const_expr(not mLSE.element_type in [cutlass.Float32]):
                raise TypeError("LSE tensor must be Float32")
            if cutlass.const_expr(not mLSElog2.element_type in [cutlass.Float32]):
                raise TypeError("LSElog2 tensor must be Float32")

        self._setup_attributes()

        # grid_dim: (m_block, num_head, batch_size)
        grid_dim = (
            cute.ceil_div(mO.shape[1], self.m_block_size),
            cute.size(mO.shape[2]),
            cute.size(mO.shape[0]),
        )
        self.kernel(
            mO,
            mdO,
            mdPsum,
            mLSE,
            mLSElog2,
            mdQaccum,
            self.gmem_tiled_copy_O,
            self.gmem_tiled_copy_dO,
            self.gmem_tiled_copy_dQaccum,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mdPsum: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mLSElog2: Optional[cute.Tensor],
        mdQaccum: Optional[cute.Tensor],
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_dO: cute.TiledCopy,
        gmem_tiled_copy_dQaccum: cute.TiledCopy,
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()
        m_block, num_head, batch_size = cute.arch.block_idx()

        # ///////////////////////////////////////////////////////////////////////////////
        # Get the appropriate tiles for this thread block.
        # ///////////////////////////////////////////////////////////////////////////////
        blkOdO_shape = (self.m_block_size, self.head_dim_padded)
        # (m_block_size, head_dim)
        gO = cute.local_tile(mO[batch_size, None, num_head, None], blkOdO_shape, (m_block, 0))
        gdO = cute.local_tile(mdO[batch_size, None, num_head, None], blkOdO_shape, (m_block, 0))

        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        gmem_thr_copy_dO = gmem_tiled_copy_dO.get_slice(tidx)
        # (CPY_Atom, CPY_M, CPY_K)
        tOgO = gmem_thr_copy_O.partition_S(gO)
        tOgdO = gmem_thr_copy_dO.partition_S(gdO)

        # ///////////////////////////////////////////////////////////////////////////////
        # Predicate: Mark indices that need to copy when problem_shape isn't a multiple
        # of tile_shape
        # ///////////////////////////////////////////////////////////////////////////////
        # Construct identity layout for KV
        cOdO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tOcO = gmem_thr_copy_O.partition_S(cOdO)
        t0OcO = gmem_thr_copy_O.get_slice(0).partition_S(cOdO)
        tOpO = predicate_k(tOcO, limit=mO.shape[3])
        tOcdO = gmem_thr_copy_dO.partition_S(cOdO)
        t0OcdO = gmem_thr_copy_dO.get_slice(0).partition_S(cOdO)
        tOpdO = predicate_k(tOcdO, limit=mdO.shape[3])

        seqlen_q = mO.shape[1]
        seqlen_q_rounded = cute.round_up(seqlen_q, self.m_block_size)

        if cutlass.const_expr(mLSE is not None):
            gLSE = cute.local_tile(
                mLSE[batch_size, num_head, None], (self.m_block_size,), (m_block,)
            )
            lse = cutlass.Float32.inf
            if tidx < seqlen_q - m_block * self.m_block_size:
                lse = gLSE[tidx]

        tOrO = cute.make_fragment_like(tOgO)
        tOrdO = cute.make_fragment_like(tOgdO)
        assert cute.size(tOgO, mode=[0]) == cute.size(tOgdO, mode=[0])
        assert cute.size(tOgO, mode=[1]) == cute.size(tOgdO, mode=[1])
        assert cute.size(tOgO, mode=[2]) == cute.size(tOgdO, mode=[2])
        for m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
            # Instead of using tOcO, we using t0OcO and subtract the offset from the limit
            # (seqlen_q - m_block * kBlockM). This is because the entries of t0OcO are known at compile time.
            if t0OcO[0, m, 0][0] < seqlen_q - m_block * self.m_block_size - tOcO[0][0]:
                cute.copy(
                    gmem_thr_copy_O,
                    tOgO[None, m, None],
                    tOrO[None, m, None],
                    pred=tOpO[None, m, None] if cutlass.const_expr(self.check_hdim_oob) else None,
                )
                cute.copy(
                    gmem_thr_copy_dO,
                    tOgdO[None, m, None],
                    tOrdO[None, m, None],
                    pred=tOpdO[None, m, None] if cutlass.const_expr(self.check_hdim_oob) else None,
                )
        # Sum across the "k" dimension
        dpsum = (tOrO.load().to(cutlass.Float32) * tOrdO.load().to(cutlass.Float32)).reduce(
            cute.ReductionOp.ADD, init_val=0.0, reduction_profile=(0, None, 1)
        )
        dpsum = warp_reduce(dpsum, operator.add, width=self.gmem_threads_per_row)
        dP_sum = cute.make_fragment(cute.size(tOrO, mode=[1]), cutlass.Float32)
        dP_sum.store(dpsum)

        # Write dPsum from rmem -> gmem
        gdPsum = cute.local_tile(
            mdPsum[batch_size, num_head, None], (self.m_block_size,), (m_block,)
        )
        # Only the thread corresponding to column 0 writes out the lse to gmem
        if tOcO[0, 0, 0][1] == 0:
            for m in cutlass.range_constexpr(cute.size(dP_sum)):
                row = tOcO[0, m, 0][0]
                gdPsum[row] = dP_sum[m] if row < mO.shape[1] - m_block * self.m_block_size else 0.0

        # Clear dQaccum
        if cutlass.const_expr(mdQaccum is not None):
            blkdQaccum_shape = (self.m_block_size * self.head_dim_padded,)
            gdQaccum = cute.local_tile(
                mdQaccum[batch_size, num_head, None], blkdQaccum_shape, (m_block,)
            )
            gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_slice(tidx)
            tQgQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum)
            zero = cute.make_fragment_like(tQgQaccum)
            zero.fill(0.0)
            cute.copy(gmem_tiled_copy_dQaccum, zero, tQgQaccum)

        if cutlass.const_expr(mLSE is not None):
            gLSElog2 = cute.local_tile(
                mLSElog2[batch_size, num_head, None], (self.m_block_size,), (m_block,)
            )
            LOG2_E = math.log2(math.e)
            if tidx < seqlen_q_rounded - m_block * self.m_block_size:
                gLSElog2[tidx] = lse * LOG2_E if lse != -cutlass.Float32.inf else 0.0

class FlashAttentionBackwardPostprocess:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        # tiled_mma: cute.TiledMma,
        head_dim: int,
        m_block_size: int = 128,
        num_threads: int = 256,
        AtomLayoutMdQ: int = 1,
        dQ_swapAB: bool = False,
    ):
        """Initializes the configuration for a flash attention v2 kernel.

        All contiguous dimensions must be at least 16 bytes aligned which indicates the head dimension
        should be a multiple of 8.

        :param head_dim: head dimension
        :type head_dim: int
        :param m_block_size: m block size
        :type m_block_size: int
        """
        self.dtype = dtype
        self.m_block_size = m_block_size
        # padding head_dim to a multiple of 32 as k_block_size
        hdim_multiple_of = 32
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_oob = head_dim != self.head_dim_padded
        # self.tiled_mma = tiled_mma
        self.num_threads = num_threads
        self.AtomLayoutMdQ = AtomLayoutMdQ
        self.dQ_swapAB = dQ_swapAB

    @staticmethod
    def can_implement(dtype, head_dim, m_block_size, num_threads) -> bool:
        """Check if the kernel can be implemented with the given parameters.

        :param dtype: data type
        :type dtype: cutlass.Numeric
        :param head_dim: head dimension
        :type head_dim: int
        :param m_block_size: m block size
        :type m_block_size: int

        :return: True if the kernel can be implemented, False otherwise
        :rtype: bool
        """
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        return True

    def _setup_attributes(self):
        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        # Thread layouts for copies
        universal_copy_bits = 128
        async_copy_elems_accum = universal_copy_bits // cutlass.Float32.width
        atom_async_copy_accum = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float32,
            num_bits_per_copy=universal_copy_bits,
        )
        # We don't do bound checking for the gmem -> smem load so we just assert here.
        assert (
            self.m_block_size * self.head_dim_padded // async_copy_elems_accum
        ) % self.tiled_mma.size == 0
        self.g2s_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            atom_async_copy_accum,
            cute.make_layout(self.tiled_mma.size),
            cute.make_layout(async_copy_elems_accum),
        )
        atom_universal_copy_accum = cute.make_copy_atom(
            # multiply by 4 for Sm90
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float32,
            num_bits_per_copy=cutlass.Float32.width,
        )
        self.s2r_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            atom_universal_copy_accum,
            cute.make_layout(self.tiled_mma.size),
            cute.make_layout(1),  # 4 for Sm90
        )

        async_copy_elems = universal_copy_bits // self.dtype.width
        # atom_universal_copy: universal copy atom for dQ store
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # tdQ_layout: thread layout for dQ store
        assert self.head_dim_padded % async_copy_elems == 0
        gmem_threads_per_row = math.gcd(
            self.head_dim_padded // async_copy_elems, self.tiled_mma.size
        )
        assert self.tiled_mma.size % gmem_threads_per_row == 0
        tdQ_layout = cute.make_ordered_layout(
            (self.tiled_mma.size // gmem_threads_per_row, gmem_threads_per_row),
            order=(1, 0),
        )
        # Value layouts for copies
        vdQ_layout = cute.make_layout((1, async_copy_elems))
        self.gmem_tiled_copy_dQ = cute.make_tiled_copy_tv(
            atom_universal_copy, tdQ_layout, vdQ_layout
        )
        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory layout: dQaccum / dQ
        # ///////////////////////////////////////////////////////////////////////////////
        self.sdQaccum_layout = cute.make_layout(self.m_block_size * self.head_dim_padded)
        # We can't just use kHeadDim here. E.g. if MMA shape is 64 x 96 but split across 2 WGs,
        # then setting kBlockKSmem to 32 will cause "Static shape_div failure".
        # We want to treat it as 64 x 48, so kBlockKSmem should be 16.
        mma_shape_n = self.tiled_mma.get_tile_size(1)
        sdQ_layout_atom = get_smem_layout_atom(self.dtype, mma_shape_n)
        self.sdQ_layout = cute.tile_to_shape(
            sdQ_layout_atom, (self.m_block_size, self.head_dim_padded), (0, 1)
        )

    @cute.jit
    def __call__(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        # Get the data type and check if it is fp16 or bf16
        if cutlass.const_expr(not mdQ.element_type in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if cutlass.const_expr(mdQaccum is not None):
            if cutlass.const_expr(not mdQaccum.element_type in [cutlass.Float32]):
                raise TypeError("dQaccum tensor must be Float32")

        num_mma_warps = self.num_threads // 32
        AtomLayoutdQ = (
            (self.AtomLayoutMdQ, num_mma_warps // self.AtomLayoutMdQ, 1)
            if cutlass.const_expr(not self.dQ_swapAB)
            else (num_mma_warps // self.AtomLayoutMdQ, self.AtomLayoutMdQ, 1)
        )
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, cutlass.Float32, (16, 8, 16)),
            AtomLayoutdQ,
            permutation_mnk=(AtomLayoutdQ[0] * 16, AtomLayoutdQ[1] * 16, 16),
        )
        self.tiled_mma = tiled_mma

        self._setup_attributes()

        smem_size = max(
            cute.size_in_bytes(cutlass.Float32, self.sdQaccum_layout),
            cute.size_in_bytes(self.dtype, self.sdQ_layout),
        )

        # grid_dim: (m_block, num_head, batch_size)
        grid_dim = (
            cute.ceil_div(mdQ.shape[1], self.m_block_size),
            cute.size(mdQ.shape[2]),
            cute.size(mdQ.shape[0]),
        )
        self.kernel(
            mdQaccum,
            mdQ,
            scale,
            tiled_mma,
            self.dQ_swapAB,
            self.sdQaccum_layout,
            self.sdQ_layout,
            self.g2s_tiled_copy_dQaccum,
            self.s2r_tiled_copy_dQaccum,
            self.gmem_tiled_copy_dQ,
        ).launch(
            grid=grid_dim,
            block=[tiled_mma.size, 1, 1],
            smem=smem_size,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        scale: cutlass.Float32,
        tiled_mma: cute.TiledMma,
        dQ_swapAB: cutlass.Constexpr,
        sdQaccum_layout: cute.Layout,
        sdQ_layout: cute.ComposedLayout,
        g2s_tiled_copy_dQaccum: cute.TiledCopy,
        s2r_tiled_copy_dQaccum: cute.TiledCopy,
        gmem_tiled_copy_dQ: cute.TiledCopy,
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()
        m_block, num_head, batch_size = cute.arch.block_idx()

        # ///////////////////////////////////////////////////////////////////////////////
        # Get the appropriate tiles for this thread block.
        # ///////////////////////////////////////////////////////////////////////////////
        blkdQaccum_shape = (self.m_block_size * self.head_dim_padded,)
        gdQaccum = cute.local_tile(
            mdQaccum[batch_size, num_head, None], blkdQaccum_shape, (m_block,)
        )
        blkdQ_shape = (self.m_block_size, self.head_dim_padded)
        gdQ = cute.local_tile(mdQ[batch_size, None, num_head, None], blkdQ_shape, (m_block, 0))

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        sdQaccum = smem.allocate_tensor(cutlass.Float32, sdQaccum_layout, byte_alignment=1024)
        sdQ = cute.make_tensor(cute.recast_ptr(sdQaccum.iterator, dtype=self.dtype), sdQ_layout)

        seqlen_q = mdQ.shape[1]
        seqlen_q_rounded = cute.round_up(seqlen_q, self.m_block_size)

        # Step 1: load dQaccum from gmem to smem
        g2s_thr_copy_dQaccum = g2s_tiled_copy_dQaccum.get_slice(tidx)
        tdQgdQaccum = g2s_thr_copy_dQaccum.partition_S(gdQaccum)
        tdQsdQaccumg2s = g2s_thr_copy_dQaccum.partition_D(sdQaccum)
        # print(tdQgdQaccum)
        # print(tdQsdQaccum)
        cute.copy(g2s_tiled_copy_dQaccum, tdQgdQaccum, tdQsdQaccumg2s)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # Step 2: load dQ from smem to rmem
        s2r_thr_copy_dQaccum = s2r_tiled_copy_dQaccum.get_slice(tidx)
        tdQsdQaccum = s2r_thr_copy_dQaccum.partition_S(sdQaccum)
        # print(s2r_tiled_copy_dQaccum)
        # print(sdQaccum)
        # thr_mma = tiled_mma.get_slice(tidx)
        # print(tiled_mma)
        acc_shape = tiled_mma.partition_shape_C(
            (self.m_block_size, self.head_dim_padded)
            if cutlass.const_expr(not dQ_swapAB)
            else (self.head_dim_padded, self.m_block_size)
        )
        acc = cute.make_fragment(acc_shape, cutlass.Float32)
        assert cute.size(acc) == cute.size(tdQsdQaccum)
        tdQrdQaccum = s2r_thr_copy_dQaccum.retile(acc)
        # Somehow even after retiling the layouts of tdQsdQaccum and tdQrdQaccum are different.
        # So we have to do a for loop to copy
        # cute.copy(s2r_tiled_copy_dQaccum, tdQsdQaccum, tdQrdQaccum)
        # print(acc)
        # print(tdQsdQaccum)  # ((1, 1), 64)
        # print(tdQrdQaccum)  # ((1, 4), 4, 4)
        for i in cutlass.range_constexpr(cute.size(tdQsdQaccum)):
            tdQrdQaccum[i] = tdQsdQaccum[i]
        # Convert tdQrdQaccum from fp32 to fp16/bf16
        rdQ = cute.make_fragment_like(acc, self.dtype)
        rdQ.store((acc.load() * scale).to(self.dtype))

        # Step 3: Copy dQ from register to smem
        cute.arch.barrier()  # make sure all threads have finished loading dQaccum
        smem_copy_atom_dQ = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=cutlass.Float32.width
        )
        smem_thr_copy_dQ = make_tiled_copy_C(smem_copy_atom_dQ, tiled_mma).get_slice(tidx)
        taccdQrdQ = smem_thr_copy_dQ.retile(rdQ)
        taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ)
        cute.copy(smem_copy_atom_dQ, taccdQrdQ, taccdQsdQ)
        # print(taccdQrdQ)
        # print(taccdQsdQ)

        # Step 4: Copy dQ from smem to register to prepare for coalesced write to gmem
        gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_slice(tidx)
        tdQgdQ = gmem_thr_copy_dQ.partition_S(gdQ)
        tdQsdQ = gmem_thr_copy_dQ.partition_D(sdQ)
        tdQrdQ = cute.make_fragment_like(tdQsdQ, self.dtype)
        cute.arch.barrier()  # make sure all smem stores are done
        # TODO: check OOB when reading from smem if kBlockM isn't evenly tiled
        cute.autovec_copy(tdQsdQ, tdQrdQ)

        # Step 5: Copy dQ from register to gmem
        cdQ = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tdQcdQ = gmem_thr_copy_dQ.partition_S(cdQ)
        tdQpdQ = utils.predicate_k(tdQcdQ, limit=mdQ.shape[3])
        for rest_m in cutlass.range_constexpr(cute.size(tdQrdQ.shape[1])):
            if tdQcdQ[0, rest_m, 0][0] < mdQ.shape[1] - m_block * self.m_block_size:
                cute.copy(
                    gmem_tiled_copy_dQ,
                    tdQrdQ[None, rest_m, None],
                    tdQgdQ[None, rest_m, None],
                    pred=tdQpdQ[None, rest_m, None],
                )


class FlashAttentionBackwardSm80:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        m_block_size: int = 64,
        n_block_size: int = 128,
        num_stages_Q: int = 2,
        num_stages_dO: int = 2,
        num_threads: int = 256,
        is_causal: bool = False,
        SdP_swapAB: bool = False,
        dKV_swapAB: bool = False,
        dQ_swapAB: bool = False,
        AtomLayoutMSdP: int = 1,
        AtomLayoutNdKV: int = 8,
        AtomLayoutMdQ: int = 1,
        V_in_regs: bool = False,
    ):
        """Initializes the configuration for a flash attention v2 kernel.

        All contiguous dimensions must be at least 16 bytes aligned which indicates the head dimension
        should be a multiple of 8.

        :param head_dim: head dimension
        :type head_dim: int
        :param m_block_size: m block size
        :type m_block_size: int
        :param n_block_size: n block size
        :type n_block_size: int
        :param num_threads: number of threads
        :type num_threads: int
        :param is_causal: is causal
        """
        self.dtype = dtype
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 32
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        # Can save registers (and hence be faster) if we don't have to check hdim predication
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.qhead_per_kvhead = qhead_per_kvhead
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.num_threads = num_threads
        self.is_causal = is_causal
        self.num_stages_Q = num_stages_Q
        self.num_stages_dO = num_stages_dO
        self.SdP_swapAB = SdP_swapAB
        self.dKV_swapAB = dKV_swapAB
        self.dQ_swapAB = dQ_swapAB
        self.AtomLayoutMSdP = AtomLayoutMSdP
        self.AtomLayoutNdKV = AtomLayoutNdKV
        self.AtomLayoutMdQ = AtomLayoutMdQ
        num_mma_warps = self.num_threads // cute.arch.WARP_SIZE
        self.Mma_dKV_is_RS = AtomLayoutMSdP == 1 and AtomLayoutNdKV == num_mma_warps and SdP_swapAB and not dKV_swapAB
        self.V_in_regs = V_in_regs
        self.share_QV_smem = V_in_regs

    @staticmethod
    def can_implement(
        dtype, head_dim, head_dim_v, m_block_size, n_block_size, num_stages_Q, num_stages_dO,
        num_threads, is_causal,
        V_in_regs=False
    ) -> bool:
        """Check if the kernel can be implemented with the given parameters.

        :param dtype: data type
        :type dtype: cutlass.Numeric
        :param head_dim: head dimension
        :type head_dim: int
        :param m_block_size: m block size
        :type m_block_size: int
        :param n_block_size: n block size
        :type n_block_size: int
        :param num_threads: number of threads
        :type num_threads: int
        :param is_causal: is causal
        :type is_causal: bool

        :return: True if the kernel can be implemented, False otherwise
        :rtype: bool
        """
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if head_dim_v % 8 != 0:
            return False
        if n_block_size % 16 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        # Check if block size setting is out of shared memory capacity
        # Shared memory usage: Q tile + (K tile + V tile) where K and V use the same tile size
        smem_usage_Q = m_block_size * head_dim * num_stages_Q * 2
        smem_usage_dO = m_block_size * head_dim_v * num_stages_dO * 2
        smem_usage_K = n_block_size * head_dim * 2
        smem_usage_V = n_block_size * head_dim_v * 2
        smem_usage_QV = (smem_usage_Q + smem_usage_V) if not V_in_regs else max(smem_usage_Q, smem_usage_V)
        smem_usage = smem_usage_QV + smem_usage_dO + smem_usage_K
        smem_capacity = sm80_utils_basic.SMEM_CAPACITY["sm80"]
        if smem_usage > smem_capacity:
            return False
        return True

    def _check_type(
        self,
        mQ_type: Type[cutlass.Numeric],
        mK_type: Type[cutlass.Numeric],
        mV_type: Type[cutlass.Numeric],
        mdO_type: Type[cutlass.Numeric],
        mLSE_type: Type[cutlass.Numeric],
        mdPsum_type: Type[cutlass.Numeric],
        mdQaccum_type: Type[cutlass.Numeric],
        mdK_type: Type[cutlass.Numeric],
        mdV_type: Type[cutlass.Numeric],
    ):
        if cutlass.const_expr(not (mQ_type == mK_type == mV_type == mdO_type)):
            raise TypeError("All tensors must have the same data type")
        if cutlass.const_expr(self.qhead_per_kvhead == 1):
            if cutlass.const_expr(not (mdK_type == mdV_type == mQ_type)):
                raise TypeError("mdK and mdV tensors must have the same data type as mQ")
        else:
            if cutlass.const_expr(not (mdK_type == mdV_type == cutlass.Float32)):
                raise TypeError("mdKaccum and mdVaccum tensors must have the data type Float32")
        if cutlass.const_expr(not mQ_type in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if cutlass.const_expr(not mLSE_type in [cutlass.Float32]):
            raise TypeError("LSE tensor must be Float32")
        if cutlass.const_expr(not mdPsum_type in [cutlass.Float32]):
            raise TypeError("dPsum tensor must be Float32")
        if cutlass.const_expr(not mdQaccum_type in [cutlass.Float32]):
            raise TypeError("dQaccum tensor must be Float32")
        assert mQ_type == self.dtype

    def _setup_attributes(self):
        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory layout: Q/K/V
        # ///////////////////////////////////////////////////////////////////////////////
        sQ_layout_atom = get_smem_layout_atom(self.dtype, self.head_dim_padded)
        self.sQ_layout = cute.tile_to_shape(
            sQ_layout_atom, (self.m_block_size, self.head_dim_padded, self.num_stages_Q), (0, 1, 2),
        )
        sK_layout_atom = sQ_layout_atom
        self.sK_layout = cute.tile_to_shape(
            sK_layout_atom, (self.n_block_size, self.head_dim_padded), (0, 1),
        )
        sV_layout_atom = get_smem_layout_atom(self.dtype, self.head_dim_v_padded)
        self.sV_layout = cute.tile_to_shape(
            sV_layout_atom, (self.n_block_size, self.head_dim_v_padded), (0, 1),
        )
        sdO_layout_atom = sV_layout_atom
        self.sdO_layout = cute.tile_to_shape(
            sdO_layout_atom, (self.m_block_size, self.head_dim_v_padded, self.num_stages_dO), (0, 1, 2),
        )
        # TODO: do we set swizzle to be 3 here explicitly?
        sPdS_layout_atom = get_smem_layout_atom(self.dtype, self.n_block_size)
        self.sPdS_layout = cute.tile_to_shape(
            sPdS_layout_atom, (self.m_block_size, self.n_block_size), (0, 1),
        )
        # We set stride to be multiple of 64 so that if ShuffleLSE, even if threads read from sLSE but out of bounds,
        # it's still a valid smem address.
        self.sLSE_layout = cute.make_layout(
            (self.m_block_size, self.num_stages_Q),
            stride=(1, cute.round_up(self.m_block_size, 64)),
        )
        sLSEMma_layout = cute.make_layout(
            (self.m_block_size, self.n_block_size, self.num_stages_Q),
            stride=(1, 0, cute.round_up(self.m_block_size, 64)),
        )
        sLSEMma_layout_transposed = cute.make_layout(
            (self.n_block_size, self.m_block_size, self.num_stages_Q),
            stride=(0, 1, cute.round_up(self.m_block_size, 64)),
        )
        self.sLSEMma_layout = sLSEMma_layout if not self.SdP_swapAB else sLSEMma_layout_transposed

        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        # Thread layouts for copies
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype.width
        # atom_async_copy: async copy atom for QKV load
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # atom_universal_copy: universal copy atom for O store
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=universal_copy_bits,
        )
        # tQK_layout: thread layout for QK load
        tQK_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        assert self.num_threads % tQK_shape_dim_1 == 0, "num_threads must be divisible by tQK_shape_dim_1"
        tQK_layout = cute.make_ordered_layout(
            (self.num_threads // tQK_shape_dim_1, tQK_shape_dim_1), order=(1, 0),
        )
        # Do we need to check if we overshot kBlockM when we load Q?
        self.is_even_m_smem_q = self.m_block_size % tQK_layout.shape[0] == 0
        # Do we need to check if we overshot kBlockN when we load K?
        self.is_even_n_smem_k = self.n_block_size % tQK_layout.shape[0] == 0
        tVdO_shape_dim_1 = sV_layout_atom.outer.shape[1] // async_copy_elems
        assert self.num_threads % tVdO_shape_dim_1 == 0, "num_threads must be divisible by tVdO_shape_dim_1"
        tVdO_layout = cute.make_ordered_layout(
            (self.num_threads // tVdO_shape_dim_1, tVdO_shape_dim_1), order=(1, 0),
        )
        # Do we need to check if we overshot kBlockN when we load V?
        self.is_even_n_smem_v = self.n_block_size % tVdO_layout.shape[0] == 0
        self.is_even_m_smem_do = self.m_block_size % tVdO_layout.shape[0] == 0

        # Value layouts for copies
        vQKVdO_layout = cute.make_layout((1, async_copy_elems))

        # gmem_tiled_copy_QK: tiled copy for QK load
        self.gmem_tiled_copy_QK = cute.make_tiled_copy_tv(atom_async_copy, tQK_layout, vQKVdO_layout)
        self.gmem_tiled_copy_VdO = cute.make_tiled_copy_tv(atom_async_copy, tVdO_layout, vQKVdO_layout)
        self.gmem_tiled_copy_dK = cute.make_tiled_copy_tv(atom_universal_copy, tQK_layout, vQKVdO_layout)
        self.gmem_tiled_copy_dV = cute.make_tiled_copy_tv(atom_universal_copy, tVdO_layout, vQKVdO_layout)
        async_copy_elems_accum = universal_copy_bits // cutlass.Float32.width
        atom_async_copy_accum = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float32,
            num_bits_per_copy=universal_copy_bits,
        )
        self.gmem_tiled_copy_LSE = cute.make_tiled_copy_tv(
            atom_async_copy_accum,
            cute.make_layout(self.num_threads),
            cute.make_layout(async_copy_elems_accum),
        )
        self.gmem_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=cutlass.Float32.width
            ),
            cute.make_layout(self.num_threads),
            cute.make_layout(1)
        )
        if cutlass.const_expr(self.qhead_per_kvhead > 1):
            self.gmem_tiled_copy_dK = self.gmem_tiled_copy_dQaccum
            self.gmem_tiled_copy_dV = self.gmem_tiled_copy_dQaccum

    def _get_tiled_mma(self):
        num_mma_warps = self.num_threads // 32
        AtomLayoutSdP = (self.AtomLayoutMSdP, num_mma_warps // self.AtomLayoutMSdP, 1) if cutlass.const_expr(not self.SdP_swapAB) else (num_mma_warps // self.AtomLayoutMSdP, self.AtomLayoutMSdP, 1)
        tiled_mma_sdp = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, cutlass.Float32, (16, 8, 16)),
            AtomLayoutSdP,
            permutation_mnk=(AtomLayoutSdP[0] * 16, AtomLayoutSdP[1] * 16, 16),
        )
        AtomLayoutdKV = (self.AtomLayoutNdKV, num_mma_warps // self.AtomLayoutNdKV, 1) if cutlass.const_expr(not self.dKV_swapAB) else (num_mma_warps // self.AtomLayoutNdKV, self.AtomLayoutNdKV, 1)
        tiled_mma_dkv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, cutlass.Float32, (16, 8, 16)),
            AtomLayoutdKV,
            permutation_mnk=(AtomLayoutdKV[0] * 16, AtomLayoutdKV[1] * 16, 16),
        )
        AtomLayoutdQ = (self.AtomLayoutMdQ, num_mma_warps // self.AtomLayoutMdQ, 1) if cutlass.const_expr(not self.dQ_swapAB) else (num_mma_warps // self.AtomLayoutMdQ, self.AtomLayoutMdQ, 1)
        tiled_mma_dq = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, cutlass.Float32, (16, 8, 16)),
            AtomLayoutdQ,
            permutation_mnk=(AtomLayoutdQ[0] * 16, AtomLayoutdQ[1] * 16, 16),
        )
        return tiled_mma_sdp, tiled_mma_dkv, tiled_mma_dq

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct, sdO_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], 1024]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout, self.sdO_layout)
        ]
        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sQV], 1024]
        sLSE_struct, sdPsum_struct = [
            cute.struct.Align[cute.struct.MemRange[cutlass.Float32, cute.cosize(layout)], 128]
            for layout in (self.sLSE_layout, self.sLSE_layout)
        ]
        sP_struct, sdS_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], 128]
            for layout in (self.sPdS_layout, self.sPdS_layout)
        ]

        @cute.struct
        class SharedStorageSeparateQV:
            sK: sK_struct
            sV: sV_struct
            sQ: sQ_struct
            sdO: sdO_struct
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sP: sP_struct
            sdS: sdS_struct
            # TODO: the case where there's no sP

        @cute.struct
        class SharedStorageSharedQV:
            sK: sK_struct
            sV: sV_struct
            sQ: sQV_struct
            sdO: sdO_struct
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sP: sP_struct
            sdS: sdS_struct

        return SharedStorageSeparateQV if cutlass.const_expr(not self.share_QV_smem) else SharedStorageSharedQV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        # Get the data type and check if it is fp16 or bf16
        self._check_type(*(t.element_type if t is not None else None
                           for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)))
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()
        tiled_mma_sdp, tiled_mma_dkv, tiled_mma_dq = self._get_tiled_mma()
        # grid_dim: (n_block, num_head, batch_size)
        grid_dim = (
            cute.ceil_div(mK.shape[1], self.n_block_size),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[0]),
        )
        softmax_scale_log2 = softmax_scale * math.log2(math.e)
        self.kernel(
            mQ,
            mK,
            mV,
            mdO,
            mLSE,
            mdPsum,
            mdQaccum,
            mdK,
            mdV,
            softmax_scale,
            softmax_scale_log2,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sdO_layout,
            self.sPdS_layout,
            self.sLSE_layout,
            self.sLSEMma_layout,
            self.gmem_tiled_copy_QK,
            self.gmem_tiled_copy_VdO,
            self.gmem_tiled_copy_dK,
            self.gmem_tiled_copy_dV,
            self.gmem_tiled_copy_LSE,
            self.gmem_tiled_copy_dQaccum,
            tiled_mma_sdp,
            tiled_mma_dkv,
            tiled_mma_dq,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccu: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sLSEMma_layout: cute.Layout,
        gmem_tiled_copy_QK: cute.TiledCopy,
        gmem_tiled_copy_VdO: cute.TiledCopy,
        gmem_tiled_copy_dK: cute.TiledCopy,
        gmem_tiled_copy_dV: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        gmem_tiled_copy_dQaccum: cute.TiledCopy,
        tiled_mma_sdp: cute.TiledMma,
        tiled_mma_dkv: cute.TiledMma,
        tiled_mma_dq: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()
        n_block, head_idx, batch_idx = cute.arch.block_idx()

        m_block_max = cute.ceil_div(mQ.shape[1], self.m_block_size)
        m_block_min = 0
        if cutlass.const_expr(self.is_causal):
            m_block_min = max(
                (n_block * self.n_block_size + mQ.shape[1] - mK.shape[1]) // self.m_block_size,
                m_block_min,
            )
        # TODO: return early if m_block_max == 0

        # ///////////////////////////////////////////////////////////////////////////////
        # Get the appropriate tiles for this thread block.
        # ///////////////////////////////////////////////////////////////////////////////
        blkQ_shape = (self.m_block_size, self.head_dim_padded)
        blkK_shape = (self.n_block_size, self.head_dim_padded)
        blkV_shape = (self.n_block_size, self.head_dim_v_padded)
        blkdO_shape = (self.m_block_size, self.head_dim_v_padded)
        # (m_block_size, head_dim, m_block)
        gQ = cute.local_tile(mQ[batch_idx, None, head_idx, None], blkQ_shape, (None, 0))
        # (n_block_size, head_dim)
        head_idx_kv = head_idx // self.qhead_per_kvhead
        gK = cute.local_tile(mK[batch_idx, None, head_idx_kv, None], blkK_shape, (n_block, 0))
        # (n_block_size, head_dim_v)
        gV = cute.local_tile(mV[batch_idx, None, head_idx_kv, None], blkV_shape, (n_block, 0))
        # (m_block_size, head_dim_v, m_block)
        gdO = cute.local_tile(mdO[batch_idx, None, head_idx, None], blkdO_shape, (None, 0))
        gLSE = cute.local_tile(mLSE[batch_idx, head_idx, None], (self.m_block_size,), (None,))
        gdPsum = cute.local_tile(mdPsum[batch_idx, head_idx, None], (self.m_block_size,), (None,))
        gdQaccum = cute.local_tile(mdQaccu[batch_idx, head_idx, None], (self.m_block_size * self.head_dim_padded,), (None,))

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sK_layout)
        if cutlass.const_expr(not self.share_QV_smem):
            sV = storage.sV.get_tensor(sV_layout)
        else:
            sV = cute.make_tensor(cute.recast_ptr(sQ.iterator, dtype=self.dtype), sV_layout)
        sdO = storage.sdO.get_tensor(sdO_layout)
        sP = storage.sP.get_tensor(sPdS_layout)
        sdS = storage.sdS.get_tensor(sPdS_layout)
        sLSE = storage.sLSE.get_tensor(sLSE_layout)
        sdPsum = storage.sdPsum.get_tensor(sLSE_layout)
        sLSEMma = storage.sLSE.get_tensor(sLSEMma_layout)
        sdPsumMma = storage.sdPsum.get_tensor(sLSEMma_layout)

        # Transpose view of tensors for tiled mma
        sQt, sdOt, sKt, sPt, sdSt = [utils.transpose_view(t) for t in (sQ, sdO, sK, sP, sdS)]

        gmem_thr_copy_QK = gmem_tiled_copy_QK.get_slice(tidx)
        gmem_thr_copy_VdO = gmem_tiled_copy_VdO.get_slice(tidx)
        gmem_thr_copy_lse = gmem_tiled_copy_LSE.get_slice(tidx)
        gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_slice(tidx)
        # (CPY_Atom, CPY_M, CPY_K, m_block)
        tQgQ = gmem_thr_copy_QK.partition_S(gQ)
        tQsQ = gmem_thr_copy_QK.partition_D(sQ)
        # (CPY_Atom, CPY_N, CPY_K)
        tKgK = gmem_thr_copy_QK.partition_S(gK)
        tKsK = gmem_thr_copy_QK.partition_D(sK)
        # (CPY_Atom, CPY_N, CPY_K)
        tVgV = gmem_thr_copy_VdO.partition_S(gV)
        tVsV = gmem_thr_copy_VdO.partition_D(sV)
        # (CPY_Atom, CPY_M, CPY_K, m_block)
        tdOgdO = gmem_thr_copy_VdO.partition_S(gdO)
        tdOsdO = gmem_thr_copy_VdO.partition_D(sdO)
        tLSEgLSE = gmem_thr_copy_lse.partition_S(gLSE)
        tLSEsLSE = gmem_thr_copy_lse.partition_D(sLSE)
        tLSEgdPsum = gmem_thr_copy_lse.partition_S(gdPsum)
        tLSEsdPsum = gmem_thr_copy_lse.partition_D(sdPsum)
        tdQgdQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum)

        # ///////////////////////////////////////////////////////////////////////////////
        # Tile MMA compute thread partitions and allocate accumulators
        # ///////////////////////////////////////////////////////////////////////////////
        thr_mma_sdp = tiled_mma_sdp.get_slice(tidx)
        thr_mma_dkv = tiled_mma_dkv.get_slice(tidx)
        thr_mma_dq = tiled_mma_dq.get_slice(tidx)
        acc_shape_dK = thr_mma_dkv.partition_shape_C((self.n_block_size, self.head_dim_padded))
        acc_shape_dV = thr_mma_dkv.partition_shape_C((self.n_block_size, self.head_dim_v_padded))
        acc_dK = cute.make_fragment(acc_shape_dK, cutlass.Float32)
        acc_dV = cute.make_fragment(acc_shape_dV, cutlass.Float32)
        acc_dK.fill(0.0)
        acc_dV.fill(0.0)

        tSrQ = mma_make_fragment_A(sQ[None, None, 0], thr_mma_sdp, swapAB=self.SdP_swapAB)
        tSrK = mma_make_fragment_B(sK, thr_mma_sdp, swapAB=self.SdP_swapAB)
        tdPrdO = mma_make_fragment_A(sdO[None, None, 0], thr_mma_sdp, swapAB=self.SdP_swapAB)
        tdPrV = mma_make_fragment_B(sV, thr_mma_sdp, swapAB=self.SdP_swapAB)
        tdVrP = mma_make_fragment_A(sPt, thr_mma_dkv, swapAB=self.dKV_swapAB)
        tdVrdO = mma_make_fragment_B(sdOt[None, None, 0], thr_mma_dkv, swapAB=self.dKV_swapAB)
        tdKrdS = mma_make_fragment_A(sdSt, thr_mma_dkv, swapAB=self.dKV_swapAB)
        tdKrQ = mma_make_fragment_B(sQt[None, None, 0], thr_mma_dkv, swapAB=self.dKV_swapAB)
        tdQrdS = mma_make_fragment_A(sdS, thr_mma_dq, swapAB=self.dQ_swapAB)
        tdQrK = mma_make_fragment_B(sKt, thr_mma_dq, swapAB=self.dQ_swapAB)

        LSEslice = (None, 0, None) if cutlass.const_expr(not self.SdP_swapAB) else (0, None, None)
        tSsLSEMma = make_acc_tensor_mn_view(thr_mma_sdp.partition_C(sLSEMma))[LSEslice]
        tSsdPsumMma = make_acc_tensor_mn_view(thr_mma_sdp.partition_C(sdPsumMma))[LSEslice]

        # ///////////////////////////////////////////////////////////////////////////////
        # Smem copy atom tiling
        # ///////////////////////////////////////////////////////////////////////////////
        smem_copy_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype,
        )
        smem_copy_atom_transposed = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype,
        )
        smem_thr_copy_QdO = make_tiled_copy_A(
            smem_copy_atom, tiled_mma_sdp, swapAB=self.SdP_swapAB
        ).get_slice(tidx)
        smem_thr_copy_KV = make_tiled_copy_B(
            smem_copy_atom, tiled_mma_sdp, swapAB=self.SdP_swapAB
        ).get_slice(tidx)
        # TODO: should this be smem_copy_atom_transposed?
        smem_thr_copy_PdSt = make_tiled_copy_A(
            smem_copy_atom_transposed, tiled_mma_dkv, swapAB=self.dKV_swapAB
        ).get_slice(tidx)
        smem_thr_copy_QdOt = make_tiled_copy_B(
            smem_copy_atom_transposed, tiled_mma_dkv, swapAB=self.dKV_swapAB
        ).get_slice(tidx)
        smem_thr_copy_dS = make_tiled_copy_A(
            smem_copy_atom, tiled_mma_dq, swapAB=self.dQ_swapAB
        ).get_slice(tidx)
        smem_thr_copy_Kt = make_tiled_copy_B(
            smem_copy_atom_transposed, tiled_mma_dq, swapAB=self.dQ_swapAB
        ).get_slice(tidx)
        # TODO: what's the number of bits? What if SdP_swapAB
        r2s_thr_copy_PdS = make_tiled_copy_C(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=2 * self.dtype.width
            ),
            tiled_mma_sdp,
        ).get_slice(tidx)

        tSsQ = smem_thr_copy_QdO.partition_S(sQ)
        tdPsdO = smem_thr_copy_QdO.partition_S(sdO)
        tSsK = smem_thr_copy_KV.partition_S(sK)
        tdPsV = smem_thr_copy_KV.partition_S(sV)
        tdVsPt = smem_thr_copy_PdSt.partition_S(sPt)
        tdKsdSt = smem_thr_copy_PdSt.partition_S(sdSt)
        tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOt)
        tdKsQt = smem_thr_copy_QdOt.partition_S(sQt)
        tdQsdS = smem_thr_copy_dS.partition_S(sdS)
        tdQsKt = smem_thr_copy_Kt.partition_S(sKt)
        tPsP = r2s_thr_copy_PdS.partition_D(sP)
        tdSsdS = r2s_thr_copy_PdS.partition_D(sdS)

        # ///////////////////////////////////////////////////////////////////////////////
        # Predicate: Mark indices that need to copy when problem_shape isn't a multiple
        # of tile_shape
        # ///////////////////////////////////////////////////////////////////////////////
        # Construct identity layout for KV
        cQ = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tQcQ = gmem_thr_copy_QK.partition_S(cQ)
        t0QcQ = gmem_thr_copy_QK.get_slice(0).partition_S(cQ)
        if cutlass.const_expr(self.head_dim_padded == self.head_dim_v_padded):
            tdOcdO = tQcQ
            t0dOcdO = t0QcQ
        else:
            cdO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
            tdOcdO = gmem_thr_copy_VdO.partition_S(cdO)
            t0dOcdO = gmem_thr_copy_VdO.get_slice(0).partition_S(cdO)
        cLSE = cute.make_identity_tensor((self.m_block_size,))
        tLSEcLSE = gmem_thr_copy_lse.partition_S(cLSE)

        # Allocate predicate tensors for m and n, here we only allocate the tile of k, and
        # use "if" on the mn dimension.
        # This is to reduce register pressure and gets 2-3% performance gain.
        tQpQ = predicate_k(tQcQ, limit=mQ.shape[3])
        if cutlass.const_expr(self.same_hdim_kv):
            tdOpdO = tQpQ
        else:
            tdOpdO = predicate_k(tdOcdO, limit=mdO.shape[3])

        # group parameters for compute_one_m_block
        mma_params = SimpleNamespace(
            thr_mma_sdp=thr_mma_sdp, thr_mma_dkv=thr_mma_dkv, thr_mma_dq=thr_mma_dq,
            tSrQ=tSrQ, tSrK=tSrK, tdPrdO=tdPrdO, tdPrV=tdPrV,
            tdVrP=tdVrP, tdVrdO=tdVrdO, tdKrdS=tdKrdS, tdKrQ=tdKrQ,
            tdQrdS=tdQrdS, tdQrK=tdQrK,
            acc_dK=acc_dK, acc_dV=acc_dV,
        )
        smem_copy_params = SimpleNamespace(
            smem_thr_copy_QdO=smem_thr_copy_QdO,
            smem_thr_copy_KV=smem_thr_copy_KV,
            smem_thr_copy_PdSt=smem_thr_copy_PdSt,
            smem_thr_copy_QdOt=smem_thr_copy_QdOt,
            smem_thr_copy_dS=smem_thr_copy_dS,
            smem_thr_copy_Kt=smem_thr_copy_Kt,
            r2s_thr_copy_PdS=r2s_thr_copy_PdS,
            tSsQ=tSsQ, tSsK=tSsK, tdPsdO=tdPsdO, tdPsV=tdPsV,
            tSsLSEMma=tSsLSEMma, tSsdPsumMma=tSsdPsumMma,
            tPsP=tPsP, tdSsdS=tdSsdS,
            tdVsPt=tdVsPt, tdVsdOt=tdVsdOt, tdKsdSt=tdKsdSt, tdKsQt=tdKsQt,
            tdQsdS=tdQsdS, tdQsKt=tdQsKt,
        )
        gmem_copy_params = SimpleNamespace(
            gmem_thr_copy_dQaccum=gmem_thr_copy_dQaccum, tdQgdQaccum=tdQgdQaccum
        )
        seqlen = SeqlenInfo(batch_idx, mQ.shape[1], mK.shape[1])
        load_Q_LSE = partial(
            self.load_Q_LSE, gmem_tiled_copy_QK, gmem_tiled_copy_LSE,
            tQgQ, tQsQ, tQcQ, t0QcQ, tQpQ,
            tLSEgLSE, tLSEsLSE, tLSEcLSE, seqlen=seqlen.seqlen_q
        )
        load_dO_dPsum = partial(
            self.load_dO_dPsum, gmem_tiled_copy_VdO, gmem_tiled_copy_LSE,
            tdOgdO, tdOsdO, tdOcdO, t0dOcdO, tdOpdO,
            tLSEgdPsum, tLSEsdPsum, tLSEcLSE, seqlen=seqlen.seqlen_q
        )
        compute_one_m_block = partial(
            self.compute_one_m_block, mma_params=mma_params,
            smem_copy_params=smem_copy_params, gmem_copy_params=gmem_copy_params,
            load_Q_LSE=load_Q_LSE, load_dO_dPsum=load_dO_dPsum,
            m_block_max=m_block_max,
            softmax_scale_log2=softmax_scale_log2,
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Prologue
        # ///////////////////////////////////////////////////////////////////////////////
        # Start async loads of the last mn-tile, where we take care of the mn residue
        self.load_V(gmem_thr_copy_VdO, tVgV, tVsV, n_block, seqlen=seqlen.seqlen_k,
                    headdim=mV.shape[3])
        if cutlass.const_expr(self.V_in_regs):
            cute.arch.cp_async_commit_group()
        self.load_K(gmem_thr_copy_QK, tKgK, tKsK, n_block, seqlen=seqlen.seqlen_k,
                    headdim=mK.shape[3])
        cute.arch.cp_async_commit_group()

        if cutlass.const_expr(self.V_in_regs):
            cute.arch.cp_async_wait_group(1)
            cute.arch.barrier()
            tdPrV_copy_view = smem_thr_copy_KV.retile(tdPrV)
            cute.copy(smem_thr_copy_KV, tdPsV, tdPrV_copy_view)
            # Sync to avoid loading Q to smem_q, which overlaps with smem_v
            cute.arch.barrier()

        m_block = m_block_min
        assert self.num_stages_Q >= self.num_stages_dO
        for stage in cutlass.range_constexpr(self.num_stages_Q):
            if cutlass.const_expr(self.num_stages_Q == 1 or stage < self.num_stages_Q - 1):
                if stage == 0 or m_block + stage < m_block_max:
                    load_Q_LSE(m_block + stage, smem_pipe_write_q=stage)
                cute.arch.cp_async_commit_group()
            if cutlass.const_expr(stage < self.num_stages_dO):
                if stage == 0 or m_block + stage < m_block_max:
                    load_dO_dPsum(m_block + stage, smem_pipe_write_q=stage)
                cute.arch.cp_async_commit_group()

        # ///////////////////////////////////////////////////////////////////////////////
        # Mainloop
        # ///////////////////////////////////////////////////////////////////////////////
        # Start processing of the first n-block.
        mask = AttentionMask(self.m_block_size, self.n_block_size, seqlen.seqlen_q, seqlen.seqlen_k)
        mask_fn = partial(
            mask.apply_mask, n_block=n_block, thr_mma=thr_mma_sdp,
            mask_seqlen=True, mask_causal=self.is_causal
        )
        smem_pipe_read_q = cutlass.Int32(0)
        smem_pipe_read_do = cutlass.Int32(0)
        smem_pipe_write_q = cutlass.Int32(self.num_stages_Q - 1)
        smem_pipe_write_do = cutlass.Int32(0)
        for m_tile in cutlass.range(m_block_min, m_block_max, unroll=1):
            compute_one_m_block(
                m_tile, smem_pipe_read_q, smem_pipe_read_do, smem_pipe_write_q, smem_pipe_write_do,
                mask_fn=mask_fn,
            )
            smem_pipe_read_q = self.advance_pipeline(smem_pipe_read_q, self.num_stages_Q)
            smem_pipe_read_do = self.advance_pipeline(smem_pipe_read_do, self.num_stages_dO)
            smem_pipe_write_q = self.advance_pipeline(smem_pipe_write_q, self.num_stages_Q)
            smem_pipe_write_do = self.advance_pipeline(smem_pipe_write_do, self.num_stages_dO)

        # ///////////////////////////////////////////////////////////////////////////////
        # Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        # If GQA, we scale dK in the postprocessing kernel instead
        if cutlass.const_expr(self.qhead_per_kvhead == 1):
            acc_dK.store(acc_dK.load() * softmax_scale)
        # reuse sK and sV data iterator
        sdK = cute.make_tensor(sK.iterator, sK_layout)
        sdV = cute.make_tensor(sV.iterator, sV_layout)
        self.epilogue(
            acc_dK, acc_dV, mdK, mdV, sdK, sdV,
            gmem_tiled_copy_dK, gmem_tiled_copy_dV, tiled_mma_dkv,
            tidx, n_block, head_idx, batch_idx
        )

    @cute.jit
    def compute_one_m_block(
        self,
        m_block: cutlass.Int32,
        smem_pipe_read_q: cutlass.Int32,
        smem_pipe_read_do: cutlass.Int32,
        smem_pipe_write_q: cutlass.Int32,
        smem_pipe_write_do: cutlass.Int32,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        gmem_copy_params: SimpleNamespace,
        load_Q_LSE: Callable,
        load_dO_dPsum: Callable,
        m_block_max: cutlass.Int32,
        softmax_scale_log2: cutlass.Float32,
        mask_fn: Optional[Callable] = None,
    ):
        def load_Q_next():
            m_block_next = m_block + (self.num_stages_Q - 1 if cutlass.const_expr(self.num_stages_Q > 1) else 1)
            if m_block_next < m_block_max:
                load_Q_LSE(m_block_next, smem_pipe_write_q)
            cute.arch.cp_async_commit_group()

        def load_dO_next():
            if m_block + self.num_stages_dO < m_block_max:
                load_dO_dPsum(m_block + self.num_stages_dO, smem_pipe_write_do)
            cute.arch.cp_async_commit_group()

        # MMA S
        acc_shape_SdP = mma_params.thr_mma_sdp.partition_shape_C(
            (self.m_block_size, self.n_block_size) if cutlass.const_expr(not self.SdP_swapAB) else (self.n_block_size, self.m_block_size)
        )
        acc_S = cute.make_fragment(acc_shape_SdP, cutlass.Float32)
        acc_S.fill(0.0)
        cute.arch.cp_async_wait_group(1 if cutlass.const_expr(self.num_stages_Q > 1) else 0)
        cute.arch.barrier()
        gemm(
            mma_params.thr_mma_sdp, acc_S, mma_params.tSrQ, mma_params.tSrK,
            smem_copy_params.tSsQ[None, None, None, smem_pipe_read_q if cutlass.const_expr(self.num_stages_Q > 1) else 0],
            smem_copy_params.tSsK,
            smem_copy_params.smem_thr_copy_QdO, smem_copy_params.smem_thr_copy_KV,
            swap_AB=self.SdP_swapAB,
        )
        tLSErLSE = cute.make_fragment_like(smem_copy_params.tSsLSEMma[None, 0])
        cute.autovec_copy(
            smem_copy_params.tSsLSEMma[None, smem_pipe_read_q if cutlass.const_expr(self.num_stages_Q > 1) else 0], tLSErLSE
        )
        if cutlass.const_expr(mask_fn is not None):
            mask_fn(acc_S, m_block=m_block)
        acc_S_mn = make_acc_tensor_mn_view(acc_S)
        bidx = 0
        # if cute.arch.thread_idx()[0] == 0 and cute.arch.block_idx()[0] == bidx: cute.print_tensor(acc_S_mn)
        # if cute.arch.thread_idx()[0] == 0 and cute.arch.block_idx()[0] == 1: cute.print_tensor(tLSErLSE)
        assert cute.size(acc_S_mn, mode=[0]) == cute.size(tLSErLSE)
        for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
            acc_S_mn[r, None].store(utils.exp2f(acc_S_mn[r, None].load() * softmax_scale_log2 - tLSErLSE[r]))
        # if cute.arch.thread_idx()[0] == 0 and cute.arch.block_idx()[0] == bidx: cute.print_tensor(acc_S_mn)

        # MMA dP
        acc_dP = cute.make_fragment(acc_shape_SdP, cutlass.Float32)
        acc_dP.fill(0.0)
        cute.arch.cp_async_wait_group(1 if cutlass.const_expr(self.num_stages_dO > 1) else 0)
        cute.arch.barrier()
        gemm(
            mma_params.thr_mma_sdp, acc_dP, mma_params.tdPrdO, mma_params.tdPrV,
            smem_copy_params.tdPsdO[None, None, None, smem_pipe_read_do if cutlass.const_expr(self.num_stages_dO > 1) else 0],
            smem_copy_params.tdPsV,
            smem_copy_params.smem_thr_copy_QdO, smem_copy_params.smem_thr_copy_KV,
            hook_fn=load_Q_next if cutlass.const_expr(self.num_stages_Q > 1) else None,
            swap_AB=self.SdP_swapAB,
        )
        tLSErdPsum = cute.make_fragment_like(smem_copy_params.tSsdPsumMma[None, 0])
        cute.autovec_copy(
            smem_copy_params.tSsdPsumMma[None, smem_pipe_read_do if cutlass.const_expr(self.num_stages_dO > 1) else 0], tLSErdPsum
        )
        acc_dP_mn = make_acc_tensor_mn_view(acc_dP)
        # if cute.arch.thread_idx()[0] == 0 and cute.arch.block_idx()[0] == bidx: cute.print_tensor(acc_dP_mn)
        assert cute.size(acc_dP_mn, mode=[0]) == cute.size(tLSErdPsum)
        for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
            acc_dP_mn[r, None].store(acc_S_mn[r, None].load() * (acc_dP_mn[r, None].load() - tLSErdPsum[r]))
        # if cute.arch.thread_idx()[0] == 0 and cute.arch.block_idx()[0] == bidx: cute.print_tensor(acc_dP_mn)
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        if cutlass.const_expr(not self.Mma_dKV_is_RS):
            tPrP = smem_copy_params.r2s_thr_copy_PdS.retile(rP)  # ((Atom,AtomNum), MMA_N, MMA_N)
            cute.copy(smem_copy_params.r2s_thr_copy_PdS, tPrP, smem_copy_params.tPsP)
        rdS = cute.make_fragment_like(acc_dP, self.dtype)
        rdS.store(acc_dP.load().to(self.dtype))
        if cutlass.const_expr(not self.Mma_dKV_is_RS):
            cute.arch.barrier()  # Make sure P is written
        # For hdim 64, It's faster to write to smem_dS first before the dV gemm
        if cutlass.const_expr(not self.Mma_dKV_is_RS):
            tdSrdS = smem_copy_params.r2s_thr_copy_PdS.retile(rdS)
            cute.copy(smem_copy_params.r2s_thr_copy_PdS, tdSrdS, smem_copy_params.tdSsdS)
        if cutlass.const_expr(self.Mma_dKV_is_RS):
            tdVrP = cute.make_tensor(rP.iterator, convert_layout_acc_frgA(rP.layout))
        else:
            tdVrP = mma_params.tdVrP

        # MMA dK
        gemm(
            mma_params.thr_mma_dkv, mma_params.acc_dV, tdVrP, mma_params.tdVrdO,
            smem_copy_params.tdVsPt,
            smem_copy_params.tdVsdOt[None, None, None, smem_pipe_read_do if cutlass.const_expr(self.num_stages_dO > 1) else 0],
            smem_copy_params.smem_thr_copy_PdSt, smem_copy_params.smem_thr_copy_QdOt,
            A_in_regs=self.Mma_dKV_is_RS,
            swap_AB=self.dKV_swapAB,
        )
        # if cute.arch.thread_idx()[0] == 0 and cute.arch.block_idx()[0] == bidx: cute.print_tensor(mma_params.acc_dV)
        cute.arch.barrier()  # Make sure dS is written

        # MMA dQ
        def dQ_mma(hook_fn):
            acc_shape_dQ = mma_params.thr_mma_dq.partition_shape_C(
                (self.m_block_size, self.head_dim_padded) if cutlass.const_expr(not self.dQ_swapAB) else (self.head_dim_padded, self.m_block_size)
            )
            acc_dQ = cute.make_fragment(acc_shape_dQ, cutlass.Float32)
            acc_dQ.fill(0.0)
            gemm(
                mma_params.thr_mma_dq, acc_dQ, mma_params.tdQrdS, mma_params.tdQrK,
                smem_copy_params.tdQsdS, smem_copy_params.tdQsKt,
                smem_copy_params.smem_thr_copy_dS, smem_copy_params.smem_thr_copy_Kt,
                swap_AB=self.dQ_swapAB,
                hook_fn=hook_fn
            )
            # ((1, 1), num_elements)
            acc_dQ_atomic = gmem_copy_params.gmem_thr_copy_dQaccum.retile(acc_dQ)
            tdQgdQaccum_atomic = gmem_copy_params.tdQgdQaccum[None, None, m_block]
            assert cute.size(acc_dQ_atomic) == cute.size(tdQgdQaccum_atomic)
            # if cute.arch.thread_idx()[0] == 0: cute.print_tensor(acc_dQ)
            for i in cutlass.range_constexpr(cute.size(acc_dQ_atomic)):
                atomic_add_fp32(acc_dQ_atomic[i], elem_pointer(tdQgdQaccum_atomic, i))
                # utils.atomic_add_fp32(acc_dQ[i], tdQgdQaccum_atomic.iterator + i * tdQgdQaccum_atomic.stride[1])
            # if cute.arch.thread_idx()[0] == 64 and cute.arch.block_idx()[0] == bidx: cute.print_tensor(acc_dQ)

        # If num_stages_Q == 1, we want to do Mma_dK first so we can start loading Q for the next iteration
        if cutlass.const_expr(self.num_stages_Q > 1):
            dQ_mma(load_dO_next)

        # MMA dK
        if cutlass.const_expr(self.Mma_dKV_is_RS):
            tdKrdS = cute.make_tensor(rdS.iterator, convert_layout_acc_frgA(rdS.layout))
        else:
            tdKrdS = mma_params.tdKrdS
        gemm(
            mma_params.thr_mma_dkv, mma_params.acc_dK, tdKrdS, mma_params.tdKrQ,
            smem_copy_params.tdKsdSt,
            smem_copy_params.tdKsQt[None, None, None, smem_pipe_read_q if cutlass.const_expr(self.num_stages_Q > 1) else 0],
            smem_copy_params.smem_thr_copy_PdSt, smem_copy_params.smem_thr_copy_QdOt,
            A_in_regs=self.Mma_dKV_is_RS,
            swap_AB=self.dKV_swapAB,
            hook_fn=load_dO_next if cutlass.const_expr(self.num_stages_Q == 1) else None,
        )
        # if cute.arch.thread_idx()[0] == 0: cute.print_tensor(mma_params.acc_dK)
        if cutlass.const_expr(self.num_stages_Q == 1):
            cute.arch.barrier()
            dQ_mma(load_Q_next)

    @cute.jit
    def epilogue(
        self,
        acc_dK: cute.Tensor,
        acc_dV: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        sdK: cute.Tensor,
        sdV: cute.Tensor,
        gmem_tiled_copy_dK: cute.TiledCopy,
        gmem_tiled_copy_dV: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        n_block: cutlass.Int32,
        num_head: cutlass.Int32,
        batch_size: cutlass.Int32,
    ):
        rdV = cute.make_fragment_like(acc_dV, self.dtype)
        rdV.store(acc_dV.load().to(self.dtype))
        rdK = cute.make_fragment_like(acc_dK, self.dtype)
        rdK.store(acc_dK.load().to(self.dtype))
        gmem_thr_copy_dK = gmem_tiled_copy_dK.get_slice(tidx)
        gmem_thr_copy_dV = gmem_tiled_copy_dV.get_slice(tidx)

        if cutlass.const_expr(self.qhead_per_kvhead == 1):
            # Make sure all threads have finished reading K and V, otherwise we get racy dQ
            # because smem_q could be changed.
            cute.arch.barrier()
            # smem copy atom for dKV
            smem_copy_atom_dKV = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=2 * self.dtype.width
            )
            smem_thr_copy_dKV = make_tiled_copy_C(smem_copy_atom_dKV, tiled_mma).get_slice(tidx)
            taccdVrdV = smem_thr_copy_dKV.retile(rdV)
            taccdKrdK = smem_thr_copy_dKV.retile(rdK)
            taccdVsdV = smem_thr_copy_dKV.partition_D(sdV)
            taccdKsdK = smem_thr_copy_dKV.partition_D(sdK)
            # copy acc O from rmem to smem with the smem copy atom
            cute.copy(smem_copy_atom_dKV, taccdVrdV, taccdVsdV)
            cute.copy(smem_copy_atom_dKV, taccdKrdK, taccdKsdK)

            blkdK_shape = (self.n_block_size, self.head_dim_padded)
            blkdV_shape = (self.n_block_size, self.head_dim_v_padded)
            gdK = cute.local_tile(mdK[batch_size, None, num_head, None], blkdK_shape, (n_block, 0))
            gdV = cute.local_tile(mdV[batch_size, None, num_head, None], blkdV_shape, (n_block, 0))
            tdKsdK = gmem_thr_copy_dK.partition_S(sdK)
            tdKgdK = gmem_thr_copy_dK.partition_D(gdK)
            tdVsdV = gmem_thr_copy_dV.partition_S(sdV)
            tdVgdV = gmem_thr_copy_dV.partition_D(gdV)
            tdKrdK = cute.make_fragment_like(tdKgdK, self.dtype)
            tdVrdV = cute.make_fragment_like(tdVgdV, self.dtype)
            # sync before all smem stores are done.
            cute.arch.barrier()
            # load acc dK and dV from smem to rmem for wider vectorization
            # Need to check OOB when reading from smem if kBlockN isn't evenly tiled
            # TODO
            cute.autovec_copy(tdKsdK, tdKrdK)
            cute.autovec_copy(tdVsdV, tdVrdV)

            cdK = cute.make_identity_tensor((self.n_block_size, self.head_dim_padded))
            tdKcdK = gmem_thr_copy_dK.partition_S(cdK)
            t0dKcdK = gmem_tiled_copy_dK.get_slice(0).partition_S(cdK)
            if cutlass.const_expr(self.head_dim_padded == self.head_dim_v_padded):
                tdVcdV = tdKcdK
                t0dVcdV = t0dKcdK
            else:
                cdV = cute.make_identity_tensor((self.n_block_size, self.head_dim_v_padded))
                tdVcdV = gmem_thr_copy_dV.partition_S(cdV)
                t0dVcdV = gmem_tiled_copy_dV.get_slice(0).partition_S(cdV)
            tdKpdK = predicate_k(tdKcdK, limit=mdK.shape[3])
            if cutlass.const_expr(self.same_hdim_kv):
                tdVpdV = tdKpdK
            else:
                tdVpdV = predicate_k(tdVcdV, limit=mdV.shape[3])
            # copy acc dK and acc_dV from rmem to gmem
            for rest_m in cutlass.range_constexpr(cute.size(tdKrdK.shape[1])):
                if t0dKcdK[0, rest_m, 0][0] < mdK.shape[1] - n_block * self.n_block_size - tdKcdK[0][0]:
                    cute.copy(
                        gmem_tiled_copy_dK,
                        tdKrdK[None, rest_m, None],
                        tdKgdK[None, rest_m, None],
                        pred=tdKpdK[None, rest_m, None] if cutlass.const_expr(self.check_hdim_oob) else None,
                    )
            for rest_m in cutlass.range_constexpr(cute.size(tdVrdV.shape[1])):
                if t0dVcdV[0, rest_m, 0][0] < mdV.shape[1] - n_block * self.n_block_size - tdVcdV[0][0]:
                    cute.copy(
                        gmem_tiled_copy_dV,
                        tdVrdV[None, rest_m, None],
                        tdVgdV[None, rest_m, None],
                        pred=tdVpdV[None, rest_m, None] if cutlass.const_expr(self.check_hdim_v_oob) else None,
                    )

        else:  # qhead_per_kvhead > 1, do atomic add
            # For Sm90, we need to sync to avoid racy writes to smem_q
            # For Sm80, we don't need to sync since we're not touching smem
            num_head_kv = num_head // self.qhead_per_kvhead
            gdV = cute.local_tile(mdV[batch_size, num_head_kv, None], (self.n_block_size * self.head_dim_v_padded,), (n_block,))
            gdK = cute.local_tile(mdK[batch_size, num_head_kv, None], (self.n_block_size * self.head_dim_padded,), (n_block,))
            tdVgdVaccum = gmem_thr_copy_dV.partition_S(gdV)
            tdKgdKaccum = gmem_thr_copy_dK.partition_S(gdK)
            acc_dV_atomic = gmem_thr_copy_dV.retile(acc_dV)
            acc_dK_atomic = gmem_thr_copy_dK.retile(acc_dK)
            assert cute.size(acc_dV_atomic) == cute.size(tdVgdVaccum)
            assert cute.size(acc_dK_atomic) == cute.size(tdKgdKaccum)
            for i in cutlass.range_constexpr(cute.size(acc_dV_atomic)):
                atomic_add_fp32(acc_dV_atomic[i], elem_pointer(tdVgdVaccum, i))
            for i in cutlass.range_constexpr(cute.size(acc_dK_atomic)):
                atomic_add_fp32(acc_dK_atomic[i], elem_pointer(tdKgdKaccum, i))

    @cute.jit
    def advance_pipeline(self, pipeline_index, num_stages: cutlass.Constexpr):
        return pipeline_index + 1 if pipeline_index < num_stages - 1 else 0

    @cute.jit
    def load_K(
        self,
        gmem_thr_copy: cute.TiledCopy,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
        headdim: cutlass.Int32,
    ):
        cK = cute.make_identity_tensor((self.n_block_size, self.head_dim_padded))
        tKcK = gmem_thr_copy.partition_S(cK)
        t0KcK = gmem_thr_copy.get_slice(0).partition_S(cK)
        tKpK = predicate_k(tKcK, limit=headdim)
        for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
            # If kBlockN doesn't evenly divide the tiled copy, only the last `n` needs to be checked
            if self.is_even_n_smem_k or n < cute.size(tKsK.shape[1]) - 1 or tKcK[0, n, 0][0] < self.n_block_size:
                # Instead of using tKcK, we using t0KcK and subtract the offset from the limit
                # (seqlen - block * kBlockN). This is because the entries of t0KcK are known at compile time.
                predicate_n = t0KcK[0, n, 0][0] < seqlen - block * self.n_block_size - tKcK[0][0]
                predicate = cute.make_fragment_like(tKpK[None, 0, None])
                for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                    for i in cutlass.range_constexpr(cute.size(predicate.shape[0])):
                        predicate[i, k] = (tKpK[i, n, k] if cutlass.const_expr(self.check_hdim_oob) else True) and predicate_n
                cute.copy(
                    gmem_thr_copy, tKgK[None, n, None], tKsK[None, n, None], pred=predicate,
                )
            # We need to clear the sK smem tiles since we'll use sKt for mma_dq

    @cute.jit
    def load_V(
        self,
        gmem_thr_copy: cute.TiledCopy,
        tVgV: cute.Tensor,
        tVsV: cute.Tensor,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
        headdim: cutlass.Int32,
    ):
        cV = cute.make_identity_tensor((self.n_block_size, self.head_dim_v_padded))
        tVcV = gmem_thr_copy.partition_S(cV)
        t0VcV = gmem_thr_copy.get_slice(0).partition_S(cV)
        tVpV = predicate_k(tVcV, limit=headdim)
        for n in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
            # If kBlockN doesn't evenly divide the tiled copy, only the last `n` needs to be checked
            if self.is_even_n_smem_v or n < cute.size(tVsV.shape[1]) - 1 or tVcV[0, n, 0][0] < self.n_block_size:
                # Instead of using tVcV, we using t0VcV and subtract the offset from the limit
                # (seqlen - block * kBlockN). This is because the entries of t0VcV are known at compile time.
                predicate_n = t0VcV[0, n, 0][0] < seqlen - block * self.n_block_size - tVcV[0][0]
                predicate = cute.make_fragment_like(tVpV[None, 0, None])
                for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                    for i in cutlass.range_constexpr(cute.size(predicate.shape[0])):
                        predicate[i, k] = (tVpV[i, n, k] if cutlass.const_expr(self.check_hdim_oob) else True) and predicate_n
                cute.copy(
                    gmem_thr_copy, tVgV[None, n, None], tVsV[None, n, None], pred=predicate,
                )

    @cute.jit
    def load_Q_LSE(
        self,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        tQgQ: cute.Tensor,
        tQsQ: cute.Tensor,
        tQcQ: cute.Tensor,
        t0QcQ: cute.Tensor,
        tQpQ: cute.Tensor,
        tLSEgLSE: cute.Tensor,
        tLSEsLSE: cute.Tensor,
        tLSEcLSE: cute.Tensor,
        block: cutlass.Int32,
        smem_pipe_write_q: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            # If kBlockM doesn't evenly divide the tiled copy, only the last `m` needs to be checked
            if self.is_even_m_smem_q or m < cute.size(tQsQ.shape[1]) - 1 or tQcQ[0, m, 0][0] < self.m_block_size:
                # Instead of using tQcQ, we using t0QcQ and subtract the offset from the limit
                # (seqlen - block * kBlockM). This is because the entries of t0QcQ are known at compile time.
                predicate_m = t0QcQ[0, m, 0][0] < seqlen - block * self.m_block_size - tQcQ[0][0]
                predicate = cute.make_fragment_like(tQpQ[None, 0, None])
                for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                    for i in cutlass.range_constexpr(cute.size(predicate.shape[0])):
                        predicate[i, k] = (tQpQ[i, m, k] if cutlass.const_expr(self.check_hdim_oob) else True) and predicate_m
                cute.copy(
                    gmem_tiled_copy_Q,
                    tQgQ[None, m, None, block],
                    tQsQ[None, m, None, smem_pipe_write_q if cutlass.const_expr(self.num_stages_Q) > 1 else 0],
                    pred=predicate,
                )
            # We need to clear the sQ smem tiles since we'll use sQt for mma_dK
        # We made sure LSE length is padded so we read `kBlockM` elements so that all
        # elements in sLSE are filled. Without this we might have uninitialized sLSE values.
        for m in cutlass.range_constexpr(cute.size(tLSEsLSE.shape[1])):
            if tLSEcLSE[0, m][0] < self.m_block_size:
                cute.copy(
                    gmem_tiled_copy_LSE,
                    tLSEgLSE[None, m, block],
                    tLSEsLSE[None, m, smem_pipe_write_q if cutlass.const_expr(self.num_stages_Q > 1) else 0],
                )

    @cute.jit
    def load_dO_dPsum(
        self,
        gmem_tiled_copy_dO: cute.TiledCopy,
        gmem_tiled_copy_dPsum: cute.TiledCopy,
        tdOgdO: cute.Tensor,
        tdOsdO: cute.Tensor,
        tdOcdO: cute.Tensor,
        t0dOcdO: cute.Tensor,
        tdOpdO: cute.Tensor,
        tdPsumgdPsum: cute.Tensor,
        tdPsumsdPsum: cute.Tensor,
        tdPsumcdPsum: cute.Tensor,
        block: cutlass.Int32,
        smem_pipe_write_q: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        for m in cutlass.range_constexpr(cute.size(tdOsdO.shape[1])):
            # If kBlockM doesn't evenly divide the tiled copy, only the last `m` needs to be checked
            if self.is_even_m_smem_do or m < cute.size(tdOsdO.shape[1]) - 1 or tdOcdO[0, m, 0][0] < self.m_block_size:
                # Instead of using tdOcdO, we using t0dOcdO and subtract the offset from the limit
                # (seqlen - block * kBlockM). This is because the entries of t0dOcdO are known at compile time.
                predicate_m = t0dOcdO[0, m, 0][0] < seqlen - block * self.m_block_size - tdOcdO[0][0]
                predicate = cute.make_fragment_like(tdOpdO[None, 0, None])
                for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                    for i in cutlass.range_constexpr(cute.size(predicate.shape[0])):
                        predicate[i, k] = (tdOpdO[i, m, k] if cutlass.const_expr(self.check_hdim_oob) else True) and predicate_m
                cute.copy(
                    gmem_tiled_copy_dO,
                    tdOgdO[None, m, None, block],
                    tdOsdO[None, m, None, smem_pipe_write_q if cutlass.const_expr(self.num_stages_dO > 1) else 0],
                    pred=predicate,
                )
            # We need to clear the sQ smem tiles since we'll use sQt for mma_dK
        # We made sure LSE length is padded so we read `kBlockM` elements so that all
        # elements in sLSE are filled. Without this we might have uninitialized sLSE values.
        for m in cutlass.range_constexpr(cute.size(tdPsumgdPsum.shape[1])):
            if tdPsumcdPsum[0, m][0] < self.m_block_size:
                cute.copy(
                    gmem_tiled_copy_dPsum,
                    tdPsumgdPsum[None, m, block],
                    tdPsumsdPsum[None, m, smem_pipe_write_q if cutlass.const_expr(self.num_stages_dO > 1) else 0],
                )

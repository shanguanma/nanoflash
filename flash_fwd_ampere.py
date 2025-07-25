# reference from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd.py
import math
import operator
from types import SimpleNamespace
from typing import Type, Callable, Optional, Tuple
from functools import partial
from dataclasses import dataclass, fields

import enum

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import const_expr
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
import cutlass.utils.ampere_helpers as sm80_utils_basic
from cutlass import Float32, Int32, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import nvvm, llvm
from cutlass.cute.runtime import from_dlpack

# for testing
import torch
import cutlass.cute.testing as testing
import argparse
import time
import cutlass.torch as cutlass_torch


@cute.jit
def clz(x: Int32) -> Int32:
    # for i in cutlass.range_constexpr(32):
    #     if (1 << (31 - i)) & x:
    #         return Int32(i)
    # return Int32(32)
    # Early exit is not supported yet
    res = Int32(32)
    done = False
    for i in cutlass.range(32):
        if ((1 << (31 - i)) & x) and not done:
            res = Int32(i)
            done = True
    return res


def find_log2(x: Int32) -> Int32:
    a: Int32 = Int32(31 - clz(x))
    return a + ((x & (x - 1)) != 0)  # Round up, add 1 if not a power of 2.


@dsl_user_op
def umulhi(a: Int32, b: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "mul.hi.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )

class FastDivmod:
    def __init__(
        self, divisor: Int32, multipler: Uint32, shift_right: Uint32, *, loc=None, ip=None
    ):
        self.divisor = divisor
        self.multiplier = multipler
        self.shift_right = shift_right
        self._loc = loc

    # called by host
    @staticmethod
    def create(divisor: Int32, *, loc=None, ip=None) -> "FastDivmod":
        """Construct the FastDivmod object, in host code.
        This precomputes some values based on the divisor and is computationally expensive.
        """
        p = Uint32(31 + find_log2(divisor))
        divisor_u32 = Uint32(divisor)
        multiplier = Uint32(((cutlass.Uint64(1) << p) + divisor_u32 - 1) // divisor_u32)
        shift_right = Uint32(p - 32)
        return FastDivmod(divisor, multiplier, shift_right, loc=loc, ip=ip)

    @cute.jit
    def div(self, dividend: Int32) -> Int32:
        return (
            Int32(umulhi(dividend, self.multiplier) >> self.shift_right)
            if self.divisor != 1
            else dividend
        )

    def divmod(self, dividend: Int32) -> Tuple[Int32, Int32]:
        quotient = self.div(dividend)
        remainder = dividend - quotient * self.divisor
        return quotient, remainder

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.divisor, self.multiplier, self.shift_right]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.divisor, self.multiplier, self.shift_right], self._values_pos
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return FastDivmod(*(tuple(obj_list)), loc=self._loc)

@dataclass
class ParamsBase:
    def __extract_mlir_values__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, cutlass.Constexpr)]
        values, self._values_pos = [], []
        for obj in non_constexpr_fields:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        constexpr_fields = {n: f for n, f in all_fields.items() if isinstance(f, cutlass.Constexpr)}
        non_constexpr_fields = {
            n: f for n, f in all_fields.items() if not isinstance(f, cutlass.Constexpr)
        }
        for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
            non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            values = values[n_items:]
        return self.__class__(**non_constexpr_fields, **constexpr_fields)


@dataclass
class TileSchedulerArguments(ParamsBase):
    num_block: Int32
    num_head: Int32
    num_batch: Int32
    seqlen_k: Int32
    headdim: Int32
    headdim_v: Int32
    total_q: Int32
    block_size: cutlass.Constexpr[int]
    mCuSeqlensQ: Optional[cute.Tensor] = None
    mSeqUsedQ: Optional[cute.Tensor] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    element_size: cutlass.Constexpr[int] = 2
    is_persistent: cutlass.Constexpr[bool] = False
    lpt: cutlass.Constexpr[bool] = False


class SingleTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block: Int32
        num_head: Int32
        num_batch: Int32

        @staticmethod
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileScheduler.Params":
            return SingleTileScheduler.Params(args.num_block, args.num_head, args.num_batch)

    def __init__(self, blk_coord: cute.Coord, *, loc=None, ip=None):
        self._blk_coord = blk_coord
        self._is_first_block = True
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return SingleTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileScheduler":
        blk_coord = cute.arch.block_idx()
        return SingleTileScheduler(blk_coord, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        return params.num_block, params.num_head, params.num_batch

    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        return cutlass.utils.WorkTileInfo(self._blk_coord, self._is_first_block)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._is_first_block = False

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self._blk_coord]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self._blk_coord], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return SingleTileScheduler(*(tuple(obj_list)), loc=self._loc)


class StaticPersistentTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block_divmod: FastDivmod
        num_head_divmod: FastDivmod
        total_blocks: Int32

        @staticmethod
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "StaticPersistentTileScheduler.Params":
            total_blocks = args.num_block * args.num_head * args.num_batch
            return StaticPersistentTileScheduler.Params(
                FastDivmod.create(args.num_block), FastDivmod.create(args.num_head), total_blocks
            )

    def __init__(
        self,
        num_block_divmod: FastDivmod,
        num_head_divmod: FastDivmod,
        total_blocks: Int32,
        tile_idx: Int32,
        *,
        loc=None,
        ip=None,
    ):
        self.num_block_divmod = num_block_divmod
        self.num_head_divmod = num_head_divmod
        self.total_blocks = total_blocks
        self._tile_idx = tile_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return StaticPersistentTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(params: Params, *, loc=None, ip=None) -> "StaticPersistentTileScheduler":
        tile_idx = cute.arch.block_idx()[0]
        return StaticPersistentTileScheduler(
            params.num_block_divmod,
            params.num_head_divmod,
            params.total_blocks,
            tile_idx,
            loc=loc,
            ip=ip,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        hardware_info = cutlass.utils.HardwareInfo()
        sm_count = hardware_info.get_device_multiprocessor_count()
        return (cutlass.min(sm_count, params.total_blocks), Int32(1), Int32(1))

    # @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        hn_idx, block_idx = self.num_block_divmod.divmod(self._tile_idx)
        batch_idx, head_idx = self.num_head_divmod.divmod(hn_idx)
        is_valid = self._tile_idx < self.total_blocks
        # if cute.arch.thread_idx()[0] == 0:
        #     cute.printf("TileScheduler: tile_idx=%d, hn_idx=%d, block_idx=%d, batch_idx=%d, head_idx=%d, is_valid=%d", self._tile_idx, hn_idx, block_idx, batch_idx, head_idx, is_valid)
        return cutlass.utils.WorkTileInfo(
            (Int32(block_idx), Int32(head_idx), Int32(batch_idx)), is_valid
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._tile_idx += cute.arch.grid_dim()[0]

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.num_block_divmod, self.num_head_divmod, self.total_blocks, self._tile_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.num_block_divmod, self.num_head_divmod, self.total_blocks, self._tile_idx],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return StaticPersistentTileScheduler(*(tuple(obj_list)), loc=self._loc)


class SingleTileLPTScheduler:
    @dataclass
    class Params(ParamsBase):
        total_blocks: Int32
        num_block_divmod: FastDivmod
        num_head_divmod: FastDivmod
        l2_minor_divmod: FastDivmod
        l2_major_divmod: FastDivmod
        l2_minor_residual_divmod: FastDivmod
        num_hb_quotient: Int32

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileLPTScheduler.Params":
            # cute.printf(args.num_block, args.num_head, args.num_batch, args.seqlen_k, args.headdim, args.headdim_v, args.total_q, args.block_size, args.qhead_per_kvhead_packgqa, args.element_size)
            size_one_kv_head = args.seqlen_k * (args.headdim + args.headdim_v) * args.element_size
            size_one_head = size_one_kv_head
            size_l2 = 50 * 1024 * 1024  # 40 MB for K & V
            # Swizzle is the size of each "section". Round swizzle to a power of 2
            # Need to be careful about the case where only one head will fit
            # swizzle is how many heads can fit in L2
            # swizzle = 1 if size_l2 < size_one_head else (size_l2 // size_one_head)
            # Seems faster if swizzle if a power of 2
            log2_floor = lambda n: 31 - clz(n)
            swizzle = 1 if size_l2 < size_one_head else (1 << log2_floor(size_l2 // size_one_head))
            # swizzle = 1 if size_l2 < size_one_head else (size_l2 // size_one_head)
            # If we're in the last section (called residual), we don't want to divide by
            # swizzle. Instead we want to divide by the remainder.
            num_hb_quotient = (args.num_head * args.num_batch) // swizzle
            num_hb_remainder = (args.num_head * args.num_batch) % swizzle
            return SingleTileLPTScheduler.Params(
                total_blocks=args.num_block * args.num_head * args.num_batch,
                num_block_divmod=FastDivmod.create(args.num_block),
                num_head_divmod=FastDivmod.create(args.num_head),
                l2_minor_divmod=FastDivmod.create(swizzle),
                l2_major_divmod=FastDivmod.create(swizzle * args.num_block),
                l2_minor_residual_divmod=FastDivmod.create(
                    max(num_hb_remainder, 1)
                ),  # don't divide by 0
                num_hb_quotient=Int32(num_hb_quotient),
            )

    def __init__(
        self,
        total_blocks: Int32,
        num_block_divmod: FastDivmod,
        num_head_divmod: FastDivmod,
        l2_minor_divmod: FastDivmod,
        l2_major_divmod: FastDivmod,
        l2_minor_residual_divmod: FastDivmod,
        num_hb_quotient: Int32,
        tile_idx: Int32,
        *,
        loc=None,
        ip=None,
    ):
        self.total_blocks = total_blocks
        self.num_block_divmod = num_block_divmod
        self.num_head_divmod = num_head_divmod
        self.l2_minor_divmod = l2_minor_divmod
        self.l2_major_divmod = l2_major_divmod
        self.l2_minor_residual_divmod = l2_minor_residual_divmod
        self.num_hb_quotient = num_hb_quotient
        self._tile_idx = tile_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return SingleTileLPTScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileLPTScheduler":
        tile_idx = cute.arch.block_idx()[0]
        return SingleTileLPTScheduler(
            params.total_blocks,
            params.num_block_divmod,
            params.num_head_divmod,
            params.l2_minor_divmod,
            params.l2_major_divmod,
            params.l2_minor_residual_divmod,
            params.num_hb_quotient,
            tile_idx,
            loc=loc,
            ip=ip,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        return (params.total_blocks, Int32(1), Int32(1))

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        # Implement LPT scheduling coordinate calculation
        bidhb, l2_mod = self.l2_major_divmod.divmod(self._tile_idx)
        # If we're in the last section (called residual), we don't want to divide by
        # swizzle. Instead we want to divide by the remainder.
        block, bidhb_residual = 0, 0
        if bidhb < self.num_hb_quotient:
            block, bidhb_residual = self.l2_minor_divmod.divmod(l2_mod)
        else:
            block, bidhb_residual = self.l2_minor_residual_divmod.divmod(l2_mod)
        bidhb_actual = bidhb * self.l2_minor_divmod.divisor + bidhb_residual
        batch_idx, head_idx = self.num_head_divmod.divmod(bidhb_actual)
        # Longest-processing-time-first
        block = self.num_block_divmod.divisor - 1 - block
        is_valid = self._tile_idx < self.total_blocks
        return cutlass.utils.WorkTileInfo(
            (Int32(block), Int32(head_idx), Int32(batch_idx)), is_valid
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        # Single tile scheduler - set to invalid tile_idx to indicate no more work
        self._tile_idx = self.total_blocks

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self.total_blocks,
            self.num_block_divmod,
            self.num_head_divmod,
            self.l2_minor_divmod,
            self.l2_major_divmod,
            self.l2_minor_residual_divmod,
            self.num_hb_quotient,
            self._tile_idx,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self.total_blocks,
                self.num_block_divmod,
                self.num_head_divmod,
                self.l2_minor_divmod,
                self.l2_major_divmod,
                self.l2_minor_residual_divmod,
                self.num_hb_quotient,
                self._tile_idx,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return SingleTileLPTScheduler(*(tuple(obj_list)), loc=self._loc)

@cute.jit
def warp_prefix_sum(val: cutlass.Int32, lane: Optional[cutlass.Int32] = None) -> cutlass.Int32:
    if cutlass.const_expr(lane is None):
        lane = cute.arch.lane_idx()
    # if cute.arch.thread_idx()[0] >= 128 and cute.arch.thread_idx()[0] < 128 + 32 and cute.arch.block_idx()[0] == 0: cute.printf("tidx = %d, val = %d", cute.arch.thread_idx()[0] % 32, val)
    for i in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
        offset = 1 << i
        # Very important that we set mask_and_clamp to 0
        partial_sum = cute.arch.shuffle_sync_up(val, offset=offset, mask_and_clamp=0)
        if lane >= offset:
            val += partial_sum
        # if cute.arch.thread_idx()[0] >= 128 and cute.arch.thread_idx()[0] < 128 + 32 and cute.arch.block_idx()[0] == 0: cute.printf("tidx = %d, partial_sum = %d, val = %d", cute.arch.thread_idx()[0] % 32, partial_sum, val)
    return val

class SingleTileVarlenScheduler:
    @dataclass
    class Params(ParamsBase):
        num_head: Int32
        num_batch: Int32
        total_q: Int32
        max_kvblock_in_l2: Int32
        block_size: cutlass.Constexpr[int]
        mCuSeqlensQ: Optional[cute.Tensor] = None
        mSeqUsedQ: Optional[cute.Tensor] = None
        qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
        lpt: cutlass.Constexpr[bool] = False

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileVarlenScheduler.Params":
            size_l2 = 50 * 1024 * 1024  # 50 MB for K & V
            max_kvblock_in_l2 = size_l2 // ((args.headdim + args.headdim_v) * args.element_size * args.block_size)
            return SingleTileVarlenScheduler.Params(
                num_head=args.num_head,
                num_batch=args.num_batch,
                total_q=args.total_q,
                max_kvblock_in_l2=max_kvblock_in_l2,
                block_size=args.block_size,
                mCuSeqlensQ=args.mCuSeqlensQ,
                mSeqUsedQ=args.mSeqUsedQ,
                qhead_per_kvhead_packgqa=args.qhead_per_kvhead_packgqa,
                lpt=args.lpt,
            )

    def __init__(
        self,
        num_head: Int32,
        num_batch: Int32,
        max_kvblock_in_l2: Int32,
        tile_idx: Int32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        block_size: cutlass.Constexpr[int] = 128,
        qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1,
        lpt: cutlass.Constexpr[bool] = False,
        *,
        loc=None,
        ip=None,
    ):
        self.num_head = num_head
        self.num_batch = num_batch
        self.max_kvblock_in_l2 = max_kvblock_in_l2
        self.mCuSeqlensQ = mCuSeqlensQ
        self.mSeqUsedQ = mSeqUsedQ
        assert self.mCuSeqlensQ is not None or self.mSeqUsedQ is not None, (
            "At least one of mCuSeqlensQ or mSeqUsedQ must be provided"
        )
        self.block_size = block_size
        self.qhead_per_kvhead_packgqa = qhead_per_kvhead_packgqa
        self.lpt = lpt
        self._tile_idx = tile_idx
        self._is_first_block = True
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return SingleTileVarlenScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileVarlenScheduler":
        tile_idx = cute.arch.block_idx()[0]
        return SingleTileVarlenScheduler(
            params.num_head,
            params.num_batch,
            params.max_kvblock_in_l2,
            tile_idx,
            mCuSeqlensQ=params.mCuSeqlensQ,
            mSeqUsedQ=params.mSeqUsedQ,
            block_size=params.block_size,
            qhead_per_kvhead_packgqa=params.qhead_per_kvhead_packgqa,
            lpt=params.lpt,
            loc=loc,
            ip=ip,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        total_blocks_max = (
            params.total_q + params.num_batch * (params.block_size - 1)
        ) // params.block_size
        return (total_blocks_max * params.num_head, Int32(1), Int32(1))

    @cute.jit
    def _get_num_m_blocks(self, lane: Int32, bidb_start: Int32) -> Int32:
        batch_idx = lane + bidb_start
        if cutlass.const_expr(self.mSeqUsedQ is not None):
            seqlen = Int32(0)
            if batch_idx < self.num_batch:
                seqlen = self.mSeqUsedQ[batch_idx]
        else:
            assert self.mCuSeqlensQ is not None
            cur_cu_seqlen = Int32(0)
            if batch_idx <= self.num_batch:
                cur_cu_seqlen = self.mCuSeqlensQ[batch_idx]
            next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
            seqlen = next_cu_seqlen - cur_cu_seqlen
        if cutlass.const_expr(self.qhead_per_kvhead_packgqa > 1):
            seqlen *= self.qhead_per_kvhead_packgqa
        return (
            cute.ceil_div(seqlen, self.block_size)
            if batch_idx < self.num_batch and lane < cute.arch.WARP_SIZE - 1
            else Int32(0)
        )

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        lane_idx = cute.arch.lane_idx()
        num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=0)
        num_m_blocks_cumulative = warp_prefix_sum(num_m_blocks, lane_idx)
        # Total number of blocks for the next 31 batches
        m_blocks_in_group = cute.arch.shuffle_sync(num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1)
        # Same for all lanes
        group_end_tile = m_blocks_in_group * self.num_head
        # if cute.arch.thread_idx()[0] == 128 + 31: cute.printf("SingleTileVarlenScheduler: tile_idx=%d, group_end_tile = %d, num_m_blocks=%d, num_m_blocks_cumulative = %d, m_blocks_in_group = %d", self._tile_idx, group_end_tile, num_m_blocks, num_m_blocks_cumulative, m_blocks_in_group)
        block, head_idx, batch_idx = Int32(0), Int32(0), Int32(0)
        next_tile_idx = self._tile_idx
        while group_end_tile <= next_tile_idx:
            batch_idx += cute.arch.WARP_SIZE - 1
            if batch_idx >= self.num_batch:
                batch_idx = Int32(self.num_batch)
                group_end_tile = next_tile_idx + 1
            else:
                num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=batch_idx)
                num_m_blocks_cumulative = warp_prefix_sum(num_m_blocks, lane_idx)
                m_blocks_in_group = cute.arch.shuffle_sync(
                    num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1
                )
                group_end_tile += m_blocks_in_group * self.num_head
        is_valid = False
        if batch_idx >= self.num_batch:
            block, head_idx, batch_idx = Int32(0), Int32(0), Int32(self.num_batch)
        else:
            group_start_tile = group_end_tile - m_blocks_in_group * self.num_head
            # if cute.arch.thread_idx()[0] == 128 + 31: cute.printf("SingleTileVarlenScheduler: tile_idx=%d, group_end_tile = %d, num_m_blocks=%d, batch_idx = %d", self._tile_idx, group_end_tile, num_m_blocks, batch_idx)
            # The next problem to process is the first one that does not have ending tile position
            # that is greater than or equal to tile index.
            batch_idx_in_group = cute.arch.popc(
                cute.arch.vote_ballot_sync(
                    group_start_tile + num_m_blocks_cumulative * self.num_head <= next_tile_idx
                )
            )
            batch_idx += batch_idx_in_group
            num_m_blocks_prev_lane = (
                0
                if batch_idx_in_group == 0
                else cute.arch.shuffle_sync(num_m_blocks_cumulative, batch_idx_in_group - 1)
            )
            num_m_blocks = cute.arch.shuffle_sync(num_m_blocks, batch_idx_in_group)
            mh_block = next_tile_idx - group_start_tile - num_m_blocks_prev_lane * self.num_head
            if cutlass.const_expr(self.lpt):
                # This is a version of the SingleTileLPTScheduler, complicated by the fact that
                # the seqlen can vary per batch.
                # TODO: is there any case where num_m_blocks is 0?
                # TODO: by right we should read the seqlen_kv but we're assuming seqlen_q == seqlen_k here
                # nheads_in_l2 = min(max(self.max_kvblock_in_l2 // num_m_blocks, 1), self.num_head)
                # Seems faster to have this be a power of 2
                nheads_in_l2 = 16 if num_m_blocks * 16 <= self.max_kvblock_in_l2 else (8 if num_m_blocks * 8 <= self.max_kvblock_in_l2 else (4 if num_m_blocks * 4 <= self.max_kvblock_in_l2 else (2 if num_m_blocks * 2 <= self.max_kvblock_in_l2 else 1)))
                nheads_in_l2 = min(nheads_in_l2, self.num_head)
                mh_in_l2 = nheads_in_l2 * num_m_blocks
                section_idx = mh_block // mh_in_l2
                l2_mod = mh_block - section_idx * mh_in_l2
                # Deal with tail section
                nheads_in_this_section = nheads_in_l2 if nheads_in_l2 * (section_idx + 1) <= self.num_head else self.num_head - section_idx * nheads_in_l2
                block = l2_mod // nheads_in_this_section
                head_idx_residual = l2_mod - block * nheads_in_this_section
                head_idx = section_idx * nheads_in_l2 + head_idx_residual
                block = num_m_blocks - 1 - block
            else:
                head_idx = mh_block // num_m_blocks
                block = mh_block - head_idx * num_m_blocks
            is_valid = self._is_first_block and batch_idx < self.num_batch
        # if cute.arch.thread_idx()[0] == 128: cute.printf("SingleTileVarlenScheduler: tile_idx=%d, batch_idx=%d, head_idx=%d, block=%d, is_valid = %d", self._tile_idx, batch_idx, head_idx, block, is_valid)
        return cutlass.utils.WorkTileInfo(
            (Int32(block), Int32(head_idx), Int32(batch_idx)), is_valid
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        # Single tile scheduler - set to invalid tile_idx to indicate no more work
        self._is_first_block = False

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self.num_head,
            self.num_batch,
            self.max_kvblock_in_l2,
            self._tile_idx,
            self.mCuSeqlensQ,
            self.mSeqUsedQ,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self.num_head,
                self.num_batch,
                self.max_kvblock_in_l2,
                self._tile_idx,
                self.mCuSeqlensQ,
                self.mSeqUsedQ,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return SingleTileVarlenScheduler(
            *(tuple(obj_list)),
            block_size=self.block_size,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead_packgqa,
            lpt=self.lpt,
            loc=self._loc,
        )


@dsl_user_op
def log2f(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "lg2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )

@dsl_user_op
def fmax(
    a: float | Float32, b: float | Float32, c: float | Float32 | None = None, *, loc=None, ip=None
) -> Float32:
    return Float32(
        nvvm.fmax(
            T.f32(),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
            loc=loc,
            ip=ip,
        )
    )

@cute.jit
def fmax_reduce(
    x: cute.TensorSSA, init_val: float | Float32 | None = None, arch: cutlass.Constexpr[int] = 80
) -> Float32:
    if cutlass.const_expr(arch < 100 or cute.size(x.shape) % 8 != 0):
        # if cutlass.const_expr(init_val is None):
        #     init_val = -cutlass.Float32.if
        # return x.reduce(cute.ReductionOp.MAX, init_val, 0)
        res = cute.make_fragment(x.shape, Float32)
        res.store(x)
        # local_max = [res[0], res[1]]
        # for i in cutlass.range_constexpr(2, cute.size(x.shape), 2):
        #     local_max[0] = fmax(local_max[0], res[i + 0])
        #     local_max[1] = fmax(local_max[1], res[i + 1])
        # local_max[0] = fmax(local_max[0], local_max[1])
        # return local_max[0] if cutlass.const_expr(init_val is None) else fmax(local_max[0], init_val)
        local_max = [res[0], res[1], res[2], res[3]]
        for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
            local_max[0] = fmax(local_max[0], res[i + 0])
            local_max[1] = fmax(local_max[1], res[i + 1])
            local_max[2] = fmax(local_max[2], res[i + 2])
            local_max[3] = fmax(local_max[3], res[i + 3])
        local_max[0] = fmax(local_max[0], local_max[1])
        local_max[2] = fmax(local_max[2], local_max[3])
        local_max[0] = fmax(local_max[0], local_max[2])
        return local_max[0] if cutlass.const_expr(init_val is None) else fmax(local_max[0], init_val)
    else:
        # [2025-06-15] x.reduce only seems to use 50% 3-input max and 50% 2-input max
        # We instead force the 3-input max.
        res = cute.make_fragment(x.shape, Float32)
        res.store(x)
        local_max = [
            fmax(init_val, res[0], res[1])
            if cutlass.const_expr(init_val is not None)
            else fmax(res[0], res[1]),
            fmax(res[2], res[3]),
            fmax(res[4], res[5]),
            fmax(res[6], res[7]),
        ]
        for i in cutlass.range_constexpr(8, cute.size(x.shape), 8):
            local_max[0] = fmax(local_max[0], res[i], res[i + 1])
            local_max[1] = fmax(local_max[1], res[i + 2], res[i + 3])
            local_max[2] = fmax(local_max[2], res[i + 4], res[i + 5])
            local_max[3] = fmax(local_max[3], res[i + 6], res[i + 7])
        local_max[0] = fmax(local_max[0], local_max[1])
        return fmax(local_max[0], local_max[2], local_max[3])

@cute.jit
def fadd_reduce(
    x: cute.TensorSSA, init_val: float | Float32 | None = None, arch: cutlass.Constexpr[int] = 80
) -> Float32:
    if cutlass.const_expr(arch < 100 or cute.size(x.shape) % 8 != 0):
        if cutlass.const_expr(init_val is None):
            init_val = Float32.zero
        return x.reduce(cute.ReductionOp.ADD, init_val, 0)
        # res = cute.make_fragment(x.shape, Float32)
        # res.store(x)
        # local_sum = [res[0], res[1], res[2], res[3]]
        # for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
        #     local_sum[0] += res[i + 0]
        #     local_sum[1] += res[i + 1]
        #     local_sum[2] += res[i + 2]
        #     local_sum[3] += res[i + 3]
        # local_sum[0] += local_sum[1]
        # local_sum[2] += local_sum[3]
        # local_sum[0] += local_sum[2]
        # return local_sum[0] if cutlass.const_expr(init_val is None) else local_sum[0] + init_val
    else:
        res = cute.make_fragment(x.shape, Float32)
        res.store(x)
        local_sum_0 = (
            cute.arch.add_packed_f32x2((init_val, 0.0), (res[0], res[1]))
            # cute.arch.add_packed_f32x2((init_val / 2, init_val / 2), (res[0], res[1]))
            if cutlass.const_expr(init_val is not None)
            else (res[0], res[1])
        )
        local_sum = [local_sum_0, (res[2], res[3]), (res[4], res[5]), (res[6], res[7])]
        for i in cutlass.range_constexpr(8, cute.size(x.shape), 8):
            local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], (res[i + 0], res[i + 1]))
            local_sum[1] = cute.arch.add_packed_f32x2(local_sum[1], (res[i + 2], res[i + 3]))
            local_sum[2] = cute.arch.add_packed_f32x2(local_sum[2], (res[i + 4], res[i + 5]))
            local_sum[3] = cute.arch.add_packed_f32x2(local_sum[3], (res[i + 6], res[i + 7]))
        local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[1])
        local_sum[2] = cute.arch.add_packed_f32x2(local_sum[2], local_sum[3])
        local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[2])
        return local_sum[0][0] + local_sum[0][1]
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

@dsl_user_op
def exp2f_asm(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def exp2f(x: cute.TensorSSA | Float32) -> cute.TensorSSA | Float32:
    """exp2f calculation for both vector and scalar.
    :param x: input value
    :type x: cute.TensorSSA or Float32
    :return: exp2 value
    :rtype: cute.TensorSSA or Float32
    """
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        res = cute.make_fragment(x.shape, Float32)
        res.store(x)
        for i in cutlass.range_constexpr(cute.size(x.shape)):
            res[i] = cute.arch.exp2(res[i])
        return res.load()
    else:
        return cute.arch.exp2(x)

class Softmax:
    def __init__(
        self,
        scale_log2: Float32,
        num_rows: cutlass.Constexpr[int],
        arch: cutlass.Constexpr[int] = 80,
    ):
        self.scale_log2 = scale_log2
        self.row_max = cute.make_fragment(num_rows, Float32)
        self.row_sum = cute.make_fragment_like(self.row_max)
        self.arch = arch

    def reset(self) -> None:
        self.row_max.fill(-Float32.inf)
        self.row_sum.fill(0.0)

    def _compute_row_max(
        self, acc_S_row: cute.TensorSSA, init_val: float | Float32 | None = None
    ) -> Float32:
        return fmax_reduce(acc_S_row, init_val, arch=self.arch)

    def _compute_row_sum(
        self, acc_S_row_exp: cute.TensorSSA, init_val: float | Float32 | None = None
    ) -> Float32:
        return fadd_reduce(acc_S_row_exp, init_val, arch=self.arch)

    @cute.jit
    def online_softmax(
        self,
        acc_S: cute.Tensor,
        is_first: cutlass.Constexpr[bool] = False,
        check_inf: cutlass.Constexpr[bool] = True,
    ) -> cute.Tensor:
        """Apply online softmax and return the row_scale to rescale O.

        :param acc_S: acc_S tensor
        :type acc_S: cute.Tensor
        :param is_first: is first n_block
        :type is_first: cutlass.Constexpr
        """
        # Change acc_S to M,N layout view.
        acc_S_mn = make_acc_tensor_mn_view(acc_S)
        row_scale = cute.make_fragment_like(self.row_max, Float32)
        # Each iteration processes one row of acc_S
        for r in cutlass.range_constexpr(cute.size(self.row_max)):
            acc_S_row = acc_S_mn[r, None].load()  # (n_block_size)
            row_max_cur = self._compute_row_max(
                acc_S_row,
                init_val=self.row_max[r] if cutlass.const_expr(not is_first) else None,
            )
            row_max_cur = warp_reduce(row_max_cur, cute.arch.fmax, width=4)
            if cutlass.const_expr(check_inf):
                row_max_cur = 0.0 if row_max_cur == -Float32.inf else row_max_cur
            if cutlass.const_expr(is_first):
                row_max_cur_scaled = row_max_cur * self.scale_log2
                acc_S_row_exp = exp2f(acc_S_row * self.scale_log2 - row_max_cur_scaled)
                acc_S_row_sum = self._compute_row_sum(acc_S_row_exp)
                row_scale[r] = 1.0
            else:
                row_max_prev = self.row_max[r]
                row_max_cur_scaled = row_max_cur * self.scale_log2
                acc_S_row_exp = exp2f(acc_S_row * self.scale_log2 - row_max_cur_scaled)
                # row_scale[r] = utils.exp2f(row_max_prev * self.scale_log2 - row_max_cur_scaled)
                row_scale[r] = exp2f((row_max_prev - row_max_cur) * self.scale_log2)
                acc_S_row_sum = (
                    self._compute_row_sum(acc_S_row_exp, init_val=self.row_sum[r] * row_scale[r])
                )
            self.row_max[r] = row_max_cur
            self.row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None].store(acc_S_row_exp)
        return row_scale

    @cute.jit
    def finalize(self, final_scale: Float32 = 1.0) -> cute.Tensor:
        """Finalize the online softmax by computing the scale and logsumexp."""
        # quad reduction for row_sum as we didn't do it during each iteration of online softmax
        self.row_sum.store(warp_reduce(self.row_sum.load(), operator.add, width=4))
        row_scale = cute.make_fragment_like(self.row_max, Float32)
        for r in cutlass.range_constexpr(cute.size(self.row_sum)):
            # if row_sum is zero or nan, set acc_O_mn_row to 1.0
            acc_O_mn_row_is_zero_or_nan = (
                self.row_sum[r] == 0.0 or self.row_sum[r] != self.row_sum[r]
            )
            row_scale[r] = (
                cute.arch.rcp_approx(self.row_sum[r] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            ) * final_scale
            row_sum_cur = self.row_sum[r]
            LN2 = math.log(2.0)
            self.row_sum[r] = (
                (self.row_max[r] * self.scale_log2 + log2f(row_sum_cur)) * LN2
                if not acc_O_mn_row_is_zero_or_nan
                else -Float32.inf
            )
        return row_scale

    @cute.jit
    def rescale_O(self, acc_O: cute.Tensor, row_scale: cute.Tensor) -> None:
        """Scale each row of acc_O by the given scale tensor.
        :param acc_O: input tensor
        :type acc_O: cute.Tensor
        :param row_scale: row_scale tensor
        :type row_scale: cute.Tensor
        """
        acc_O_mn = make_acc_tensor_mn_view(acc_O)
        assert cute.size(row_scale) == cute.size(acc_O_mn, mode=[0])
        for r in cutlass.range_constexpr(cute.size(row_scale)):
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])

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


@dataclass(frozen=True)
class BlockInfo:
    m_block_size: cutlass.Constexpr[int]
    n_block_size: cutlass.Constexpr[int]
    is_causal: cutlass.Constexpr[bool]
    is_local: cutlass.Constexpr[bool] = False
    window_size_left: Optional[cutlass.Int32] = None
    window_size_right: Optional[cutlass.Int32] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1

    @cute.jit
    def get_n_block_min_max(
        self, seqlen_info: SeqlenInfo, m_block: cutlass.Int32
    ) -> Tuple[cutlass.Int32, cutlass.Int32]:
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.n_block_size)
        if cutlass.const_expr(
            self.is_causal or (self.is_local and self.window_size_right is not None)
        ):
            m_idx_max = (m_block + 1) * self.m_block_size
            if cutlass.const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_right = n_idx if cutlass.const_expr(self.is_causal) else n_idx + self.window_size_right
            n_block_max = min(n_block_max, cute.ceil_div(n_idx_right, self.n_block_size))
        n_block_min = 0
        if cutlass.const_expr(self.is_local and self.window_size_left is not None):
            m_idx_min = m_block * self.m_block_size
            if cutlass.const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
            n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_left = n_idx - self.window_size_left
            n_block_min = cutlass.max(n_idx_left // self.n_block_size, 0)
        return n_block_min, n_block_max

    @cute.jit
    def get_n_block_min_causal_local_mask(
        self,
        seqlen_info: SeqlenInfo,
        m_block: cutlass.Int32,
        n_block_min: cutlass.Int32,
    ) -> cutlass.Int32:
        """If we have separate iterations with causal or local masking at the start, where do we stop"""
        m_idx_min = m_block * self.m_block_size
        if cutlass.const_expr(self.qhead_per_kvhead_packgqa > 1):
            m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
        n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
        n_idx_right = (
            n_idx
            if cutlass.const_expr(not self.is_local or self.window_size_right is None)
            else n_idx + self.window_size_right
        )
        return cutlass.max(n_block_min, n_idx_right // self.n_block_size)

    @cute.jit
    def get_n_block_min_before_local_mask(
        self,
        seqlen_info: SeqlenInfo,
        m_block: cutlass.Int32,
        n_block_min: cutlass.Int32,
    ) -> cutlass.Int32:
        """If we have separate iterations with local masking at the end, where do we stop the non-masked iterations"""
        if cutlass.const_expr(not self.is_local or self.window_size_left is None):
            return n_block_min
        else:
            m_idx_max = (m_block + 1) * self.m_block_size
            if cutlass.const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_left = n_idx - self.window_size_left
            return cutlass.max(n_block_min, cute.ceil_div(n_idx_left, self.n_block_size))

@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


class PackGQA:
    def __init__(
        self,
        m_block_size: cutlass.Constexpr[int],
        head_dim_padded: cutlass.Constexpr[int],
        check_hdim_oob: cutlass.Constexpr[bool],
        qhead_per_kvhead: cutlass.Constexpr[bool],
    ):
        self.m_block_size = m_block_size
        self.head_dim_padded = head_dim_padded
        self.check_hdim_oob = check_hdim_oob
        self.qhead_per_kvhead = qhead_per_kvhead

    @cute.jit
    def compute_ptr(
        self,
        tensor: cute.Tensor,
        cRows: cute.Tensor,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        threads_per_row: cutlass.Constexpr[int],
        num_threads: cutlass.Constexpr[int],
    ):
        num_ptr_per_thread = cute.ceil_div(cute.size(cRows), threads_per_row)
        tPrPtr = cute.make_fragment(num_ptr_per_thread, cutlass.Int64)
        for i in cutlass.range_constexpr(num_ptr_per_thread):
            row = i * num_threads + cRows[tidx % threads_per_row][0]
            idx = block * self.m_block_size + row
            m_idx = idx // self.qhead_per_kvhead
            h_idx = idx - m_idx * self.qhead_per_kvhead
            tPrPtr[i] = elem_pointer(tensor, ((h_idx, m_idx),)).toint()
        return tPrPtr

    @cute.jit
    def load_Q(
        self,
        mQ: cute.Tensor,  # ((qhead_per_kvhead, seqlen_q), headdim)
        sQ: cute.Tensor,  # (m_block_size, head_dim_padded)
        gmem_tiled_copy: cute.TiledCopy,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
        cQ = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tQsQ = gmem_thr_copy.partition_D(sQ)
        tQcQ = gmem_thr_copy.partition_S(cQ)
        t0QcQ = gmem_thr_copy.get_slice(0).partition_S(cQ)
        tQpQ = predicate_k(tQcQ, limit=mQ.shape[1])
        tQcQ_row = tQcQ[0, None, 0]
        threads_per_row = gmem_tiled_copy.layout_tv_tiled.shape[0][0]
        assert cute.arch.WARP_SIZE % threads_per_row == 0, "threads_per_row must divide WARP_SIZE"
        num_threads = gmem_tiled_copy.size
        tPrQPtr = self.compute_ptr(mQ[None, 0], tQcQ_row, tidx, block, threads_per_row, num_threads)
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            q_ptr_i64 = shuffle_sync(
                tPrQPtr[m // threads_per_row], m % threads_per_row, width=threads_per_row
            )
            q_gmem_ptr = cute.make_ptr(
                mQ.element_type, q_ptr_i64, cute.AddressSpace.gmem, assumed_align=16
            )
            if (
                t0QcQ[0, m, 0][0]
                < seqlen * self.qhead_per_kvhead - block * self.m_block_size - tQcQ_row[0][0]
            ):
                mQ_cur = cute.make_tensor(q_gmem_ptr, (self.head_dim_padded,))
                elems_per_load = cute.size(tQsQ.shape[0][0])
                mQ_cur_copy = cute.tiled_divide(mQ_cur, (elems_per_load,))
                for k in cutlass.range_constexpr(cute.size(tQsQ.shape[2])):
                    ki = tQcQ[0, 0, k][1] // elems_per_load
                    cute.copy(
                        gmem_thr_copy,
                        mQ_cur_copy[None, ki],
                        tQsQ[None, m, k],
                        pred=tQpQ[None, m, k] if cutlass.const_expr(self.check_hdim_oob) else None,
                    )
            # We don't need to clear the sQ smem tiles since we'll only write out the valid outputs

    @cute.jit
    def store_LSE(
        self,
        mLSE: cute.Tensor,  # (qhead_per_kvhead, seqlen_q)
        tLSErLSE: cute.Tensor,  # (m_block_size, head_dim_padded)
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        thr_mma = tiled_mma.get_slice(tidx)
        caccO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        taccOcO = thr_mma.partition_C(caccO)
        taccOcO_row = make_acc_tensor_mn_view(taccOcO)[None, 0]
        assert cute.size(tLSErLSE) == cute.size(taccOcO_row)
        threads_per_row = tiled_mma.tv_layout_C.shape[0][0]
        assert cute.arch.WARP_SIZE % threads_per_row == 0, "threads_per_row must divide WARP_SIZE"
        assert cute.size(tLSErLSE) <= threads_per_row
        num_threads = tiled_mma.size
        tPrLSEPtr = self.compute_ptr(mLSE, taccOcO_row, tidx, block, threads_per_row, num_threads)
        for m in cutlass.range_constexpr(cute.size(tLSErLSE)):
            lse_ptr_i64 = shuffle_sync(
                tPrLSEPtr[m // threads_per_row],
                m % threads_per_row,
                width=threads_per_row,
            )
            lse_gmem_ptr = cute.make_ptr(
                mLSE.element_type, lse_ptr_i64, cute.AddressSpace.gmem, assumed_align=4
            )
            row = block * self.m_block_size + taccOcO_row[m][0]
            # Only the thread corresponding to column 0 writes out the lse to gmem
            if taccOcO[0][1] == 0 and row < seqlen * self.qhead_per_kvhead:
                mLSE_copy = cute.make_tensor(lse_gmem_ptr, (1,))
                mLSE_copy[0] = tLSErLSE[m]

    @cute.jit
    def store_O(
        self,
        mO: cute.Tensor,  # ((qhead_per_kvhead, seqlen_q), headdim)
        tOrO: cute.Tensor,  # (m_block_size, head_dim_padded) split across threads according to gmem_tiled_copy
        gmem_tiled_copy: cute.TiledCopy,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tOcO = gmem_thr_copy.partition_S(cO)
        t0OcO = gmem_thr_copy.get_slice(0).partition_S(cO)
        tOpO = predicate_k(tOcO, limit=mO.shape[1])
        tOcO_row = tOcO[0, None, 0]
        threads_per_row = gmem_tiled_copy.layout_tv_tiled.shape[0][0]
        assert cute.arch.WARP_SIZE % threads_per_row == 0, "threads_per_row must divide WARP_SIZE"
        num_threads = gmem_tiled_copy.size
        tPrOPtr = self.compute_ptr(mO[None, 0], tOcO_row, tidx, block, threads_per_row, num_threads)
        for m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
            o_ptr_i64 = shuffle_sync(
                tPrOPtr[m // threads_per_row], m % threads_per_row, width=threads_per_row
            )
            o_gmem_ptr = cute.make_ptr(
                mO.element_type, o_ptr_i64, cute.AddressSpace.gmem, assumed_align=16
            )
            if (
                t0OcO[0, m, 0][0]
                < seqlen * self.qhead_per_kvhead - block * self.m_block_size - tOcO_row[0][0]
            ):
                mO_cur = cute.make_tensor(o_gmem_ptr, (self.head_dim_padded,))
                elems_per_load = cute.size(tOrO.shape[0][0])
                mO_cur_copy = cute.tiled_divide(mO_cur, (elems_per_load,))
                for k in cutlass.range_constexpr(cute.size(tOrO.shape[2])):
                    ki = tOcO[0, 0, k][1] // elems_per_load
                    cute.copy(
                        gmem_thr_copy,
                        tOrO[None, m, k],
                        mO_cur_copy[None, ki],
                        pred=tOpO[None, m, k] if cutlass.const_expr(self.check_hdim_oob) else None,
                    )




class NamedBarrierFwd(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    WarpSchedulerWG1 = enum.auto()
    WarpSchedulerWG2 = enum.auto()
    WarpSchedulerWG3 = enum.auto()
    PFull = enum.auto()
    PEmpty = enum.auto()

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



def get_smem_store_atom(
    arch: cutlass.Constexpr[int], element_type: Type[cute.Numeric]
) -> cute.CopyAtom:
    if cutlass.const_expr(arch < 90):
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            element_type,
            num_bits_per_copy=2 * element_type.width,
        )
    else:
        return cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4),
            element_type,
        )

def make_tiled_copy_C(copy_atom: cute.CopyAtom, tiled_mma: cute.TiledMma) -> cute.TiledCopy:
    return cute.make_tiled_copy(
        copy_atom,
        layout_tv=tiled_mma.tv_layout_C_tiled,
        tiler_mn=(tiled_mma.get_tile_size(0), tiled_mma.get_tile_size(1)),
    )
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



def transpose_view(a: cute.Tensor) -> cute.Tensor:
    """Transpose the first two dimensions of a tensor on smem."""
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


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


class FlashAttentionForwardBase:

    arch: int = 80

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        pack_gqa: bool = True,
        m_block_size: int = 128,
        n_block_size: int = 128,
        num_stages: int = 1,
        num_threads: int = 128,
        Q_in_regs: bool = False,
    ):
        """Initializes the configuration for a flash attention kernel.

        All contiguous dimensions must be at least 16 bytes aligned, which means that the head dimension
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
        hdim_multiple_of = 16
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        # Can save registers (and hence be faster) if we don't have to check hdim predication
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal = is_causal
        self.is_local = is_local
        self.pack_gqa = pack_gqa
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.num_threads = num_threads
        self.num_stages = num_stages
        self.Q_in_regs = Q_in_regs

    @staticmethod
    def can_implement(
        dtype, head_dim, head_dim_v, m_block_size, n_block_size, num_stages, num_threads, is_causal,
        Q_in_regs=False
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
        smem_usage_Q = m_block_size * head_dim * 2
        smem_usage_K = n_block_size * head_dim * num_stages * 2
        smem_usage_V = n_block_size * head_dim_v * num_stages * 2
        smem_usage_QV = (smem_usage_Q + smem_usage_V) if not Q_in_regs else max(smem_usage_Q, smem_usage_V)
        smem_usage = smem_usage_QV + smem_usage_K
        # TODO: sm86 and sm89
        smem_capacity = sm80_utils_basic.SMEM_CAPACITY["sm80"]
        if smem_usage > smem_capacity:
            return False
        # Check if twice the block size is divisible by the number of threads
        if (m_block_size * 2) % num_threads != 0:
            return False
        return True

    def _check_type(
        self,
        mQ_type: Type[cutlass.Numeric],
        mK_type: Type[cutlass.Numeric],
        mV_type: Type[cutlass.Numeric],
        mO_type: Type[cutlass.Numeric],
        mLSE_type: Type[cutlass.Numeric] | None,
        #mCuSeqlensQ_type: Type[cutlass.Numeric] | None,
        #mCuSeqlensK_type: Type[cutlass.Numeric] | None,
        #mSeqUsedQ_type: Type[cutlass.Numeric] | None,
        #mSeqUsedK_type: Type[cutlass.Numeric] | None,
    ):
        # Get the data type and check if it is fp16 or bf16
        if const_expr(not (mQ_type == mK_type == mV_type == mO_type)):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mLSE_type not in [None, cutlass.Float32]):
            raise TypeError("LSE tensor must be Float32")
        #if const_expr(mCuSeqlensQ_type not in [None, cutlass.Int32]):
        #    raise TypeError("cu_seqlens_q tensor must be Int32")
        #if const_expr(mCuSeqlensK_type not in [None, cutlass.Int32]):
        #     raise TypeError("cu_seqlens_k tensor must be Int32")
        # if const_expr(mSeqUsedQ_type not in [None, cutlass.Int32]):
        #     raise TypeError("seqused_q tensor must be Int32")
        # if const_expr(mSeqUsedK_type not in [None, cutlass.Int32]):
        #     raise TypeError("seqused_k tensor must be Int32")
        assert mQ_type == self.dtype

    def _setup_attributes(self):
        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory layout: Q/K/V
        # ///////////////////////////////////////////////////////////////////////////////
        sQ_layout_atom, sK_layout_atom, sV_layout_atom, sO_layout_atom, sP_layout_atom = self._get_smem_layout_atom()
        self.sQ_layout = cute.tile_to_shape(
            sQ_layout_atom, (self.m_block_size, self.head_dim_padded), (0, 1),
        )
        self.sK_layout = cute.tile_to_shape(
            sK_layout_atom, (self.n_block_size, self.head_dim_padded, self.num_stages), (0, 1, 2),
        )
        self.sV_layout = cute.tile_to_shape(
            sV_layout_atom, (self.n_block_size, self.head_dim_v_padded, self.num_stages), (0, 1, 2),
        )
        self.sO_layout = cute.tile_to_shape(
            sO_layout_atom, (self.m_block_size, self.head_dim_v_padded), (0, 1),
        )
        if const_expr(sP_layout_atom is not None):
            self.sP_layout = cute.tile_to_shape(
                sP_layout_atom, (self.m_block_size, self.n_block_size), (0, 1),
            )
        else:
            self.sP_layout = None

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
        # tQ_layout and tK_layout: thread layout for QK load
        tQK_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        assert self.num_Q_load_threads % tQK_shape_dim_1 == 0, "num_threads must be divisible by tQK_shape_dim_1"
        assert self.num_producer_threads % tQK_shape_dim_1 == 0, "num_threads must be divisible by tQK_shape_dim_1"
        tQ_layout = cute.make_ordered_layout(
            (self.num_Q_load_threads // tQK_shape_dim_1, tQK_shape_dim_1), order=(1, 0),
        )
        tK_layout = cute.make_ordered_layout(
            (self.num_producer_threads // tQK_shape_dim_1, tQK_shape_dim_1), order=(1, 0),
        )
        # So that we don't have to check if we overshoot kBlockM when we load Q
        assert self.m_block_size % tQ_layout.shape[0] == 0
        tV_shape_dim_1 = sV_layout_atom.outer.shape[1] // async_copy_elems
        tV_layout = cute.make_ordered_layout(
            (self.num_producer_threads // tV_shape_dim_1, tV_shape_dim_1), order=(1, 0),
        )
        # TODO: need a different layout for O if O dtype is not the same as V dtype
        # tO_layout: thread layout for O store
        tO_layout = cute.make_ordered_layout(
            (self.num_epilogue_threads // tV_shape_dim_1, tV_shape_dim_1), order=(1, 0),
        )
        # So that we don't have to check if we overshoot kBlockM when we store O
        assert self.m_block_size % tO_layout.shape[0] == 0

        # Value layouts for copies
        vQKV_layout = cute.make_layout((1, async_copy_elems))
        vO_layout = vQKV_layout

        self.gmem_tiled_copy_Q = cute.make_tiled_copy_tv(atom_async_copy, tQ_layout, vQKV_layout)
        self.gmem_tiled_copy_K = cute.make_tiled_copy_tv(atom_async_copy, tK_layout, vQKV_layout)
        self.gmem_tiled_copy_V = cute.make_tiled_copy_tv(atom_async_copy, tV_layout, vQKV_layout)
        # gmem_tiled_copy_O: tiled copy for O store
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tO_layout, vO_layout)

    def _get_smem_layout_atom(self):
        raise NotImplementedError()

    def _get_tiled_mma(self):
        raise NotImplementedError()

    def _get_shared_storage_cls(self):
        raise NotImplementedError()

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        softcap: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        """Configures and launches the flash attention kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(_, _, _, 1)
        """
        raise NotImplementedError()

    @cute.jit
    def epilogue(
        self,
        acc_O: cute.Tensor,
        lse: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sO: cute.Tensor,
        seqlen: SeqlenInfo,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        m_block: cutlass.Int32,
        head_idx: cutlass.Int32,
        batch_idx: cutlass.Int32,
    ):
        # store acc_O
        rO = cute.make_fragment_like(acc_O, self.dtype)
        rO.store(acc_O.load().to(self.dtype))
        # Make sure all threads have finished reading V
        cute.arch.barrier(barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_epilogue_threads)
        smem_copy_atom_O = utils.get_smem_store_atom(self.arch, self.dtype)
        smem_thr_copy_O = utils.make_tiled_copy_C(smem_copy_atom_O, tiled_mma).get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        # copy acc O from rmem to smem with the smem copy atom
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
        pack_gqa = PackGQA(self.m_block_size, self.head_dim_padded, self.check_hdim_oob, self.qhead_per_kvhead)

        # Write LSE from rmem -> gmem
        if const_expr(mLSE is not None):
            if const_expr(not seqlen.has_cu_seqlens_q):
                mLSE_cur = mLSE[None, head_idx, batch_idx]
            else:
                offset = seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
                mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx])
            if const_expr(not self.pack_gqa):
                gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_block,))
                gLSE_expanded_layout = cute.append(
                    gLSE.layout, cute.make_layout((self.head_dim_v_padded,), stride=(0,))
                )
                gLSE_expanded = cute.make_tensor(gLSE.iterator, gLSE_expanded_layout)
                thr_mma = tiled_mma.get_slice(tidx)
                taccOgLSE = utils.make_acc_tensor_mn_view(thr_mma.partition_C(gLSE_expanded))
                assert cute.size(taccOgLSE, mode=[0]) == cute.size(lse)
                taccOcO = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cO))
                t0accOcO = utils.make_acc_tensor_mn_view(thr_mma.get_slice(0).partition_C(cO))
                # Only the thread corresponding to column 0 writes out the lse to gmem
                if taccOcO[0][1] == 0:
                    for m in cutlass.range_constexpr(cute.size(taccOgLSE.shape[1])):
                        if t0accOcO[m, 0][0] < seqlen.seqlen_q - m_block * self.m_block_size - taccOcO[0][0]:
                            taccOgLSE[m, 0] = lse[m]
            else:
                pack_gqa.store_LSE(mLSE_cur, lse, tiled_mma, tidx, m_block, seqlen.seqlen_q)

        if const_expr(not seqlen.has_cu_seqlens_q):
            mO_cur = mO[None, None, head_idx, batch_idx]
        else:
            offset = seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
            mO_cur = cute.domain_offset((offset, 0), mO[None, None, head_idx])
        # thr_mma = tiled_mma.get_slice(tidx)
        # taccOgO = thr_mma.partition_C(gO)
        # cute.autovec_copy(rO, taccOgO)
        # sync to make sure all smem stores are done
        if const_expr(self.use_tma_O):
            # ensure smem writes are visible to TMA
            cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
            cute.arch.barrier_arrive(barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE)
            gO = cute.local_tile(mO_cur, (self.m_block_size, self.head_dim_v_padded), (m_block, 0))
            tOsO, tOgO = cpasync.tma_partition(
                tma_atom_O,
                0,
                cute.make_layout(1),
                cute.group_modes(sO, 0, 2),
                cute.group_modes(gO, 0, 2),
            )
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if warp_idx == 4:
                cute.arch.barrier(barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE)
                cute.copy(tma_atom_O, tOsO, tOgO)
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
        else:
            cute.arch.barrier(barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_epilogue_threads)
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            tOsO = gmem_thr_copy_O.partition_S(sO)
            tOrO = cute.make_fragment_like(tOsO, self.dtype)
            # load acc O from smem to rmem for wider vectorization
            cute.autovec_copy(tOsO, tOrO)
            if const_expr(not self.pack_gqa):
                gO = cute.local_tile(mO_cur, (self.m_block_size, self.head_dim_v_padded), (m_block, 0))
                tOgO = gmem_thr_copy_O.partition_D(gO)
                tOcO = gmem_thr_copy_O.partition_S(cO)
                t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
                tOpO = predicate_k(tOcO, limit=mO.shape[1])
                # copy acc O from rmem to gmem
                for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                    if t0OcO[0, rest_m, 0][0] < seqlen.seqlen_q - m_block * self.m_block_size - tOcO[0][0]:
                        cute.copy(
                            gmem_tiled_copy_O,
                            tOrO[None, rest_m, None],
                            tOgO[None, rest_m, None],
                            pred=tOpO[None, rest_m, None] if const_expr(self.check_hdim_v_oob) else None,
                        )
            else:
                pack_gqa.store_O(mO_cur, tOrO, gmem_tiled_copy_O, tidx, m_block, seqlen.seqlen_q)

    @cute.jit
    def advance_pipeline(self, pipeline_index):
        return pipeline_index + 1 if pipeline_index < self.num_stages - 1 else 0

    @cute.jit
    def load_Q(
        self,
        gmem_thr_copy: cute.TiledCopy,
        gQ: cute.Tensor,
        sQ: cute.Tensor,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
        headdim: cutlass.Int32,
    ):
        tQsQ, tQgQ = gmem_thr_copy.partition_D(sQ), gmem_thr_copy.partition_S(gQ)
        cQ = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tQcQ = gmem_thr_copy.partition_S(cQ)
        t0QcQ = gmem_thr_copy.get_slice(0).partition_S(cQ)
        tQpQ = predicate_k(tQcQ, limit=headdim)
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            # Instead of using tQcQ, we using t0QcQ and subtract the offset from the limit
            # (seqlen - block * kBlockM). This is because the entries of t0QcQ are known at compile time.
            if t0QcQ[0, m, 0][0] < seqlen - block * self.m_block_size - tQcQ[0][0]:
                cute.copy(
                    gmem_thr_copy,
                    tQgQ[None, m, None],
                    tQsQ[None, m, None],
                    pred=tQpQ[None, m, None] if const_expr(self.check_hdim_oob) else None,
                )
            # We don't need to clear the sQ smem tiles since we'll only write out the valid outputs

    @cute.jit
    def load_K(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
        tKcK: cute.Tensor,
        t0KcK: cute.Tensor,
        tKpK: cute.Tensor,
        block: cutlass.Int32,
        smem_pipe_write: cutlass.Int32,
        seqlen: cutlass.Int32,
        need_predicates: cutlass.Constexpr,
    ):
        # Do we need to check if we overshoot kBlockN when we load K?
        is_even_n_smem_k = self.n_block_size % gmem_tiled_copy.tiler_mn[0].shape == 0
        if const_expr(need_predicates or not is_even_n_smem_k):
            # Instead of using tKcK, we using t0KcK and subtract the offset from the limit
            # (seqlen - block * kBlockN). This is because the entries of t0KcK are known at compile time.
            if const_expr(is_even_n_smem_k):
                seqlen_limit = seqlen - block * self.n_block_size
            else:
                if const_expr(not need_predicates):
                    seqlen_limit = self.n_block_size
                else:
                    seqlen_limit = cutlass.min(seqlen - block * self.n_block_size, self.n_block_size)
            seqlen_limit -= tKcK[0][0]
            for n in cutlass.range_constepxr(cute.size(tKsK.shape[1])):
                if t0KcK[0, n, 0][0] < seqlen_limit:
                    cute.copy(
                        gmem_tiled_copy,
                        tKgK[None, n, None, block],
                        tKsK[None, n, None, smem_pipe_write if const_expr(self.num_stages > 1) else 0],
                        pred=tKpK[None, n, None] if const_expr(self.check_hdim_oob) else None,
                    )
                # We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
        else:
            cute.copy(
                gmem_tiled_copy,
                tKgK[None, None, None, block],
                tKsK[None, None, None, smem_pipe_write if const_expr(self.num_stages > 1) else 0],
                pred=tKpK if const_expr(self.check_hdim_oob) else None,
            )

    @cute.jit
    def load_V(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        tVgV: cute.Tensor,
        tVsV: cute.Tensor,
        tVcV: cute.Tensor,
        t0VcV: cute.Tensor,
        tVpV: cute.Tensor,
        block: cutlass.Int32,
        smem_pipe_write: cutlass.Int32,
        seqlen: cutlass.Int32,
        need_predicates: cutlass.Constexpr,
    ):
        # Do we need to check if we overshoot kBlockN when we load V?
        is_even_n_smem_v = self.n_block_size % gmem_tiled_copy.tiler_mn[0].shape == 0
        if const_expr(need_predicates or not is_even_n_smem_v):
            for n in cutlass.range_constepxr(cute.size(tVsV.shape[1])):
                # If kBlockN doesn't evenly divide the tiled copy, only the last `n` needs to be checked
                if is_even_n_smem_v or n < cute.size(tVsV.shape[1]) - 1 or tVcV[0, n, 0][0] < self.n_block_size:
                    predicate = tVpV[None, n, None] if const_expr(self.check_hdim_v_oob) else None
                    if const_expr(need_predicates):
                        seqlen_limit = seqlen - block * self.n_block_size - tVcV[0][0]
                        predicate_n = t0VcV[0, n, 0][0] < seqlen_limit
                        predicate = cute.make_fragment_like(tVpV[None, 0, None])
                        for k in cutlass.range_constepxr(cute.size(predicate.shape[1])):
                            for i in cutlass.range_constepxr(cute.size(predicate.shape[0])):
                                predicate[i, k] = (tVpV[i, n, k] if const_expr(self.check_hdim_v_oob) else True) and predicate_n
                    cute.copy(
                        gmem_tiled_copy,
                        tVgV[None, n, None, block],
                        tVsV[None, n, None, smem_pipe_write if const_expr(self.num_stages > 1) else 0],
                        pred=predicate,
                    )
        else:
            cute.copy(
                gmem_tiled_copy,
                tVgV[None, None, None, block],
                tVsV[None, None, None, smem_pipe_write if const_expr(self.num_stages > 1) else 0],
                pred=tVpV if const_expr(self.check_hdim_v_oob) else None,
            )


class FlashAttentionForwardSm80(FlashAttentionForwardBase):

    def _get_smem_layout_atom(self):
        sQ_layout_atom = get_smem_layout_atom(self.dtype, self.head_dim_padded)
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = get_smem_layout_atom(self.dtype, self.head_dim_v_padded)
        sO_layout_atom = sV_layout_atom
        sP_layout_atom = None
        return sQ_layout_atom, sK_layout_atom, sV_layout_atom, sO_layout_atom, sP_layout_atom

    def _get_tiled_mma(self):
        tiled_mma_qk = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, cutlass.Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )
        tiled_mma_pv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, cutlass.Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], 1024]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
        ]
        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sQV], 1024]

        @cute.struct
        class SharedStorageQKV:
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct

        @cute.struct
        class SharedStorageSharedQV:
            sQ: sQV_struct
            sK: sK_struct

        return SharedStorageQKV if const_expr(not self.Q_in_regs) else SharedStorageSharedQV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        stream: cuda.CUstream,
        softmax_scale: Optional[cutlass.Float32] = None,
        softcap: Optional[cutlass.Float32] = None,
        window_size_left: Optional[cutlass.Int32] = None,
        window_size_right: Optional[cutlass.Int32] = None,
    ):
        """Configures and launches the flash attention kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(_, _, _, 1)
        """
        self._check_type(*(t.element_type if t is not None else None for t in (mQ, mK, mV, mO, mLSE)))
        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_pv.size
        self.num_producer_threads = self.num_threads
        self.num_Q_load_threads = self.num_threads
        self.num_epilogue_threads = self.num_threads
        # self.use_tma_O = self.arch >= 90 and mCuSeqlensQ is None
        self.use_tma_O = self.arch >= 90
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()
        mQ, mK, mV, mO = [cute.make_tensor(t.iterator, cute.select(t.layout, mode=[1, 3, 2, 0])) for t in (mQ, mK, mV, mO)]
        mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=[2, 1, 0]))
        # grid_dim: (m_block, num_head, batch_size)
        grid_dim = (
            cute.ceil_div(mQ.shape[0], self.m_block_size),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
        )
        # If there's tanh softcapping, we do tanh(scores * softmax_scale / softcap_val) * softcap_val.
        # Right after this, we multiply by log2(e) before applying exp2.
        # To reduce the number of instructions, we instead pre-multiply softmax_scale / softcap_val
        # (assigning it to softcap_val) and pre-multiply softcap_val * log2(e)
        # (assigning it to softmax_scale_log2).
        LOG2_E = math.log2(math.e)
        if const_expr(softcap is not None):
            softmax_scale_log2 = softmax_scale * LOG2_E
            softcap_val = None
        else:
            softmax_scale_log2 = softcap * LOG2_E
            softcap_val = cutlass.Float32(softmax_scale / softcap)
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            softmax_scale_log2,
            softcap_val,
            window_size_left,
            window_size_right,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sP_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
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
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale_log2: cutlass.Float32,
        softcap_val: Optional[cutlass.Float32],
        window_size_left: cutlass.Int32,
        window_size_right: cutlass.Int32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()
        m_block, num_head, batch_size = cute.arch.block_idx()

        block_info = BlockInfo(
            self.m_block_size, self.n_block_size, self.is_causal, self.is_local,
            window_size_left, window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        seqlen = SeqlenInfo(seqlen_q=mQ.shape[0], seqlen_k=mK.shape[0])
        n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
        # TODO: return early if n_block_max == 0
        # if self.is_causal:
        #     if n_block_max <= 0:
        #         return
        n_block = n_block_max - 1

        # ///////////////////////////////////////////////////////////////////////////////
        # Get the appropriate tiles for this thread block.
        # ///////////////////////////////////////////////////////////////////////////////
        blkQ_shape = (self.m_block_size, self.head_dim_padded)
        blkK_shape = (self.n_block_size, self.head_dim_padded)
        blkV_shape = (self.n_block_size, self.head_dim_v_padded)
        gQ = cute.local_tile(mQ[None, None, num_head, batch_size], blkQ_shape, (m_block, 0))
        num_head_kv = num_head // self.qhead_per_kvhead
        gK = cute.local_tile(mK[None, None, num_head_kv, batch_size], blkK_shape, (None, 0))
        gV = cute.local_tile(mV[None, None, num_head_kv, batch_size], blkV_shape, (None, 0))

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sK_layout)
        if const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout)
        else:
            sV = cute.make_tensor(cute.recast_ptr(sQ.iterator, dtype=self.dtype), sV_layout)
        # Transpose view of V to tensor with layout (head_dim_v, n_block_size) for tiled mma
        sVt = utils.transpose_view(sV)

        gmem_thr_copy_K = gmem_tiled_copy_K.get_slice(tidx)
        gmem_thr_copy_V = gmem_tiled_copy_V.get_slice(tidx)
        # (CPY_Atom, CPY_N, CPY_K, n_block)
        tKsK, tKgK = gmem_thr_copy_K.partition_D(sK), gmem_thr_copy_K.partition_S(gK)
        # (CPY_Atom, CPY_N, CPY_K, n_block)
        tVsV, tVgV = gmem_thr_copy_V.partition_D(sV), gmem_thr_copy_V.partition_S(gV)

        # ///////////////////////////////////////////////////////////////////////////////
        # Tile MMA compute thread partitions and allocate accumulators
        # ///////////////////////////////////////////////////////////////////////////////
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
        tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))
        acc_shape_O = thr_mma_pv.partition_shape_C((self.m_block_size, self.head_dim_v_padded))
        acc_O = cute.make_fragment(acc_shape_O, cutlass.Float32)
        acc_O.fill(0.0)

        # ///////////////////////////////////////////////////////////////////////////////
        # Smem copy atom tiling
        # ///////////////////////////////////////////////////////////////////////////////
        smem_copy_atom_QK = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype,
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype,
        )
        smem_thr_copy_Q = utils.make_tiled_copy_A(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_copy_K = utils.make_tiled_copy_B(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_copy_V = utils.make_tiled_copy_B(smem_copy_atom_V, tiled_mma_pv).get_slice(tidx)

        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)

        # ///////////////////////////////////////////////////////////////////////////////
        # Predicate: Mark indices that need to copy when problem_shape isn't a multiple
        # of tile_shape
        # ///////////////////////////////////////////////////////////////////////////////
        # Construct identity layout for KV
        cK = cute.make_identity_tensor((self.n_block_size, self.head_dim_padded))
        tKcK = gmem_thr_copy_K.partition_S(cK)
        t0KcK = gmem_thr_copy_K.get_slice(0).partition_S(cK)
        if const_expr(self.head_dim_padded == self.head_dim_v_padded):
            tVcV = tKcK
            t0VcV = t0KcK
        else:
            cV = cute.make_identity_tensor((self.n_block_size, self.head_dim_v_padded))
            tVcV = gmem_thr_copy_V.partition_S(cV)
            t0VcV = gmem_thr_copy_V.get_slice(0).partition_S(cV)
        # Allocate predicate tensors for m and n, here we only allocate the tile of k, and
        # use "if" on the mn dimension.
        # This is to reduce register pressure and gets 2-3% performance gain.
        tKpK = predicate_k(tKcK, limit=mK.shape[1])
        if const_expr(self.same_hdim_kv):
            tVpV = tKpK
        else:
            tVpV = predicate_k(tVcV, limit=mV.shape[1])

        # shape: (atom_v_m * rest_m)
        softmax = Softmax(softmax_scale_log2, num_rows=acc_O.shape[0][0] * acc_O.shape[1])
        softmax.reset()

        # group parameters for compute_one_n_block
        mma_params = SimpleNamespace(
            thr_mma_qk=thr_mma_qk, thr_mma_pv=thr_mma_pv,
            tSrQ=tSrQ, tSrK=tSrK, tOrVt=tOrVt, acc_O=acc_O,
        )
        smem_copy_params = SimpleNamespace(
            smem_thr_copy_Q=smem_thr_copy_Q,
            smem_thr_copy_K=smem_thr_copy_K,
            smem_thr_copy_V=smem_thr_copy_V,
            tSsQ=tSsQ, tSsK=tSsK, tOsVt=tOsVt,
        )
        load_K = partial(self.load_K, gmem_tiled_copy_K, tKgK, tKsK, tKcK, t0KcK, tKpK,
                         seqlen=seqlen.seqlen_k)
        load_V = partial(self.load_V, gmem_tiled_copy_V, tVgV, tVsV, tVcV, t0VcV, tVpV,
                         seqlen=seqlen.seqlen_k)
        # Softcapping needs to happen before masking since if we apply after masking, softcapping can turn
        # -inf to e.g. -50.0, which can affect the attention softmax.
        def scoremod_premask_fn(acc_S):
            if const_expr(softcap_val is not None):
                acc_S.store(cute.math.tanh(acc_S.load() * softcap_val, fastmath=True))

        compute_one_n_block = partial(
            self.compute_one_n_block, mma_params=mma_params, smem_copy_params=smem_copy_params,
            softmax=softmax, load_K=load_K, load_V=load_V, scoremod_premask_fn=scoremod_premask_fn,
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Prologue
        # ///////////////////////////////////////////////////////////////////////////////
        # Start async loads of the last mn-tile, where we take care of the mn residue
        gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
        self.load_Q(gmem_thr_copy_Q, gQ, sQ, m_block, seqlen=seqlen.seqlen_q, headdim=mQ.shape[1])
        cute.arch.cp_async_commit_group()

        def preprocess_Q():
            cute.arch.cp_async_wait_group(self.num_stages * 2 - 1)
            if const_expr(self.Q_in_regs):
                cute.arch.barrier()
                tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
                cute.copy(smem_thr_copy_Q, tSsQ, tSrQ_copy_view)

        # If Q_in_regs, we load Q, then load 1 stage of K, then (optionally) rotate Q and
        # read from smem_q to registers, then load V.
        # If !Q_in_regs, we load Q, load all stages of K & V, then (optionally) rotate Q.
        if const_expr(self.Q_in_regs):
            load_K(n_block, smem_pipe_write=0, need_predicates=True)
            cute.arch.cp_async_commit_group()
            preprocess_Q()
            cute.arch.barrier()  # Make sure all threads have read smem_q before loading V

        for stage in cutlass.range_constepxr(self.num_stages):
            if const_expr(not self.Q_in_regs or stage > 0):
                if stage == 0 or n_block - stage >= 0:
                    load_K(n_block - stage, smem_pipe_write=stage, need_predicates=stage==0)
                cute.arch.cp_async_commit_group()
            if const_expr(stage < self.num_stages - 1):
                if stage == 0 or n_block - stage >= 0:
                    load_V(n_block - stage, smem_pipe_write=stage, need_predicates=stage==0)
                cute.arch.cp_async_commit_group()
        if const_expr(not self.Q_in_regs):
            preprocess_Q()

        # ///////////////////////////////////////////////////////////////////////////////
        # Mainloop
        # ///////////////////////////////////////////////////////////////////////////////
        # Start processing of the first n-block.
        # For performance reason, we separate out two kinds of iterations:
        # those that need masking on S, and those that don't.
        # We need masking on S for the very last block when K and V has length not multiple of n_block_size.
        # We also need masking on S if it's causal, for the last several blocks.
        mask = AttentionMask(
            self.m_block_size, self.n_block_size, seqlen.seqlen_q, seqlen.seqlen_k,
            window_size_left, window_size_right,
            self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        mask_fn = partial(
            mask.apply_mask, m_block=m_block, thr_mma=thr_mma_qk,
            mask_causal=self.is_causal, mask_local=self.is_local,
        )

        # First iteration with seqlen masking
        smem_pipe_read = cutlass.Int32(0)
        smem_pipe_write = cutlass.Int32(self.num_stages - 1)
        compute_one_n_block(n_block, smem_pipe_read, smem_pipe_write, is_first_n_block=True,
                            check_inf=True, mask_fn=partial(mask_fn, mask_seqlen=True))
        smem_pipe_read = self.advance_pipeline(smem_pipe_read)
        smem_pipe_write = self.advance_pipeline(smem_pipe_write)
        # Next couple of iterations with causal masking
        if const_expr(self.is_causal or self.is_local):
            n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                seqlen, m_block, n_block_min
            )
            for n_tile in cutlass.range(n_block_max - 1 - n_block_min_causal_local_mask, unroll=1):
                n_block = n_block_max - 2 - n_tile
                compute_one_n_block(n_block, smem_pipe_read, smem_pipe_write, check_inf=True,
                                    mask_fn=partial(mask_fn, mask_seqlen=False))
                smem_pipe_read = self.advance_pipeline(smem_pipe_read)
                smem_pipe_write = self.advance_pipeline(smem_pipe_write)
        # The remaining iterations have no masking
        for n_tile in cutlass.range(n_block, unroll=1):
            compute_one_n_block(n_block - n_tile - 1, smem_pipe_read, smem_pipe_write, check_inf=True)
            smem_pipe_read = self.advance_pipeline(smem_pipe_read)
            smem_pipe_write = self.advance_pipeline(smem_pipe_write)
        # TODO: local

        # normalize acc_O by row_sum and calculate the lse
        row_scale = softmax.finalize()
        softmax.rescale_O(acc_O, row_scale)

        # ///////////////////////////////////////////////////////////////////////////////
        # Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        # reuse sQ's data iterator
        sO = cute.make_tensor(sQ.iterator, sO_layout)
        self.epilogue(
            acc_O, softmax.row_sum, mO, mLSE, sO,
            gmem_tiled_copy_O, None, tiled_mma_pv, tidx, m_block, num_head, batch_size
        )

    @cute.jit
    def compute_one_n_block(
        self,
        n_block: cutlass.Int32,
        smem_pipe_read: cutlass.Int32,
        smem_pipe_write: cutlass.Int32,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        load_K: Callable,
        load_V: Callable,
        scoremod_premask_fn: Callable,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
    ):
        """Compute one n_block of S/O.

        This function provides different variants for processing the first n block versus
        subsequent blocks.
        """
        def sync():
            cute.arch.cp_async_wait_group(self.num_stages * 2 - 2)
            cute.arch.barrier()

        acc_shape_S = mma_params.thr_mma_qk.partition_shape_C((self.m_block_size, self.n_block_size))
        acc_S = cute.make_fragment(acc_shape_S, cutlass.Float32)
        acc_S.fill(0.0)
        # wait for smem tile QK before mma calculation for S
        sync()
        # need predicates for the first tile
        def load_V_next():
            if self.num_stages == 1 or n_block - self.num_stages + 1 >= 0:
                load_V(n_block - self.num_stages + 1, smem_pipe_write,
                       need_predicates=is_first_n_block and self.num_stages == 1)
            cute.arch.cp_async_commit_group()
        load_V_next()
        gemm(
            mma_params.thr_mma_qk, acc_S, mma_params.tSrQ, mma_params.tSrK,
            smem_copy_params.tSsQ,
            smem_copy_params.tSsK[None, None, None, smem_pipe_read if const_expr(self.num_stages > 1) else 0],
            smem_copy_params.smem_thr_copy_Q, smem_copy_params.smem_thr_copy_K,
            # hook_fn=load_V_next,
            A_in_regs=self.Q_in_regs,
        )
        scoremod_premask_fn(acc_S)
        smem_pipe_write = self.advance_pipeline(smem_pipe_write)
        def load_K_next():
            if n_block - self.num_stages >= 0:
                load_K(n_block - self.num_stages, smem_pipe_write, need_predicates=False)
            cute.arch.cp_async_commit_group()
        # wait for smem tile V for O
        if const_expr(self.num_stages == 1):
            sync()
            load_K_next()
        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=check_inf)
        softmax.rescale_O(mma_params.acc_O, row_scale)
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP = cute.make_tensor(rP.iterator, utils.convert_layout_acc_frgA(rP.layout))
        if const_expr(self.num_stages > 1):
            sync()
            load_K_next()
        gemm_rs(
            mma_params.thr_mma_pv, mma_params.acc_O, tOrP, mma_params.tOrVt,
            smem_copy_params.tOsVt[None, None, None, smem_pipe_read if const_expr(self.num_stages > 1) else 0],
            smem_copy_params.smem_thr_copy_V,
            # hook_fn=load_K_next,
        )
        # if const_expr(self.num_stages > 1):
        #     load_K_next()
torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}

def convert_from_dlpack(x, leading_dim, alignment=16, divisibility=1) -> cute.Tensor:
    return (
        from_dlpack(x, assumed_align=alignment)
        .mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(
            mode=leading_dim, stride_order=x.dim_order(), divisibility=divisibility
        )
    )

def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x
# for testing
def run_flash_attention_fwd(
    dtype: Type[cutlass.Numeric],
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    num_head: int,
    head_dim: int,
    softmax_scale: Optional[float] = None,
    m_block_size: int = 128,
    n_block_size: int = 128,
    num_threads: int = 128,
    is_causal: bool = False,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,

    num_stages: int = 1,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    softcap: Optional[float] = None,
):
    

    # Create tensor Q/K/V/O
    def create_tensor(
        batch_size: int,
        seqlen: int,
        num_head: int,
        head_dim: int,
        dtype: Type[cutlass.Numeric],
    ) -> cute.Tensor:
        # (batch_size, seqlen, num_head, head_dim)
        shape = (batch_size, seqlen, num_head, head_dim)
        return (
            torch.empty(*shape, dtype=torch.int32).random_(-2, 2).to(dtype=dtype).cuda()
        )

    q = create_tensor(
        batch_size, seqlen_q, num_head, head_dim, cutlass_torch.dtype(dtype)
    )
    k = create_tensor(
        batch_size, seqlen_k, num_head, head_dim, cutlass_torch.dtype(dtype)
    )
    v = create_tensor(
        batch_size, seqlen_k, num_head, head_dim, cutlass_torch.dtype(dtype)
    )
    o = create_tensor(
        batch_size, seqlen_q, num_head, head_dim, cutlass_torch.dtype(dtype)
    )
    # head_dim_v = v.shape[-1]
    # # Skip unsupported testcase
    # if not FlashAttentionForwardSm80.can_implement(
    #     #dtype,
    #     #head_dim,
    #     #m_block_size,
    #     #n_block_size,
    #     #num_threads,
    #     #is_causal,
    #     dtype, head_dim, head_dim_v, m_block_size, n_block_size, num_stages, num_threads, is_causal,
    #     Q_in_regs=False
    # ):
    #     raise TypeError(
    #         f"Unsupported testcase {dtype}, {head_dim}, {m_block_size}, {n_block_size}, {num_threads}, {is_causal}"
    #     )
    
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]
    seqlen_k, num_head_kv, _ = k.shape[-3:]
    head_dim_v = v.shape[-1]
    if cu_seqlens_k is None:
        assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
        assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    else:
        assert k.shape == (seqlen_k, num_head_kv, head_dim)
        assert v.shape == (seqlen_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (batch_size + 1,), "cu_seqlens_k must have shape (batch_size + 1,)"
    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), "cu_seqlens_q must have shape (batch_size + 1,)"
    assert seqused_q is None or seqused_q.shape == (batch_size,), "seqused_q must have shape (batch_size,)"
    assert seqused_k is None or seqused_k.shape == (batch_size,), "seqused_k must have shape (batch_size,)"
    assert q.dtype in [torch.float16, torch.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            assert t.dtype == torch.int32, "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be int32"
            assert t.stride(0) == 1, "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be contiguous"
    assert all(t is None or t.is_cuda for t in (q, k, v, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    assert head_dim <= 256, "head_dim must be less than or equal to 256"
    alignment = 16 // q.element_size()
    assert head_dim % alignment == 0, f"head_dim must be divisible by {alignment}"
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if softcap == 0.0:
        softcap = None
    qhead_per_kvhead = num_head // num_head_kv

    out_torch_dtype = q.dtype
    device = q.device
    q_batch_seqlen_shape = (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
    out = torch.empty(*q_batch_seqlen_shape, num_head, head_dim_v, dtype=out_torch_dtype, device=device)
    lse_shape = (batch_size, num_head, seqlen_q) if cu_seqlens_q is None else (num_head, total_q)
    requires_grad = q.requires_grad or k.requires_grad or v.requires_grad
    lse = torch.empty(lse_shape, dtype=torch.float32, device=device) if requires_grad else None

    dtype = torch2cute_dtype_map[q.dtype]
    q_tensor, k_tensor, v_tensor, o_tensor = [
        convert_from_dlpack(
            t.detach(), leading_dim=t.ndim - 1, divisibility=128 // dtype.width
        ) for t in (q, k, v, out)
    ]
    lse_tensor = convert_from_dlpack(lse, leading_dim=lse.ndim - 1, alignment=4) if lse is not None else None
    cu_seqlens_q_tensor, cu_seqlens_k_tensor, seqused_q_tensor, seqused_k_tensor = [
        from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0) if t is not None else None
        for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
    ]
    print(f"lse_tensor: {lse_tensor}")
    if is_causal:
        window_size_right = 0
    local = window_size_left is not None or window_size_right is not None
    if window_size_left is not None or window_size_right is not None:
        if window_size_left is None and window_size_right == 0:
            is_causal, local = True, False
        else:
            is_causal, local = False, True
    fa2_fwd = FlashAttentionForwardSm80(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=is_causal,
                is_local=local,
                pack_gqa=False,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                num_stages=num_stages,
                #num_stages=2,
                num_threads=num_threads,
                Q_in_regs=False,
    )
    # assume input is 16B align.
    # q_tensor = (
    #     from_dlpack(q, assumed_align=16)
    #     .mark_layout_dynamic(leading_dim=3)
    #     .mark_compact_shape_dynamic(
    #         mode=3, stride_order=q.dim_order(), divisibility=(128 // dtype.width)
    #     )
    # )
    # k_tensor = (
    #     from_dlpack(k, assumed_align=16)
    #     .mark_layout_dynamic(leading_dim=3)
    #     .mark_compact_shape_dynamic(
    #         mode=3, stride_order=k.dim_order(), divisibility=(128 // dtype.width)
    #     )
    # )
    # v_tensor = (
    #     from_dlpack(v, assumed_align=16)
    #     .mark_layout_dynamic(leading_dim=3)
    #     .mark_compact_shape_dynamic(
    #         mode=3, stride_order=v.dim_order(), divisibility=(128 // dtype.width)
    #     )
    # )
    # o_tensor = (
    #     from_dlpack(o, assumed_align=16)
    #     .mark_layout_dynamic(leading_dim=3)
    #     .mark_compact_shape_dynamic(
    #         mode=3, stride_order=o.dim_order(), divisibility=(128 // dtype.width)
    #     )
    # )

    # Get current CUDA stream from PyTorch
    torch_stream = torch.cuda.current_stream()
    # Get the raw stream pointer as a CUstream
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    # compile the fa2 forward pass
    # fn __call__ of fa2_fwd
    # solve the err( Duo MA): 
    # cutlass.base_dsl.common.DSLRuntimeError: DSLRuntimeError: expects argument #8 (softmax_scale) to be one of (<class 'cutlass.base_dsl.typing.Float32'>, <class 'NoneType'>), but got <class 'float'>
    #softmax_scale = cutlass.base_dsl.dsl.const(softmax_scale)
    #softmax_scale_2 = torch.tensor(softmax_scale, dtype=torch.float32, device=device)
    #softmax_tensor = convert_from_dlpack(softmax_scale, leading_dim=softmax_scale.ndim - 1, alignment=16)
    #softmax_scale_tensor = from_dlpack(softmax_scale_2).mark_layout_dynamic()
    softmax_scale_cute = cutlass.Float32(softmax_scale) # float -> cutlass.base_dsl.typing.Float32
    #softcap = cutlass.Float32(softcap)
    compiled_fa2_fwd = cute.compile(
        fa2_fwd, q_tensor, k_tensor, v_tensor, o_tensor,lse_tensor,current_stream, softmax_scale_cute, softcap, window_size_left,window_size_right,
    )
    # warmup
    for _ in range(warmup_iterations):
        compiled_fa2_fwd(
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,current_stream, softmax_scale_cute, softcap,window_size_left,window_size_right,
            
        )
    # run the compiled fa2 forward pass
    for _ in range(iterations):
        compiled_fa2_fwd(
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,current_stream, softmax_scale_cute, softcap,window_size_left,window_size_right,
            
        )
    torch.cuda.synchronize()

    print("Executing fa2 forward kernel...")

    avg_time_us = testing.benchmark(
        compiled_fa2_fwd,
        kernel_arguments=testing.JitArguments(q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor,current_stream, softmax_scale_cute, softcap,window_size_left,window_size_right),
        warmup_iterations=warmup_iterations,
        profiling_iterations=iterations,
        use_cuda_graphs=False,
    )

    print(f"Using cutlass.cute.testing.benchmark Kernel execution time: {avg_time_us / 1e3:.4f} ms")

    print("Executing fa2 forward kernel...")

    avg_time_us = testing.benchmark(
        ref_flash_attn_fwd,
        kernel_arguments=testing.JitArguments(q, k, v, softmax_scale_cute, is_causal),
        warmup_iterations=warmup_iterations,
        profiling_iterations=iterations,
        use_cuda_graphs=False,
    )

    print(f"Using cutlass.cute.testing.benchmark, Ref Kernel execution time: {avg_time_us / 1e3:.4f} ms")


    # reference implementation
    ref_o = ref_flash_attn_fwd(q,k,v,softmax_scale, is_causal)


    #Verify that the data results are correct
    if not skip_ref_check:
        print("Verifying results...")
        torch.testing.assert_close(o.cpu(), ref_o.cpu(), atol=1e-02, rtol=1e-04)
        print("Results verified successfully!")

    print("")

    from triton.testing import do_bench


    fn = lambda: compiled_fa2_fwd(q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor,current_stream, softmax_scale_cute, softcap,window_size_left,window_size_right)
    time.sleep(0.5)
    avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
    #mem_bw = (2 * x.numel() * dtype.width // 8) / (avg_time / 1000) / 1e9
    print(f"Using triton.testing.do_bench,  Kernel execution time: {avg_time:.4f} ms")
    #print(f"Mem throughput: {mem_bw:.2f} GB/s")

    fn = lambda: ref_flash_attn_fwd(q, k, v, softmax_scale,is_causal)
    for _ in range(5):
        fn()  # warm up
    time.sleep(0.5)
    avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
    #mem_bw = (2 * x.numel() * dtype.width // 8) / (avg_time / 1000) / 1e9
    print(f"Using triton.testing.do_bench, Ref Kernel execution time: {avg_time:.4f} ms")
    #print(f"Ref Mem throughput: {mem_bw:.2f} GB/s")



def ref_flash_attn_fwd(q,k,v,softmax_scale, is_causal):
    # reference implementation
    q_ref = q.permute(0, 2, 1, 3)
    k_ref = k.permute(0, 2, 1, 3)
    v_ref = v.permute(0, 2, 1, 3)
    torch.backends.cuda.enable_flash_sdp(enabled=True)
    ref_o = torch.nn.functional.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, scale=softmax_scale, is_causal=is_causal
    ).permute(0, 2, 1, 3)

    return ref_o




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="example of flash attention v2 with CuTe on GPU"
    )
    parser.add_argument("--dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seqlen_q", type=int, default=8192)
    parser.add_argument("--seqlen_k", type=int, default=8192)
    parser.add_argument("--num_head", type=int, default=16)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--softmax_scale", type=float, default=0.5)
    parser.add_argument("--m_block_size", type=int, default=128)
    parser.add_argument("--n_block_size", type=int, default=64)
    parser.add_argument("--num_threads", type=int, default=128)
    parser.add_argument("--is_causal", action="store_true", help="Enable causal mask")
    parser.add_argument("--warmup_iterations", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--skip_ref_check", action="store_true", help="Skip reference check"
    )

    args = parser.parse_args()
    run_flash_attention_fwd(
        args.dtype,
        args.batch_size,
        args.seqlen_q,
        args.seqlen_k,
        args.num_head,
        args.head_dim,
        args.softmax_scale,
        args.m_block_size,
        args.n_block_size,
        args.num_threads,
        args.is_causal,
    )

    print("PASS")                                                                                     

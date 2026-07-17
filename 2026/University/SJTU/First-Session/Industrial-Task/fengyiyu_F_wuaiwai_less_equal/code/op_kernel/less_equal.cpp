/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Kernel侧核函数实现
#include "kernel_operator.h"

#include "less_equal_tiling.h"
#include "tiling_key_less_equal.h"

using namespace AscendC;

template <class DT_X1>
class KernelLessEqual {
public:
    __aicore__ inline KernelLessEqual() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const LessEqualTilingData &td) {
        mode = td.mode;
        totalElements = td.totalElements;
        tileSize = td.tileSize;
        perCore = td.perCore;
        ndim = td.ndim;
        lastDimLen = td.lastDimLen;
        totalRows = td.totalRows;
        for (uint32_t i = 0; i < LE_MAX_DIM; ++i) {
            outShape[i] = td.outShape[i];
            x1StrideArr[i] = td.x1Stride[i];
            x2StrideArr[i] = td.x2Stride[i];
        }

        x1Gm.SetGlobalBuffer((__gm__ DT_X1 *)x1);
        x2Gm.SetGlobalBuffer((__gm__ DT_X1 *)x2);
        yGm.SetGlobalBuffer((__gm__ int8_t *)y);

        if (totalElements == 0) { return; }  // empty tensor: nothing to allocate

        // Input queues (double buffered), output int8 queue (double buffered)
        pipe.InitBuffer(qX1, 2, tileSize * (uint32_t)sizeof(DT_X1));
        pipe.InitBuffer(qX2, 2, tileSize * (uint32_t)sizeof(DT_X1));
        pipe.InitBuffer(qOut, 2, tileSize * (uint32_t)sizeof(int8_t));

        InitComputeBuffers();

        // Prebuild ones/zeros half vectors once.
        LocalTensor<half> ones = bufOnes.Get<half>();
        LocalTensor<half> zeros = bufZeros.Get<half>();
        Duplicate(ones, (half)1.0, tileSize);
        Duplicate(zeros, (half)0.0, tileSize);
    }

    __aicore__ inline void Process() {
        if (totalElements == 0) { return; }
        if (mode == 0) {
            ProcessFast();
        } else {
            ProcessBcast();
        }
    }

private:
    __aicore__ inline void InitComputeBuffers() {
        pipe.InitBuffer(bufMask, ((tileSize / 8) + 31) / 32 * 32);
        pipe.InitBuffer(bufOnes, tileSize * (uint32_t)sizeof(half));
        pipe.InitBuffer(bufZeros, tileSize * (uint32_t)sizeof(half));
        pipe.InitBuffer(bufOutHalf, tileSize * (uint32_t)sizeof(half));
        if constexpr (std::is_same_v<DT_X1, int8_t>) {
            pipe.InitBuffer(bufAHalf, tileSize * (uint32_t)sizeof(half));
            pipe.InitBuffer(bufBHalf, tileSize * (uint32_t)sizeof(half));
        }
        if constexpr (std::is_same_v<DT_X1, int32_t>) {
            pipe.InitBuffer(bufI32, tileSize * (uint32_t)sizeof(int32_t));
        }
    }

    // Round compute count up to 256 elems so the bit-packed Compare mask is 32B-granular
    // (256 bits = 32 bytes) and the Compare/Select count footprint satisfies the 256B rule
    // for half (256B) and float/int32 (512B). Only real L bytes are ever copied to GM.
    __aicore__ inline uint32_t RoundUp256(uint32_t v) {
        return (v + 255) / 256 * 256;
    }

    // Produce 0/1 int8 for Lc lanes from aLocal,bLocal (native DT_X1). Writes outI8.
    // aScalar/bScalar: when true, that operand tensor holds only element [0] (a
    // scalar-broadcast row) and must be materialized to Lc lanes here. For every
    // dtype EXCEPT int8 the materialization already happened in LoadOperand
    // (Duplicate<DT_X1> is valid for half/float/int32). For int8, Duplicate<int8_t>
    // (b8) is NOT supported on A2, so the scalar is expanded in the half domain.
    __aicore__ inline void ComputeTile(const LocalTensor<DT_X1> &aLocal,
                                       const LocalTensor<DT_X1> &bLocal,
                                       const LocalTensor<int8_t> &outI8,
                                       uint32_t Lc,
                                       bool aScalar, bool bScalar) {
        LocalTensor<uint8_t> mask = bufMask.Get<uint8_t>();
        LocalTensor<half> ones = bufOnes.Get<half>();
        LocalTensor<half> zeros = bufZeros.Get<half>();
        LocalTensor<half> outHalf = bufOutHalf.Get<half>();

        if constexpr (std::is_same_v<DT_X1, int32_t>) {
            // Exact integer route: m = min(x1,x2); (m == x1) <=> x1 <= x2.
            // Scalar operands were already Duplicate<int32_t>'d in LoadOperand.
            LocalTensor<int32_t> m = bufI32.Get<int32_t>();
            Min(m, aLocal, bLocal, (int32_t)Lc);
            Compare(mask, m, aLocal, CMPMODE::EQ, Lc);
        } else if constexpr (std::is_same_v<DT_X1, int8_t>) {
            // int8 -> half lossless (|v| <= 127 exact in half), compare as half.
            LocalTensor<half> aHalf = bufAHalf.Get<half>();
            LocalTensor<half> bHalf = bufBHalf.Get<half>();
            if (aScalar) {
                int8_t va = aLocal.GetValue(0);
                Duplicate(aHalf, (half)(float)(int32_t)va, Lc);
            } else {
                Cast(aHalf, aLocal, RoundMode::CAST_NONE, Lc);
            }
            if (bScalar) {
                int8_t vb = bLocal.GetValue(0);
                Duplicate(bHalf, (half)(float)(int32_t)vb, Lc);
            } else {
                Cast(bHalf, bLocal, RoundMode::CAST_NONE, Lc);
            }
            Compare(mask, aHalf, bHalf, CMPMODE::LE, Lc);
        } else {
            // half / float direct. Scalar operands were already Duplicate<DT_X1>'d.
            Compare(mask, aLocal, bLocal, CMPMODE::LE, Lc);
        }

        // Expand bit-mask -> half 1.0/0.0, then cast to int8 0/1.
        Select(outHalf, mask, ones, zeros, SELMODE::VSEL_TENSOR_TENSOR_MODE, Lc);
        Cast(outI8, outHalf, RoundMode::CAST_RINT, Lc);
    }

    // ---------------- FAST path ----------------
    __aicore__ inline void ProcessFast() {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t off = blockIdx * perCore;
        if (off >= totalElements) { return; }
        uint32_t myElems = perCore;
        if (off + myElems > totalElements) { myElems = totalElements - off; }

        DataCopyPadExtParams<DT_X1> pad{false, 0, 0, (DT_X1)0};

        for (uint32_t t = 0; t < myElems; t += tileSize) {
            uint32_t L = tileSize;
            if (t + L > myElems) { L = myElems - t; }
            uint32_t Lc = RoundUp256(L);
            uint32_t base = off + t;

            LocalTensor<DT_X1> aLocal = qX1.AllocTensor<DT_X1>();
            DataCopyPad(aLocal, x1Gm[base],
                        DataCopyExtParams{1, L * (uint32_t)sizeof(DT_X1), 0, 0, 0}, pad);
            qX1.EnQue(aLocal);

            LocalTensor<DT_X1> bLocal = qX2.AllocTensor<DT_X1>();
            DataCopyPad(bLocal, x2Gm[base],
                        DataCopyExtParams{1, L * (uint32_t)sizeof(DT_X1), 0, 0, 0}, pad);
            qX2.EnQue(bLocal);

            aLocal = qX1.DeQue<DT_X1>();
            bLocal = qX2.DeQue<DT_X1>();
            LocalTensor<int8_t> outI8 = qOut.AllocTensor<int8_t>();
            ComputeTile(aLocal, bLocal, outI8, Lc, false, false);
            qX1.FreeTensor(aLocal);
            qX2.FreeTensor(bLocal);
            qOut.EnQue(outI8);

            outI8 = qOut.DeQue<int8_t>();
            DataCopyPad(yGm[base], outI8, DataCopyExtParams{1, L, 0, 0, 0});
            qOut.FreeTensor(outI8);
        }
    }

    // ---------------- BCAST path ----------------
    __aicore__ inline void initRowState(uint32_t r0) {
        x1Base = 0;
        x2Base = 0;
        uint32_t rem = r0;
        // leading dims are [0 .. ndim-2]
        for (uint32_t d = 0; d + 1 < ndim; ++d) {
            uint32_t mult = 1;
            for (uint32_t e = d + 1; e + 1 < ndim; ++e) { mult *= outShape[e]; }
            uint32_t id = (mult == 0) ? 0 : (rem / mult);
            rem = (mult == 0) ? rem : (rem % mult);
            idx[d] = id;
            x1Base += id * x1StrideArr[d];
            x2Base += id * x2StrideArr[d];
        }
    }

    __aicore__ inline void advanceRowState() {
        if (ndim < 2) { return; }
        int32_t d = (int32_t)ndim - 2;
        idx[d]++;
        x1Base += x1StrideArr[d];
        x2Base += x2StrideArr[d];
        while (d > 0 && idx[d] >= outShape[d]) {
            x1Base -= outShape[d] * x1StrideArr[d];
            x2Base -= outShape[d] * x2StrideArr[d];
            idx[d] = 0;
            d--;
            idx[d]++;
            x1Base += x1StrideArr[d];
            x2Base += x2StrideArr[d];
        }
    }

    __aicore__ inline void LoadOperand(TQue<TPosition::VECIN, 2> &q,
                                       const GlobalTensor<DT_X1> &gm,
                                       uint32_t baseIdx, uint32_t lastStride,
                                       uint32_t c, uint32_t chunk, uint32_t Lc,
                                       LocalTensor<DT_X1> &out) {
        DataCopyPadExtParams<DT_X1> pad{false, 0, 0, (DT_X1)0};
        LocalTensor<DT_X1> loc = q.AllocTensor<DT_X1>();
        if (lastStride == 0) {
            // scalar broadcast across the row: load 1 element.
            DataCopyPad(loc, gm[(uint32_t)baseIdx],
                        DataCopyExtParams{1, (uint32_t)sizeof(DT_X1), 0, 0, 0}, pad);
            q.EnQue(loc);
            loc = q.DeQue<DT_X1>();
            // Materialize to Lc lanes here for half/float/int32 (Duplicate supports them).
            // int8_t (b8) Duplicate is NOT supported on A2 -> leave the single element at
            // loc[0]; ComputeTile's int8 branch expands the scalar in the half domain.
            if constexpr (!std::is_same_v<DT_X1, int8_t>) {
                DT_X1 v = loc.GetValue(0);
                Duplicate(loc, v, Lc);
            }
        } else {
            // contiguous run (stride 1): load 'chunk' elements starting baseIdx + c
            DataCopyPad(loc, gm[(uint32_t)baseIdx + c],
                        DataCopyExtParams{1, chunk * (uint32_t)sizeof(DT_X1), 0, 0, 0}, pad);
            q.EnQue(loc);
            loc = q.DeQue<DT_X1>();
        }
        out = loc;
    }

    __aicore__ inline void ProcessBcast() {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t rowOff = blockIdx * perCore;
        if (rowOff >= totalRows) { return; }
        uint32_t myRows = perCore;
        if (rowOff + myRows > totalRows) { myRows = totalRows - rowOff; }

        uint32_t L = lastDimLen;
        uint32_t s1last = x1StrideArr[ndim - 1];
        uint32_t s2last = x2StrideArr[ndim - 1];

        initRowState(rowOff);

        for (uint32_t r = 0; r < myRows; ++r) {
            uint32_t outRowBase = (rowOff + r) * L;
            for (uint32_t c = 0; c < L; c += tileSize) {
                uint32_t chunk = tileSize;
                if (c + chunk > L) { chunk = L - c; }
                uint32_t Lc = RoundUp256(chunk);

                LocalTensor<DT_X1> aLocal;
                LocalTensor<DT_X1> bLocal;
                LoadOperand(qX1, x1Gm, x1Base, s1last, c, chunk, Lc, aLocal);
                LoadOperand(qX2, x2Gm, x2Base, s2last, c, chunk, Lc, bLocal);

                // A scalar-broadcast operand has last-dim stride 0. For int8 its value
                // still lives only at loc[0] and is expanded inside ComputeTile; for
                // half/float/int32 it was already Duplicate'd in LoadOperand (flags are
                // then harmless no-ops in ComputeTile's non-int8 branches).
                bool aScalar = (s1last == 0);
                bool bScalar = (s2last == 0);
                LocalTensor<int8_t> outI8 = qOut.AllocTensor<int8_t>();
                ComputeTile(aLocal, bLocal, outI8, Lc, aScalar, bScalar);
                qX1.FreeTensor(aLocal);
                qX2.FreeTensor(bLocal);
                qOut.EnQue(outI8);

                outI8 = qOut.DeQue<int8_t>();
                DataCopyPad(yGm[outRowBase + c], outI8, DataCopyExtParams{1, chunk, 0, 0, 0});
                qOut.FreeTensor(outI8);
            }
            if (r + 1 < myRows) { advanceRowState(); }
        }
    }

    // Pipe / queues / buffers
    TPipe pipe;
    TQue<TPosition::VECIN, 2> qX1;
    TQue<TPosition::VECIN, 2> qX2;
    TQue<TPosition::VECOUT, 2> qOut;
    TBuf<TPosition::VECCALC> bufMask;
    TBuf<TPosition::VECCALC> bufOnes;
    TBuf<TPosition::VECCALC> bufZeros;
    TBuf<TPosition::VECCALC> bufOutHalf;
    TBuf<TPosition::VECCALC> bufAHalf;
    TBuf<TPosition::VECCALC> bufBHalf;
    TBuf<TPosition::VECCALC> bufI32;

    GlobalTensor<DT_X1> x1Gm;
    GlobalTensor<DT_X1> x2Gm;
    GlobalTensor<int8_t> yGm;

    // Tiling data
    uint32_t mode;
    uint32_t totalElements;
    uint32_t tileSize;
    uint32_t perCore;
    uint32_t ndim;
    uint32_t lastDimLen;
    uint32_t totalRows;
    uint32_t outShape[LE_MAX_DIM];
    uint32_t x1StrideArr[LE_MAX_DIM];
    uint32_t x2StrideArr[LE_MAX_DIM];

    // BCAST incremental row state
    uint32_t x1Base;
    uint32_t x2Base;
    uint32_t idx[LE_MAX_DIM];
};

template <typename DT_X1>
 __global__ __aicore__ void less_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LessEqualTilingData);
    GET_TILING_DATA_WITH_STRUCT(LessEqualTilingData, tiling_data, tiling);
    KernelLessEqual<DT_X1> op;
    op.Init(x1, x2, y, tiling_data);
    op.Process();
}

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

        if (totalElements == 0) return;

        pipe.InitBuffer(qX1, 2, tileSize * (uint32_t)sizeof(DT_X1));
        pipe.InitBuffer(qX2, 2, tileSize * (uint32_t)sizeof(DT_X1));
        pipe.InitBuffer(qOut, 2, tileSize * (uint32_t)sizeof(int8_t));

        pipe.InitBuffer(bufMask, ((tileSize / 8) + 31) / 32 * 32);
        pipe.InitBuffer(bufOnes, tileSize * (uint32_t)sizeof(half));
        pipe.InitBuffer(bufOutHalf, tileSize * (uint32_t)sizeof(half));

        if constexpr (std::is_same_v<DT_X1, int8_t>) {
            pipe.InitBuffer(bufAHalf, tileSize * (uint32_t)sizeof(half));
            pipe.InitBuffer(bufBHalf, tileSize * (uint32_t)sizeof(half));
        }
        if constexpr (std::is_same_v<DT_X1, int32_t>) {
            pipe.InitBuffer(bufI32, tileSize * (uint32_t)sizeof(int32_t));
        }

        LocalTensor<half> ones = bufOnes.Get<half>();
        Duplicate(ones, (half)1.0, tileSize);
    }

    __aicore__ inline void Process() {
        if (totalElements == 0) return;
        if (mode == 0) ProcessFast();
        else ProcessBcast();
    }

private:
    __aicore__ inline uint32_t RoundUp256(uint32_t v) {
        return (v + 255) / 256 * 256;
    }

    __aicore__ inline void ComputeTile(const LocalTensor<DT_X1> &aLocal,
                                       const LocalTensor<DT_X1> &bLocal,
                                       const LocalTensor<int8_t> &outI8,
                                       uint32_t Lc,
                                       bool aScalar, bool bScalar) {
        LocalTensor<uint8_t> mask = bufMask.Get<uint8_t>();
        LocalTensor<half> ones = bufOnes.Get<half>();
        LocalTensor<half> outHalf = bufOutHalf.Get<half>();

        if constexpr (std::is_same_v<DT_X1, int32_t>) {
            LocalTensor<int32_t> m = bufI32.Get<int32_t>();
            Min(m, aLocal, bLocal, (int32_t)Lc);
            Compare(mask, m, aLocal, CMPMODE::EQ, Lc);
        } else if constexpr (std::is_same_v<DT_X1, int8_t>) {
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
            Compare(mask, aLocal, bLocal, CMPMODE::LE, Lc);
        }

        Select(outHalf, mask, ones, (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, Lc);
        Cast(outI8, outHalf, RoundMode::CAST_RINT, Lc);
    }

    __aicore__ inline void ProcessFast() {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t off = blockIdx * perCore;
        if (off >= totalElements) return;
        uint32_t myElems = perCore;
        if (off + myElems > totalElements) myElems = totalElements - off;

        DataCopyPadExtParams<DT_X1> pad{false, 0, 0, (DT_X1)0};

        for (uint32_t t = 0; t < myElems; t += tileSize) {
            uint32_t L = tileSize;
            if (t + L > myElems) L = myElems - t;
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

    __aicore__ inline void initRowState(uint32_t r0) {
        x1Base = 0;
        x2Base = 0;
        uint32_t rem = r0;
        for (uint32_t d = 0; d + 1 < ndim; ++d) {
            uint32_t mult = 1;
            for (uint32_t e = d + 1; e + 1 < ndim; ++e) mult *= outShape[e];
            uint32_t id = (mult == 0) ? 0 : (rem / mult);
            rem = (mult == 0) ? rem : (rem % mult);
            idx[d] = id;
            x1Base += (int32_t)id * x1StrideArr[d];
            x2Base += (int32_t)id * x2StrideArr[d];
        }
    }

    __aicore__ inline void advanceRowState() {
        if (ndim < 2) return;
        int32_t d = (int32_t)ndim - 2;
        idx[d]++;
        x1Base += x1StrideArr[d];
        x2Base += x2StrideArr[d];
        while (d > 0 && idx[d] >= outShape[d]) {
            x1Base -= (int32_t)outShape[d] * x1StrideArr[d];
            x2Base -= (int32_t)outShape[d] * x2StrideArr[d];
            idx[d] = 0;
            d--;
            idx[d]++;
            x1Base += x1StrideArr[d];
            x2Base += x2StrideArr[d];
        }
    }

    __aicore__ inline void LoadOperand(TQue<TPosition::VECIN, 2> &q,
                                       const GlobalTensor<DT_X1> &gm,
                                       int32_t baseIdx, int32_t lastStride,
                                       uint32_t c, uint32_t chunk, uint32_t Lc,
                                       LocalTensor<DT_X1> &out) {
        DataCopyPadExtParams<DT_X1> pad{false, 0, 0, (DT_X1)0};
        LocalTensor<DT_X1> loc = q.AllocTensor<DT_X1>();
        if (lastStride == 0) {
            DataCopyPad(loc, gm[(uint32_t)baseIdx],
                        DataCopyExtParams{1, (uint32_t)sizeof(DT_X1), 0, 0, 0}, pad);
            q.EnQue(loc);
            loc = q.DeQue<DT_X1>();
            if constexpr (!std::is_same_v<DT_X1, int8_t>) {
                DT_X1 v = loc.GetValue(0);
                Duplicate(loc, v, Lc);
            }
        } else {
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
        if (rowOff >= totalRows) return;
        uint32_t myRows = perCore;
        if (rowOff + myRows > totalRows) myRows = totalRows - rowOff;

        uint32_t L = lastDimLen;
        int32_t s1last = x1StrideArr[ndim - 1];
        int32_t s2last = x2StrideArr[ndim - 1];

        initRowState(rowOff);

        for (uint32_t r = 0; r < myRows; ++r) {
            uint32_t outRowBase = (rowOff + r) * L;
            for (uint32_t c = 0; c < L; c += tileSize) {
                uint32_t chunk = tileSize;
                if (c + chunk > L) chunk = L - c;
                uint32_t Lc = RoundUp256(chunk);

                LocalTensor<DT_X1> aLocal;
                LocalTensor<DT_X1> bLocal;
                LoadOperand(qX1, x1Gm, x1Base, s1last, c, chunk, Lc, aLocal);
                LoadOperand(qX2, x2Gm, x2Base, s2last, c, chunk, Lc, bLocal);

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
            if (r + 1 < myRows) advanceRowState();
        }
    }

    TPipe pipe;
    TQue<TPosition::VECIN, 2> qX1;
    TQue<TPosition::VECIN, 2> qX2;
    TQue<TPosition::VECOUT, 2> qOut;
    TBuf<TPosition::VECCALC> bufMask;
    TBuf<TPosition::VECCALC> bufOnes;
    TBuf<TPosition::VECCALC> bufOutHalf;
    TBuf<TPosition::VECCALC> bufAHalf;
    TBuf<TPosition::VECCALC> bufBHalf;
    TBuf<TPosition::VECCALC> bufI32;

    GlobalTensor<DT_X1> x1Gm;
    GlobalTensor<DT_X1> x2Gm;
    GlobalTensor<int8_t> yGm;

    uint32_t mode;
    uint32_t totalElements;
    uint32_t tileSize;
    uint32_t perCore;
    uint32_t ndim;
    uint32_t lastDimLen;
    uint32_t totalRows;
    uint32_t outShape[LE_MAX_DIM];
    int32_t  x1StrideArr[LE_MAX_DIM];
    int32_t  x2StrideArr[LE_MAX_DIM];

    int32_t x1Base;
    int32_t x2Base;
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

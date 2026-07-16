// Kernel侧核函数实现
// 优化：DataCopy替代DataCopyPad + scalar×scalar广播快速路径 + 双缓冲流水线
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
        bufferNum = td.bufferNum;
        if (mode == 1) {
            for (uint32_t i = 0; i < LE_MAX_DIM; ++i) {
                outShape[i] = td.outShape[i];
                x1StrideArr[i] = td.x1Stride[i];
                x2StrideArr[i] = td.x2Stride[i];
            }
        }
        x1Gm.SetGlobalBuffer((__gm__ DT_X1 *)x1);
        x2Gm.SetGlobalBuffer((__gm__ DT_X1 *)x2);
        yGm.SetGlobalBuffer((__gm__ int8_t *)y);

        if (totalElements == 0) { return; }

        pipe.InitBuffer(qX1, bufferNum, tileSize * (uint32_t)sizeof(DT_X1));
        pipe.InitBuffer(qX2, bufferNum, tileSize * (uint32_t)sizeof(DT_X1));
        pipe.InitBuffer(qOut, bufferNum, tileSize * (uint32_t)sizeof(int8_t));
        pipe.InitBuffer(bufMask, ((tileSize / 8) + 31) / 32 * 32);
        pipe.InitBuffer(bufOnes, tileSize * (uint32_t)sizeof(half));
        pipe.InitBuffer(bufOutHalf, tileSize * (uint32_t)sizeof(half));

        if constexpr (std::is_same_v<DT_X1, int32_t>) {
            pipe.InitBuffer(bufI32, tileSize * (uint32_t)sizeof(int32_t));
        }
        if constexpr (std::is_same_v<DT_X1, int8_t>) {
            pipe.InitBuffer(bufAHalf, tileSize * (uint32_t)sizeof(half));
            pipe.InitBuffer(bufBHalf, tileSize * (uint32_t)sizeof(half));
        }

        LocalTensor<half> ones = bufOnes.Get<half>();
        Duplicate(ones, (half)1.0, tileSize);

        if (mode == 1 && ndim >= 2) {
            uint32_t mult = 1;
            for (int32_t d = (int32_t)ndim - 2; d >= 0; --d) {
                rowMult[d] = mult;
                mult *= outShape[d];
            }
        }
    }

    __aicore__ inline void Process() {
        if (totalElements == 0) { return; }
        if (mode == 0) { ProcessFast(); }
        else            { ProcessBcast(); }
    }

private:
    __aicore__ inline uint32_t RoundUp256(uint32_t v) {
        return (v + 255u) & ~255u;
    }

    // ---- 通用计算核心 ----
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
        Select(outHalf, mask, ones, (half)0, SELMODE::VSEL_TENSOR_SCALAR_MODE, Lc);
        Cast(outI8, outHalf, RoundMode::CAST_RINT, Lc);
    }

    // ---- scalar×scalar 比较：直接读 GM 标量，一次比较填整行 ----
    // 仅 int32/int8 支持 AICore 标量比较；half/float 走正常 pipeline
    __aicore__ inline void FillRowScalar(uint32_t outRowBase, uint32_t L, int8_t val) {
        LocalTensor<half> tmp = bufOutHalf.Get<half>();
        Duplicate(tmp, (half)(float)(int32_t)val, L);
        LocalTensor<int8_t> outI8 = qOut.AllocTensor<int8_t>();
        Cast(outI8, tmp, RoundMode::CAST_RINT, L);
        qOut.EnQue(outI8);
        outI8 = qOut.DeQue<int8_t>();
        DataCopyPad(yGm[outRowBase], outI8,
                    DataCopyExtParams{1, L, 0, 0, 0});
        qOut.FreeTensor(outI8);
    }

    // ==================== FAST 路径 ====================
    __aicore__ inline void ProcessFast() {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t off = blockIdx * perCore;
        if (off >= totalElements) { return; }
        uint32_t myElems = perCore;
        if (off + myElems > totalElements) { myElems = totalElements - off; }

        if (myElems <= tileSize) { ProcessFastSingle(off, myElems); return; }

        // 多 tile 双缓冲流水线
        {
            LocalTensor<DT_X1> a = qX1.AllocTensor<DT_X1>();
            DataCopy(a, x1Gm[off], tileSize);
            qX1.EnQue(a);
            LocalTensor<DT_X1> b = qX2.AllocTensor<DT_X1>();
            DataCopy(b, x2Gm[off], tileSize);
            qX2.EnQue(b);
        }

        for (uint32_t t = 0; t < myElems; t += tileSize) {
            uint32_t L = tileSize;
            if (t + L > myElems) { L = myElems - t; }
            uint32_t Lc = RoundUp256(L);
            uint32_t base = off + t;

            if (t > 0) {
                LocalTensor<int8_t> prevOut = qOut.DeQue<int8_t>();
                DataCopy(yGm[base - tileSize], prevOut, tileSize);
                qOut.FreeTensor(prevOut);
            }

            LocalTensor<DT_X1> aLocal = qX1.DeQue<DT_X1>();
            LocalTensor<DT_X1> bLocal = qX2.DeQue<DT_X1>();

            if (t + tileSize < myElems) {
                uint32_t nextOff = off + t + tileSize;
                uint32_t nxtL = tileSize;
                if (nextOff + nxtL > off + myElems) { nxtL = off + myElems - nextOff; }
                LocalTensor<DT_X1> na = qX1.AllocTensor<DT_X1>();
                DataCopy(na, x1Gm[nextOff], nxtL);
                qX1.EnQue(na);
                LocalTensor<DT_X1> nb = qX2.AllocTensor<DT_X1>();
                DataCopy(nb, x2Gm[nextOff], nxtL);
                qX2.EnQue(nb);
            }

            LocalTensor<int8_t> outI8 = qOut.AllocTensor<int8_t>();
            ComputeTile(aLocal, bLocal, outI8, Lc, false, false);
            qX1.FreeTensor(aLocal);
            qX2.FreeTensor(bLocal);
            qOut.EnQue(outI8);
        }

        {
            uint32_t lastIdx = (myElems - 1u) / tileSize * tileSize;
            uint32_t lastL = myElems - lastIdx;
            LocalTensor<int8_t> lastOut = qOut.DeQue<int8_t>();
            DataCopy(yGm[off + lastIdx], lastOut, lastL);
            qOut.FreeTensor(lastOut);
        }
    }

    // 单 tile 路径：DataCopy 直接拷贝，无额外队列开销
    __aicore__ inline void ProcessFastSingle(uint32_t off, uint32_t n) {
        uint32_t Lc = RoundUp256(n);
        LocalTensor<DT_X1> aLocal = qX1.AllocTensor<DT_X1>();
        DataCopy(aLocal, x1Gm[off], n);
        qX1.EnQue(aLocal);
        LocalTensor<DT_X1> bLocal = qX2.AllocTensor<DT_X1>();
        DataCopy(bLocal, x2Gm[off], n);
        qX2.EnQue(bLocal);
        aLocal = qX1.DeQue<DT_X1>();
        bLocal = qX2.DeQue<DT_X1>();
        LocalTensor<int8_t> outI8 = qOut.AllocTensor<int8_t>();
        ComputeTile(aLocal, bLocal, outI8, Lc, false, false);
        qX1.FreeTensor(aLocal);
        qX2.FreeTensor(bLocal);
        qOut.EnQue(outI8);
        outI8 = qOut.DeQue<int8_t>();
        DataCopy(yGm[off], outI8, n);
        qOut.FreeTensor(outI8);
    }

    // ==================== BCAST 路径 ====================
    __aicore__ inline void initRowState(uint32_t r0) {
        x1Base = 0; x2Base = 0;
        uint32_t rem = r0;
        for (uint32_t d = 0; d + 1u < ndim; ++d) {
            uint32_t mult = rowMult[d];
            uint32_t id = (mult == 0) ? 0 : (rem / mult);
            rem -= id * mult;
            idx[d] = id;
            x1Base += (int32_t)id * x1StrideArr[d];
            x2Base += (int32_t)id * x2StrideArr[d];
        }
    }

    __aicore__ inline void advanceRowState() {
        if (ndim < 2) { return; }
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

    __aicore__ inline void EnqueOperand(TQue<TPosition::VECIN, 1> &q,
                                         const GlobalTensor<DT_X1> &gm,
                                         int32_t baseIdx, int32_t lastStride,
                                         uint32_t c, uint32_t chunk) {
        DataCopyPadExtParams<DT_X1> pad{false, 0, 0, (DT_X1)0};
        LocalTensor<DT_X1> loc = q.AllocTensor<DT_X1>();
        if (lastStride == 0) {
            DataCopyPad(loc, gm[(uint32_t)baseIdx],
                        DataCopyExtParams{1, (uint32_t)sizeof(DT_X1), 0, 0, 0}, pad);
        } else {
            DataCopy(loc, gm[(uint32_t)baseIdx + c], chunk);
        }
        q.EnQue(loc);
    }

    __aicore__ inline void MaterializeDequeued(LocalTensor<DT_X1> &loc, bool isScalar,
                                                uint32_t Lc) {
        if constexpr (!std::is_same_v<DT_X1, int8_t>) {
            if (isScalar) {
                DT_X1 v = loc.GetValue(0);
                Duplicate(loc, v, Lc);
            }
        }
    }

    __aicore__ inline void ProcessBcast() {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t rowOff = blockIdx * perCore;
        if (rowOff >= totalRows) { return; }
        uint32_t myRows = perCore;
        if (rowOff + myRows > totalRows) { myRows = totalRows - rowOff; }

        uint32_t L = lastDimLen;
        int32_t s1last = x1StrideArr[ndim - 1];
        int32_t s2last = x2StrideArr[ndim - 1];
        bool aScalar = (s1last == 0);
        bool bScalar = (s2last == 0);

        initRowState(rowOff);

        // ===== scalar×scalar 快速路径：一行只做一次比较 =====
        // 仅 int32/int8 的标量比较在 AICore 上合法；half/float 回退正常 pipeline
        if constexpr (std::is_same_v<DT_X1, int32_t> || std::is_same_v<DT_X1, int8_t>) {
            if (aScalar && bScalar) {
                for (uint32_t r = 0; r < myRows; ++r) {
                    DT_X1 va = x1Gm.GetValue((uint32_t)x1Base);
                    DT_X1 vb = x2Gm.GetValue((uint32_t)x2Base);
                    int8_t res = (va <= vb) ? 1 : 0;
                    FillRowScalar((rowOff + r) * L, L, res);
                    if (r + 1 < myRows) { advanceRowState(); }
                }
                return;
            }
        }

        // ===== 单列 tile 快速路径 =====
        if (L <= tileSize) {
            uint32_t Lc = RoundUp256(L);
            for (uint32_t r = 0; r < myRows; ++r) {
                EnqueOperand(qX1, x1Gm, x1Base, s1last, 0, L);
                EnqueOperand(qX2, x2Gm, x2Base, s2last, 0, L);
                LocalTensor<DT_X1> aLoc = qX1.DeQue<DT_X1>();
                LocalTensor<DT_X1> bLoc = qX2.DeQue<DT_X1>();
                MaterializeDequeued(aLoc, aScalar, Lc);
                MaterializeDequeued(bLoc, bScalar, Lc);
                LocalTensor<int8_t> outI8 = qOut.AllocTensor<int8_t>();
                ComputeTile(aLoc, bLoc, outI8, Lc, aScalar, bScalar);
                qX1.FreeTensor(aLoc);
                qX2.FreeTensor(bLoc);
                qOut.EnQue(outI8);
                outI8 = qOut.DeQue<int8_t>();
                DataCopyPad(yGm[(rowOff + r) * L], outI8,
                            DataCopyExtParams{1, L, 0, 0, 0});
                qOut.FreeTensor(outI8);
                if (r + 1 < myRows) { advanceRowState(); }
            }
            return;
        }

        // ===== 多列 tile 双缓冲流水线 =====
        for (uint32_t r = 0; r < myRows; ++r) {
            uint32_t outRowBase = (rowOff + r) * L;
            EnqueOperand(qX1, x1Gm, x1Base, s1last, 0, tileSize);
            EnqueOperand(qX2, x2Gm, x2Base, s2last, 0, tileSize);

            for (uint32_t c = 0; c < L; c += tileSize) {
                uint32_t chunk = tileSize;
                if (c + chunk > L) { chunk = L - c; }
                uint32_t Lc = RoundUp256(chunk);

                if (c > 0) {
                    LocalTensor<int8_t> prevOut = qOut.DeQue<int8_t>();
                    DataCopyPad(yGm[outRowBase + c - tileSize], prevOut,
                                DataCopyExtParams{1, tileSize, 0, 0, 0});
                    qOut.FreeTensor(prevOut);
                }

                LocalTensor<DT_X1> aLoc = qX1.DeQue<DT_X1>();
                LocalTensor<DT_X1> bLoc = qX2.DeQue<DT_X1>();
                MaterializeDequeued(aLoc, aScalar, Lc);
                MaterializeDequeued(bLoc, bScalar, Lc);

                if (c + tileSize < L) {
                    uint32_t nc = c + tileSize;
                    uint32_t nchunk = tileSize;
                    if (nc + nchunk > L) { nchunk = L - nc; }
                    EnqueOperand(qX1, x1Gm, x1Base, s1last, nc, nchunk);
                    EnqueOperand(qX2, x2Gm, x2Base, s2last, nc, nchunk);
                }

                LocalTensor<int8_t> outI8 = qOut.AllocTensor<int8_t>();
                ComputeTile(aLoc, bLoc, outI8, Lc, aScalar, bScalar);
                qX1.FreeTensor(aLoc);
                qX2.FreeTensor(bLoc);
                qOut.EnQue(outI8);
            }

            {
                uint32_t lastC = (L - 1u) / tileSize * tileSize;
                uint32_t lastChunk = L - lastC;
                LocalTensor<int8_t> lastOut = qOut.DeQue<int8_t>();
                DataCopyPad(yGm[outRowBase + lastC], lastOut,
                            DataCopyExtParams{1, lastChunk, 0, 0, 0});
                qOut.FreeTensor(lastOut);
            }

            if (r + 1 < myRows) { advanceRowState(); }
        }
    }

    // ---- Pipe / 队列 / 缓冲 ----
    TPipe pipe;
    TQue<TPosition::VECIN, 1> qX1, qX2;
    TQue<TPosition::VECOUT, 1> qOut;
    TBuf<TPosition::VECCALC> bufMask, bufOnes, bufOutHalf, bufAHalf, bufBHalf, bufI32;
    GlobalTensor<DT_X1> x1Gm, x2Gm;
    GlobalTensor<int8_t> yGm;

    uint32_t mode, totalElements, tileSize, perCore, ndim, lastDimLen, totalRows, bufferNum;
    uint32_t outShape[LE_MAX_DIM];
    int32_t  x1StrideArr[LE_MAX_DIM], x2StrideArr[LE_MAX_DIM];
    uint32_t rowMult[LE_MAX_DIM];
    int32_t  x1Base, x2Base;
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

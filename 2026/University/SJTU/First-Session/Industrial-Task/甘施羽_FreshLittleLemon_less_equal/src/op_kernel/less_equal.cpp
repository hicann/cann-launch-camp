// Kernel侧核函数实现
#include "kernel_operator.h"

#include "less_equal_tiling.h"
#include "tiling_key_less_equal.h"

using namespace AscendC;

namespace {
constexpr uint32_t BUFFER_NUM = 2;
constexpr int32_t CMP_MODE_LE = 3;  // CMPMODE::LE
constexpr int32_t CMP_MODE_EQ = 2;  // CMPMODE::EQ

__aicore__ inline uint32_t LeCeilDiv(uint32_t a, uint32_t b) {
    return b == 0 ? 0 : (a + b - 1) / b;
}
__aicore__ inline uint32_t LeAlignUp(uint32_t v, uint32_t a) {
    return (v + a - 1) / a * a;
}
__aicore__ inline uint32_t LeMin(uint32_t a, uint32_t b) { return a < b ? a : b; }

// 编译期决定 Compare 使用的计算类型：int8 需先 Cast 到 half；其余用原生类型
template <typename T>
struct CmpType { using type = T; };
template <>
struct CmpType<int8_t> { using type = half; };
}  // namespace

template <typename T>
class KernelLessEqual {
public:
    using C = typename CmpType<T>::type;              // Compare 计算类型

    __aicore__ inline KernelLessEqual() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const LessEqualTilingData &t) {
        totalLength = t.totalLength;
        tileLength = t.tileLength;
        blockDim = t.blockDim;
        needBroadcast = t.needBroadcast;
        ndim = t.ndim;
        for (uint32_t d = 0; d < LESS_EQUAL_MAX_DIM; d++) {
            shapeOut[d] = t.shapeOut[d];
            dimStride[d] = t.dimStride[d];
            strideX1[d] = t.strideX1[d];
            strideX2[d] = t.strideX2[d];
        }

        x1Gm.SetGlobalBuffer((__gm__ T *)x1);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2);
        yGm.SetGlobalBuffer((__gm__ int8_t *)y);

        // 输入/输出队列（double buffer）
        pipe.InitBuffer(inQueueX1, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(int8_t));

        // 计算用中间 buffer
        if (!IsSameType<T, C>::value) {
            pipe.InitBuffer(castX1Buf, tileLength * sizeof(C));
            pipe.InitBuffer(castX2Buf, tileLength * sizeof(C));
        }
        pipe.InitBuffer(maskBuf, tileLength * sizeof(uint8_t));
        // Select 用 half 常量：1.0 / 0.0，一次性填好（cnt<=tileLength 全覆盖）
        pipe.InitBuffer(oneBuf, tileLength * sizeof(half));
        pipe.InitBuffer(zeroBuf, tileLength * sizeof(half));
        pipe.InitBuffer(resBuf, tileLength * sizeof(half));
        // int32 走 min+eq 需要一个 int32 临时 buffer
        if constexpr (IsSameType<T, int32_t>::value) {
            pipe.InitBuffer(minBuf, tileLength * sizeof(int32_t));
        }

        // 常量 0/1 只需初始化一次，移出逐 tile 循环
        LocalTensor<half> oneT = oneBuf.Get<half>();
        LocalTensor<half> zeroT = zeroBuf.Get<half>();
        Duplicate(oneT, static_cast<half>(1.0f), tileLength);
        Duplicate(zeroT, static_cast<half>(0.0f), tileLength);
    }

    __aicore__ inline void Process() {
        if (totalLength == 0) {
            return;  // 空张量
        }
        if (needBroadcast == 0) {
            ProcessFast();
        } else {
            ProcessBroadcast();
        }
    }

private:
    // ---------- Fast path: x1、x2 同形状，1D 展平逐块处理 ----------
    __aicore__ inline void ProcessFast() {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t elemPerCore = LeCeilDiv(totalLength, blockDim);
        uint32_t coreStart = blockIdx * elemPerCore;
        if (coreStart >= totalLength) {
            return;
        }
        uint32_t coreElems = LeMin(elemPerCore, totalLength - coreStart);

        uint32_t numTiles = LeCeilDiv(coreElems, tileLength);
        for (uint32_t tile = 0; tile < numTiles; tile++) {
            uint32_t offset = coreStart + tile * tileLength;
            uint32_t n = LeMin(tileLength, coreStart + coreElems - offset);

            // CopyIn 两个输入
            LocalTensor<T> x1Local = inQueueX1.AllocTensor<T>();
            LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();
            DataCopyExtParams cp{1, n * (uint32_t)sizeof(T), 0, 0, 0};
            DataCopyPadExtParams<T> pad{false, 0, 0, (T)0};
            DataCopyPad(x1Local, x1Gm[offset], cp, pad);
            DataCopyPad(x2Local, x2Gm[offset], cp, pad);
            inQueueX1.EnQue(x1Local);
            inQueueX2.EnQue(x2Local);

            LocalTensor<T> a = inQueueX1.DeQue<T>();
            LocalTensor<T> b = inQueueX2.DeQue<T>();
            LocalTensor<int8_t> outLocal = outQueueY.AllocTensor<int8_t>();

            ComputeLE(a, b, outLocal, n);

            inQueueX1.FreeTensor(a);
            inQueueX2.FreeTensor(b);
            outQueueY.EnQue(outLocal);

            LocalTensor<int8_t> o = outQueueY.DeQue<int8_t>();
            DataCopyExtParams cpo{1, n * (uint32_t)sizeof(int8_t), 0, 0, 0};
            DataCopyPad(yGm[offset], o, cpo);
            outQueueY.FreeTensor(o);
        }
    }

    // ---------- Broadcast path: 按输出“行”(最后一维)处理 ----------
    __aicore__ inline void ProcessBroadcast() {
        uint32_t lastDim = shapeOut[ndim - 1];
        if (lastDim == 0) {
            return;
        }
        uint32_t numRows = totalLength / lastDim;

        uint32_t blockIdx = GetBlockIdx();
        uint32_t rowsPerCore = LeCeilDiv(numRows, blockDim);
        uint32_t rowStart = blockIdx * rowsPerCore;
        if (rowStart >= numRows) {
            return;
        }
        uint32_t rowEnd = LeMin(rowStart + rowsPerCore, numRows);

        // 快路径：整行可装入一个 tile，且某操作数在“行”维度不变（外层 stride 全 0），
        // 则该操作数一行只需从 GM 读一次并在本核内复用，消除按行重复的冗余搬运。
        bool rowFits = (lastDim <= tileLength);
        bool inv1 = OuterAllZero(strideX1);  // x1 跨行不变
        bool inv2 = OuterAllZero(strideX2);  // x2 跨行不变
        if (rowFits && (inv1 || inv2)) {
            ProcessBroadcastCached(rowStart, rowEnd, lastDim, inv1, inv2);
            return;
        }

        // 通用回退：逐行、必要时按 tile 拆分
        uint32_t outerIdx[LESS_EQUAL_MAX_DIM];
        uint32_t base1 = 0, base2 = 0;
        InitRowState(rowStart, outerIdx, base1, base2);
        for (uint32_t row = rowStart; row < rowEnd; row++) {
            uint32_t dstRowOffset = row * lastDim;
            ProcessOneRow(base1, base2, dstRowOffset, lastDim);
            if (row + 1 < rowEnd) {
                AdvanceRowState(outerIdx, base1, base2);
            }
        }
    }

    // 外层维（[0, ndim-2]）stride 是否全 0（该操作数跨行不变）
    __aicore__ inline bool OuterAllZero(const uint32_t (&stride)[LESS_EQUAL_MAX_DIM]) {
        for (uint32_t d = 0; d + 1 < ndim; d++) {
            if (stride[d] != 0) {
                return false;
            }
        }
        return true;
    }

    // 缓存快路径：整行一个 tile；不变操作数只加载一次并跨行复用
    __aicore__ inline void ProcessBroadcastCached(uint32_t rowStart, uint32_t rowEnd,
                                                  uint32_t lastDim, bool inv1, bool inv2) {
        bool bl1 = (strideX1[ndim - 1] == 0);  // x1 最后一维是否广播（整行同值）
        bool bl2 = (strideX2[ndim - 1] == 0);

        // 不变操作数：加载一次并持有（不 Free），跨行复用
        LocalTensor<T> cached1, cached2;
        if (inv1) {
            cached1 = inQueueX1.AllocTensor<T>();
            CopyInSeg(cached1, x1Gm, 0, bl1, 0, lastDim);
            inQueueX1.EnQue(cached1);
            cached1 = inQueueX1.DeQue<T>();
            if (bl1) FillVal(cached1, cached1.GetValue(0), lastDim);
        }
        if (inv2) {
            cached2 = inQueueX2.AllocTensor<T>();
            CopyInSeg(cached2, x2Gm, 0, bl2, 0, lastDim);
            inQueueX2.EnQue(cached2);
            cached2 = inQueueX2.DeQue<T>();
            if (bl2) FillVal(cached2, cached2.GetValue(0), lastDim);
        }

        // 变化操作数的按行基址：用混合进制计数器推进
        uint32_t outerIdx[LESS_EQUAL_MAX_DIM];
        uint32_t base1 = 0, base2 = 0;
        InitRowState(rowStart, outerIdx, base1, base2);

        for (uint32_t row = rowStart; row < rowEnd; row++) {
            uint32_t dstRowOffset = row * lastDim;

            // 取得两操作数的行 tensor：不变则用缓存，变化则本行加载
            LocalTensor<T> a, b;
            LocalTensor<T> fresh1, fresh2;
            if (inv1) {
                a = cached1;
            } else {
                fresh1 = inQueueX1.AllocTensor<T>();
                CopyInSeg(fresh1, x1Gm, base1, bl1, 0, lastDim);
                inQueueX1.EnQue(fresh1);
                fresh1 = inQueueX1.DeQue<T>();
                if (bl1) FillVal(fresh1, fresh1.GetValue(0), lastDim);
                a = fresh1;
            }
            if (inv2) {
                b = cached2;
            } else {
                fresh2 = inQueueX2.AllocTensor<T>();
                CopyInSeg(fresh2, x2Gm, base2, bl2, 0, lastDim);
                inQueueX2.EnQue(fresh2);
                fresh2 = inQueueX2.DeQue<T>();
                if (bl2) FillVal(fresh2, fresh2.GetValue(0), lastDim);
                b = fresh2;
            }

            LocalTensor<int8_t> outLocal = outQueueY.AllocTensor<int8_t>();
            ComputeLE(a, b, outLocal, lastDim);
            if (!inv1) inQueueX1.FreeTensor(fresh1);
            if (!inv2) inQueueX2.FreeTensor(fresh2);
            outQueueY.EnQue(outLocal);

            LocalTensor<int8_t> o = outQueueY.DeQue<int8_t>();
            DataCopyExtParams cpo{1, lastDim * (uint32_t)sizeof(int8_t), 0, 0, 0};
            DataCopyPad(yGm[dstRowOffset], o, cpo);
            outQueueY.FreeTensor(o);

            if (row + 1 < rowEnd) {
                AdvanceRowState(outerIdx, base1, base2);
            }
        }

        if (inv1) inQueueX1.FreeTensor(cached1);
        if (inv2) inQueueX2.FreeTensor(cached2);
    }

    // 由行平坦索引反推外层多维索引，并计算输入基址（仅首行用除法/取模）
    __aicore__ inline void InitRowState(uint32_t rowFlat, uint32_t (&outerIdx)[LESS_EQUAL_MAX_DIM],
                                        uint32_t &base1, uint32_t &base2) {
        base1 = 0;
        base2 = 0;
        uint32_t remaining = rowFlat;
        // 外层维为 [0, ndim-2]，其 dimStride 需除以 lastDim 得到“按行”的步长
        uint32_t lastDim = shapeOut[ndim - 1];
        for (uint32_t d = 0; d + 1 < ndim; d++) {
            uint32_t rowStrideD = dimStride[d] / lastDim;  // 每单位该维索引跨越的行数
            uint32_t i_d = (rowStrideD == 0) ? 0 : (remaining / rowStrideD);
            if (rowStrideD != 0) {
                remaining = remaining % rowStrideD;
            }
            outerIdx[d] = i_d;
            base1 += i_d * strideX1[d];
            base2 += i_d * strideX2[d];
        }
        outerIdx[ndim - 1] = 0;
    }

    // 混合进制计数器：外层索引 +1，进位；同步维护 base1/base2（仅加减）
    __aicore__ inline void AdvanceRowState(uint32_t (&outerIdx)[LESS_EQUAL_MAX_DIM],
                                           uint32_t &base1, uint32_t &base2) {
        int32_t d = (int32_t)ndim - 2;  // 最内层的外层维
        while (d >= 0) {
            outerIdx[d]++;
            base1 += strideX1[d];
            base2 += strideX2[d];
            if (outerIdx[d] < shapeOut[d]) {
                break;
            }
            // 溢出：回退该维，进位到上一维
            base1 -= shapeOut[d] * strideX1[d];
            base2 -= shapeOut[d] * strideX2[d];
            outerIdx[d] = 0;
            d--;
        }
    }

    // 处理一行（长度 lastDim），必要时按 tileLength 分块
    __aicore__ inline void ProcessOneRow(uint32_t base1, uint32_t base2,
                                         uint32_t dstRowOffset, uint32_t lastDim) {
        bool bcast1 = (strideX1[ndim - 1] == 0);  // x1 在最后一维是否广播
        bool bcast2 = (strideX2[ndim - 1] == 0);

        uint32_t done = 0;
        while (done < lastDim) {
            uint32_t n = LeMin(tileLength, lastDim - done);

            // CopyIn：广播输入只搬 1 个元素，否则搬 n 个
            LocalTensor<T> x1Local = inQueueX1.AllocTensor<T>();
            LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();
            CopyInSeg(x1Local, x1Gm, base1, bcast1, done, n);
            CopyInSeg(x2Local, x2Gm, base2, bcast2, done, n);
            inQueueX1.EnQue(x1Local);
            inQueueX2.EnQue(x2Local);

            LocalTensor<T> a = inQueueX1.DeQue<T>();
            LocalTensor<T> b = inQueueX2.DeQue<T>();
            // DeQue 后读取标量安全：广播输入把单元素填满整段
            if (bcast1) {
                FillVal(a, a.GetValue(0), n);
            }
            if (bcast2) {
                FillVal(b, b.GetValue(0), n);
            }

            LocalTensor<int8_t> outLocal = outQueueY.AllocTensor<int8_t>();
            ComputeLE(a, b, outLocal, n);

            inQueueX1.FreeTensor(a);
            inQueueX2.FreeTensor(b);
            outQueueY.EnQue(outLocal);

            LocalTensor<int8_t> o = outQueueY.DeQue<int8_t>();
            DataCopyExtParams cpo{1, n * (uint32_t)sizeof(int8_t), 0, 0, 0};
            DataCopyPad(yGm[dstRowOffset + done], o, cpo);
            outQueueY.FreeTensor(o);

            done += n;
        }
    }

    // CopyIn 一行的一段：广播维只搬 1 个元素到 dst[0]；否则连续搬 n 个
    __aicore__ inline void CopyInSeg(const LocalTensor<T> &dst, const GlobalTensor<T> &src,
                                     uint32_t base, bool bcast, uint32_t segStart, uint32_t n) {
        DataCopyPadExtParams<T> pad{false, 0, 0, (T)0};
        if (bcast) {
            DataCopyExtParams cp1{1, (uint32_t)sizeof(T), 0, 0, 0};
            DataCopyPad(dst, src[base], cp1, pad);
        } else {
            DataCopyExtParams cp{1, n * (uint32_t)sizeof(T), 0, 0, 0};
            DataCopyPad(dst, src[base + segStart], cp, pad);
        }
    }

    // 用标量值填充 dst[0..n)。Duplicate 不支持 int8，int8 走标量循环。
    __aicore__ inline void FillVal(const LocalTensor<T> &dst, T v, uint32_t n) {
        if constexpr (IsSameType<T, int8_t>::value) {
            for (uint32_t i = 0; i < n; i++) {
                dst.SetValue(i, v);
            }
        } else {
            Duplicate(dst, v, n);
        }
    }

    // 核心比较：a<=b -> out(int8, 0/1)，长度 n
    __aicore__ inline void ComputeLE(const LocalTensor<T> &a, const LocalTensor<T> &b,
                                     const LocalTensor<int8_t> &out, uint32_t n) {
        // Compare/Select/Cast 的 count 需满足 count*sizeof 为 256B 的整数倍。
        // 对齐到 256 元素（half:512B, float/int32:1024B 均为 256 倍数），
        // 上界不超过 tileLength（tileLength 已是 256 的整数倍）。尾部 [n,cnt) 为
        // 未初始化数据，但输出只写回 n 个元素，故无影响。
        uint32_t cnt = LeMin(LeAlignUp(n, 256), tileLength);

        LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
        LocalTensor<half> oneT = oneBuf.Get<half>();
        LocalTensor<half> zeroT = zeroBuf.Get<half>();
        LocalTensor<half> resT = resBuf.Get<half>();

        if constexpr (IsSameType<T, int32_t>::value) {
            // int32 硬件 Compare 仅支持 EQ，不支持 LE。
            // 利用恒等式 (a<=b) <=> (min(a,b)==a)：Min 与 EQ 对 int32 均精确。
            LocalTensor<int32_t> mn = minBuf.Get<int32_t>();
            Min(mn, a, b, cnt);
            Compare(mask, mn, a, static_cast<CMPMODE>(CMP_MODE_EQ), cnt);
        } else if constexpr (IsSameType<T, C>::value) {
            Compare(mask, a, b, static_cast<CMPMODE>(CMP_MODE_LE), cnt);
        } else {
            // int8 需先 Cast 到 half 再比较（Compare 不支持 int8）
            LocalTensor<C> ac = castX1Buf.Get<C>();
            LocalTensor<C> bc = castX2Buf.Get<C>();
            Cast(ac, a, RoundMode::CAST_NONE, cnt);
            Cast(bc, b, RoundMode::CAST_NONE, cnt);
            Compare(mask, ac, bc, static_cast<CMPMODE>(CMP_MODE_LE), cnt);
        }

        // 位掩码 -> half 0/1 -> int8 输出（half->int8 有直接 Cast 通路）
        Select(resT, mask, oneT, zeroT, SELMODE::VSEL_TENSOR_TENSOR_MODE, cnt);
        Cast(out, resT, RoundMode::CAST_RINT, cnt);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX1;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX2;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<TPosition::VECCALC> castX1Buf;
    TBuf<TPosition::VECCALC> castX2Buf;
    TBuf<TPosition::VECCALC> maskBuf;
    TBuf<TPosition::VECCALC> oneBuf;
    TBuf<TPosition::VECCALC> zeroBuf;
    TBuf<TPosition::VECCALC> resBuf;
    TBuf<TPosition::VECCALC> minBuf;

    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<int8_t> yGm;

    uint32_t totalLength;
    uint32_t tileLength;
    uint32_t blockDim;
    uint32_t needBroadcast;
    uint32_t ndim;
    uint32_t shapeOut[LESS_EQUAL_MAX_DIM];
    uint32_t dimStride[LESS_EQUAL_MAX_DIM];
    uint32_t strideX1[LESS_EQUAL_MAX_DIM];
    uint32_t strideX2[LESS_EQUAL_MAX_DIM];
};

template <typename DT_X1>
__global__ __aicore__ void less_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LessEqualTilingData);
    GET_TILING_DATA_WITH_STRUCT(LessEqualTilingData, tiling_data, tiling);
    KernelLessEqual<DT_X1> op;
    op.Init(x1, x2, y, tiling_data);
    op.Process();
}

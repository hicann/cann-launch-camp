#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

// =========================================================================
// 通用实现：用于硬件直接支持基础数学运算的 float 和 half 类型
// =========================================================================
template<typename T>
class KernelLogSigmoid {
public:
    __aicore__ inline KernelLogSigmoid() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t alignNum) {
        this->alignNum = alignNum;
        uint32_t blockId = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();

        uint32_t elementsPerCore = (totalLength + blockNum - 1) / blockNum;
        elementsPerCore = (elementsPerCore + alignNum - 1) / alignNum * alignNum;

        uint32_t startOffset = blockId * elementsPerCore;
        if (startOffset >= totalLength) {
            coreLength = 0;
        } else {
            coreLength = totalLength - startOffset;
            if (coreLength > elementsPerCore) {
                coreLength = elementsPerCore;
            }
        }

        if (coreLength == 0) return;

        uint32_t alignCoreLength = (coreLength + alignNum - 1) / alignNum * alignNum;
        xGm.SetGlobalBuffer((__gm__ T*)x + startOffset, alignCoreLength);
        yGm.SetGlobalBuffer((__gm__ T*)y + startOffset, alignCoreLength);

        tileLen = 8192;
        tileLen = (tileLen / alignNum) * alignNum;

        tileNum = coreLength / tileLen;
        tailLen = coreLength % tileLen;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLen * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLen * sizeof(T));
        pipe.InitBuffer(calcBuf1, tileLen * sizeof(T));
        pipe.InitBuffer(calcBuf2, tileLen * sizeof(T));
    }

    __aicore__ inline void Process() {
        if (coreLength == 0) return;
        for (uint32_t i = 0; i < tileNum; i++) {
            Compute(i, tileLen);
        }
        if (tailLen > 0) {
            Compute(tileNum, tailLen);
        }
    }

private:
    __aicore__ inline void Compute(uint32_t progress, uint32_t length) {
        uint32_t alignLen = (length + alignNum - 1) / alignNum * alignNum;

        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        LocalTensor<T> minLocal = calcBuf1.Get<T>();
        LocalTensor<T> zeroLocal = calcBuf2.Get<T>();

        DataCopy(xLocal, xGm[progress * tileLen], alignLen);
        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.DeQue<T>();

        Duplicate(zeroLocal, (T)0.0, alignLen);
        Min(minLocal, xLocal, zeroLocal, alignLen);

        Abs(xLocal, xLocal, alignLen);
        Muls(xLocal, xLocal, (T)-1.0, alignLen);
        Exp(xLocal, xLocal, alignLen);
        Adds(xLocal, xLocal, (T)1.0, alignLen);
        Ln(xLocal, xLocal, alignLen);

        Sub(yLocal, minLocal, xLocal, alignLen);

        outQueueY.EnQue<T>(yLocal);
        inQueueX.FreeTensor(xLocal);
        yLocal = outQueueY.DeQue<T>();

        DataCopy(yGm[progress * tileLen], yLocal, alignLen);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> calcBuf1;
    TBuf<QuePosition::VECCALC> calcBuf2;
    GlobalTensor<T> xGm, yGm;
    uint32_t coreLength;
    uint32_t tileLen;
    uint32_t tileNum;
    uint32_t tailLen;
    uint32_t alignNum;
};

// =========================================================================
// BF16 专用实现：先 Cast 成 float，算完再 Cast 回去
// =========================================================================
class KernelLogSigmoidBfloat16 {
public:
    __aicore__ inline KernelLogSigmoidBfloat16() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t alignNum) {
        this->alignNum = alignNum;
        uint32_t blockId = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();

        uint32_t elementsPerCore = (totalLength + blockNum - 1) / blockNum;
        elementsPerCore = (elementsPerCore + alignNum - 1) / alignNum * alignNum;

        uint32_t startOffset = blockId * elementsPerCore;
        if (startOffset >= totalLength) {
            coreLength = 0;
        } else {
            coreLength = totalLength - startOffset;
            if (coreLength > elementsPerCore) {
                coreLength = elementsPerCore;
            }
        }

        if (coreLength == 0) return;

        uint32_t alignCoreLength = (coreLength + alignNum - 1) / alignNum * alignNum;
        xGm.SetGlobalBuffer((__gm__ bfloat16_t*)x + startOffset, alignCoreLength);
        yGm.SetGlobalBuffer((__gm__ bfloat16_t*)y + startOffset, alignCoreLength);

        tileLen = 4096; 
        tileLen = (tileLen / alignNum) * alignNum;

        tileNum = coreLength / tileLen;
        tailLen = coreLength % tileLen;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLen * sizeof(bfloat16_t));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLen * sizeof(bfloat16_t));
        pipe.InitBuffer(calcBufXFloat, tileLen * sizeof(float));
        pipe.InitBuffer(calcBufMinFloat, tileLen * sizeof(float));
        pipe.InitBuffer(calcBufZeroFloat, tileLen * sizeof(float));
    }

    __aicore__ inline void Process() {
        if (coreLength == 0) return;
        for (uint32_t i = 0; i < tileNum; i++) {
            Compute(i, tileLen);
        }
        if (tailLen > 0) {
            Compute(tileNum, tailLen);
        }
    }

private:
    __aicore__ inline void Compute(uint32_t progress, uint32_t length) {
        uint32_t alignLen = (length + alignNum - 1) / alignNum * alignNum;

        LocalTensor<bfloat16_t> xLocal = inQueueX.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> yLocal = outQueueY.AllocTensor<bfloat16_t>();

        LocalTensor<float> xFloat = calcBufXFloat.Get<float>();
        LocalTensor<float> minFloat = calcBufMinFloat.Get<float>();
        LocalTensor<float> zeroFloat = calcBufZeroFloat.Get<float>();

        DataCopy(xLocal, xGm[progress * tileLen], alignLen);
        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.DeQue<bfloat16_t>();

        // 1. bf16 转换为 float (向上转换无精度丢失，使用 CAST_NONE)
        Cast(xFloat, xLocal, RoundMode::CAST_NONE, alignLen);

        // 2. float 精度下运算
        Duplicate(zeroFloat, 0.0f, alignLen);
        Min(minFloat, xFloat, zeroFloat, alignLen);

        Abs(xFloat, xFloat, alignLen);
        Muls(xFloat, xFloat, -1.0f, alignLen);
        Exp(xFloat, xFloat, alignLen);
        Adds(xFloat, xFloat, 1.0f, alignLen);
        Ln(xFloat, xFloat, alignLen);

        Sub(xFloat, minFloat, xFloat, alignLen);

        // 3. float 转换回 bf16 (【核心修复点】向下转换必须使用 CAST_ROUND，否则指令静默失效导致输出垃圾数据)
        Cast(yLocal, xFloat, RoundMode::CAST_ROUND, alignLen);

        outQueueY.EnQue<bfloat16_t>(yLocal);
        inQueueX.FreeTensor(xLocal);
        yLocal = outQueueY.DeQue<bfloat16_t>();

        DataCopy(yGm[progress * tileLen], yLocal, alignLen);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> calcBufXFloat;
    TBuf<QuePosition::VECCALC> calcBufMinFloat;
    TBuf<QuePosition::VECCALC> calcBufZeroFloat;
    GlobalTensor<bfloat16_t> xGm, yGm;
    uint32_t coreLength;
    uint32_t tileLen;
    uint32_t tileNum;
    uint32_t tailLen;
    uint32_t alignNum;
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(1)) {
        KernelLogSigmoid<float> op;
        op.Init(x, y, tilingData.totalLength, tilingData.ALIGN_NUM);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelLogSigmoid<half> op;
        op.Init(x, y, tilingData.totalLength, tilingData.ALIGN_NUM);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelLogSigmoidBfloat16 op;
        op.Init(x, y, tilingData.totalLength, tilingData.ALIGN_NUM);
        op.Process();
    }
}

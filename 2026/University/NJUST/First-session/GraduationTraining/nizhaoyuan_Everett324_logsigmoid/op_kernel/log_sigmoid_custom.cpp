#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_LENGTH = 1024;

__aicore__ inline uint32_t CeilDivCustom(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline uint32_t AlignUpCustom(uint32_t a, uint32_t align)
{
    return (a + align - 1) / align * align;
}

class KernelLogSigmoidFloat {
public:
    __aicore__ inline KernelLogSigmoidFloat() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t totalLength,
        uint32_t coreNum)
    {
        uint32_t coreIdx = GetBlockIdx();

        uint32_t alignNum = 32 / sizeof(float);
        uint32_t blockLength = CeilDivCustom(totalLength, coreNum);
        blockLength = AlignUpCustom(blockLength, alignNum);

        uint32_t start = coreIdx * blockLength;

        if (start >= totalLength) {
            coreLength = 0;
            return;
        }

        uint32_t remain = totalLength - start;
        coreLength = remain < blockLength ? remain : blockLength;

        xGm.SetGlobalBuffer((__gm__ float*)x + start, coreLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + start, coreLength);

        pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, TILE_LENGTH * sizeof(float));

        pipe.InitBuffer(tmpBuf1, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(tmpBuf2, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(tmpBuf3, TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (coreLength == 0) {
            return;
        }

        uint32_t loopCount = coreLength / TILE_LENGTH;
        uint32_t tail = coreLength % TILE_LENGTH;

        for (uint32_t i = 0; i < loopCount; ++i) {
            ComputeTile(i * TILE_LENGTH, TILE_LENGTH);
        }

        if (tail > 0) {
            ComputeTile(loopCount * TILE_LENGTH, tail);
        }
    }

private:
    __aicore__ inline void ComputeTile(uint32_t offset, uint32_t len)
    {
        LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
        DataCopy(xLocal, xGm[offset], len);
        inQueue.EnQue(xLocal);

        xLocal = inQueue.DeQue<float>();

        LocalTensor<float> yLocal = outQueue.AllocTensor<float>();

        LocalTensor<float> tmpAbs = tmpBuf1.Get<float>();
        LocalTensor<float> tmpExp = tmpBuf2.Get<float>();
        LocalTensor<float> tmpMin = tmpBuf3.Get<float>();

        // tmpAbs = abs(x)
        Abs(tmpAbs, xLocal, len);

        // tmpAbs = -abs(x)
        Muls(tmpAbs, tmpAbs, static_cast<float>(-1.0), len);

        // tmpExp = exp(-abs(x))
        Exp(tmpExp, tmpAbs, len);

        // tmpExp = 1 + exp(-abs(x))
        Adds(tmpExp, tmpExp, static_cast<float>(1.0), len);

        // tmpAbs = log(1 + exp(-abs(x)))
        Ln(tmpAbs, tmpExp, len);

        // tmpMin = min(x, 0)
        Mins(tmpMin, xLocal, static_cast<float>(0.0), len);

        // y = min(x, 0) - log(1 + exp(-abs(x)))
        Sub(yLocal, tmpMin, tmpAbs, len);

        inQueue.FreeTensor(xLocal);

        outQueue.EnQue(yLocal);
        yLocal = outQueue.DeQue<float>();

        DataCopy(yGm[offset], yLocal, len);

        outQueue.FreeTensor(yLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    TBuf<QuePosition::VECCALC> tmpBuf1;
    TBuf<QuePosition::VECCALC> tmpBuf2;
    TBuf<QuePosition::VECCALC> tmpBuf3;

    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;

    uint32_t coreLength = 0;
};

template <typename TYPE>
class KernelLogSigmoidLowPrecision {
public:
    __aicore__ inline KernelLogSigmoidLowPrecision() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t totalLength,
        uint32_t coreNum)
    {
        uint32_t coreIdx = GetBlockIdx();

        uint32_t alignNum = 32 / sizeof(TYPE);
        uint32_t blockLength = CeilDivCustom(totalLength, coreNum);
        blockLength = AlignUpCustom(blockLength, alignNum);

        uint32_t start = coreIdx * blockLength;

        if (start >= totalLength) {
            coreLength = 0;
            return;
        }

        uint32_t remain = totalLength - start;
        coreLength = remain < blockLength ? remain : blockLength;

        xGm.SetGlobalBuffer((__gm__ TYPE*)x + start, coreLength);
        yGm.SetGlobalBuffer((__gm__ TYPE*)y + start, coreLength);

        pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_LENGTH * sizeof(TYPE));
        pipe.InitBuffer(outQueue, BUFFER_NUM, TILE_LENGTH * sizeof(TYPE));

        pipe.InitBuffer(tmpBuf1, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(tmpBuf2, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(tmpBuf3, TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (coreLength == 0) {
            return;
        }

        uint32_t loopCount = coreLength / TILE_LENGTH;
        uint32_t tail = coreLength % TILE_LENGTH;

        for (uint32_t i = 0; i < loopCount; ++i) {
            ComputeTile(i * TILE_LENGTH, TILE_LENGTH);
        }

        if (tail > 0) {
            ComputeTile(loopCount * TILE_LENGTH, tail);
        }
    }

private:
    __aicore__ inline void ComputeTile(uint32_t offset, uint32_t len)
    {
        LocalTensor<TYPE> xLocal = inQueue.AllocTensor<TYPE>();
        DataCopy(xLocal, xGm[offset], len);
        inQueue.EnQue(xLocal);

        xLocal = inQueue.DeQue<TYPE>();

        LocalTensor<TYPE> yLocal = outQueue.AllocTensor<TYPE>();

        LocalTensor<float> xFp32 = tmpBuf1.Get<float>();
        LocalTensor<float> tmp1 = tmpBuf2.Get<float>();
        LocalTensor<float> tmp2 = tmpBuf3.Get<float>();

        // half / bfloat16 先转成 float32 计算
        Cast(xFp32, xLocal, RoundMode::CAST_NONE, len);

        // tmp1 = abs(x)
        Abs(tmp1, xFp32, len);

        // tmp1 = -abs(x)
        Muls(tmp1, tmp1, static_cast<float>(-1.0), len);

        // tmp2 = exp(-abs(x))
        Exp(tmp2, tmp1, len);

        // tmp2 = 1 + exp(-abs(x))
        Adds(tmp2, tmp2, static_cast<float>(1.0), len);

        // tmp1 = log(1 + exp(-abs(x)))
        Ln(tmp1, tmp2, len);

        // tmp2 = min(x, 0)
        Mins(tmp2, xFp32, static_cast<float>(0.0), len);

        // xFp32 = min(x, 0) - log(1 + exp(-abs(x)))
        Sub(xFp32, tmp2, tmp1, len);

        // 结果转回 half / bfloat16
        Cast(yLocal, xFp32, RoundMode::CAST_RINT, len);

        inQueue.FreeTensor(xLocal);

        outQueue.EnQue(yLocal);
        yLocal = outQueue.DeQue<TYPE>();

        DataCopy(yGm[offset], yLocal, len);

        outQueue.FreeTensor(yLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    TBuf<QuePosition::VECCALC> tmpBuf1;
    TBuf<QuePosition::VECCALC> tmpBuf2;
    TBuf<QuePosition::VECCALC> tmpBuf3;

    GlobalTensor<TYPE> xGm;
    GlobalTensor<TYPE> yGm;

    uint32_t coreLength = 0;
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    uint32_t totalLength = tilingData.size;
    uint32_t coreNum = tilingData.core_num == 0 ? 1 : tilingData.core_num;

    if (TILING_KEY_IS(1)) {
        KernelLogSigmoidFloat op;
        op.Init(x, y, totalLength, coreNum);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelLogSigmoidLowPrecision<half> op;
        op.Init(x, y, totalLength, coreNum);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelLogSigmoidLowPrecision<bfloat16_t> op;
        op.Init(x, y, totalLength, coreNum);
        op.Process();
    }
}

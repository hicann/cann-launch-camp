#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

using namespace AscendC;

constexpr uint32_t TILE_LENGTH = 2048;

__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline uint32_t AlignUp(uint32_t a, uint32_t b)
{
    return CeilDiv(a, b) * b;
}

template <typename T>
class KernelLogSigmoidCustom {
public:
    __aicore__ inline KernelLogSigmoidCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalSize, uint32_t blockDim)
    {
        uint32_t blockIdx = GetBlockIdx();

        // DataCopy 按 32B 对齐。float 对齐 8 个元素，half/bfloat16 对齐 16 个元素。
        constexpr uint32_t alignNum = 32 / sizeof(T);

        uint32_t perCoreSize = AlignUp(CeilDiv(totalSize, blockDim), alignNum);
        uint32_t offset = blockIdx * perCoreSize;

        this->blockOffset = offset;
        if (offset >= totalSize) {
            this->blockLength = 0;
        } else {
            uint32_t remain = totalSize - offset;
            this->blockLength = remain > perCoreSize ? perCoreSize : remain;
        }

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + this->blockOffset, this->blockLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + this->blockOffset, this->blockLength);

        pipe.InitBuffer(inQueue, 1, TILE_LENGTH * sizeof(T));
        pipe.InitBuffer(outQueue, 1, TILE_LENGTH * sizeof(T));

        pipe.InitBuffer(xFloatBuf, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(tmpBuf, TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength == 0) {
            return;
        }

        uint32_t loopCount = this->blockLength / TILE_LENGTH;
        uint32_t tailCount = this->blockLength % TILE_LENGTH;

        for (uint32_t i = 0; i < loopCount; ++i) {
            Compute(i * TILE_LENGTH, TILE_LENGTH);
        }

        if (tailCount > 0) {
            Compute(loopCount * TILE_LENGTH, tailCount);
        }
    }

private:
    __aicore__ inline void Compute(uint32_t offset, uint32_t count)
    {
        LocalTensor<T> xLocal = inQueue.AllocTensor<T>();
        DataCopy(xLocal, xGm[offset], count);
        inQueue.EnQue(xLocal);

        xLocal = inQueue.DeQue<T>();

        LocalTensor<T> yLocal = outQueue.AllocTensor<T>();
        LocalTensor<float> xFloat = xFloatBuf.Get<float>();
        LocalTensor<float> tmp = tmpBuf.Get<float>();

        // x -> float
        Cast(xFloat, xLocal, RoundMode::CAST_NONE, count);

        // LogSigmoid(x) = -log(1 + exp(-x))
        Muls(tmp, xFloat, static_cast<float>(-1.0), count);
        Exp(tmp, tmp, count);
        Adds(tmp, tmp, static_cast<float>(1.0), count);
        Ln(tmp, tmp, count);
        Muls(tmp, tmp, static_cast<float>(-1.0), count);

        // float -> T
        Cast(yLocal, tmp, RoundMode::CAST_RINT, count);

        outQueue.EnQue<T>(yLocal);

        yLocal = outQueue.DeQue<T>();
        DataCopy(yGm[offset], yLocal, count);

        inQueue.FreeTensor(xLocal);
        outQueue.FreeTensor(yLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;

    TBuf<TPosition::VECCALC> xFloatBuf;
    TBuf<TPosition::VECCALC> tmpBuf;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;

    uint32_t blockOffset = 0;
    uint32_t blockLength = 0;
};

// float 特化：输入本身已经是 float，不需要额外 Cast 到 float。
template <>
class KernelLogSigmoidCustom<float> {
public:
    __aicore__ inline KernelLogSigmoidCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalSize, uint32_t blockDim)
    {
        uint32_t blockIdx = GetBlockIdx();

        constexpr uint32_t alignNum = 32 / sizeof(float);

        uint32_t perCoreSize = AlignUp(CeilDiv(totalSize, blockDim), alignNum);
        uint32_t offset = blockIdx * perCoreSize;

        this->blockOffset = offset;
        if (offset >= totalSize) {
            this->blockLength = 0;
        } else {
            uint32_t remain = totalSize - offset;
            this->blockLength = remain > perCoreSize ? perCoreSize : remain;
        }

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x) + this->blockOffset, this->blockLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y) + this->blockOffset, this->blockLength);

        pipe.InitBuffer(inQueue, 1, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(outQueue, 1, TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength == 0) {
            return;
        }

        uint32_t loopCount = this->blockLength / TILE_LENGTH;
        uint32_t tailCount = this->blockLength % TILE_LENGTH;

        for (uint32_t i = 0; i < loopCount; ++i) {
            Compute(i * TILE_LENGTH, TILE_LENGTH);
        }

        if (tailCount > 0) {
            Compute(loopCount * TILE_LENGTH, tailCount);
        }
    }

private:
    __aicore__ inline void Compute(uint32_t offset, uint32_t count)
    {
        LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
        DataCopy(xLocal, xGm[offset], count);
        inQueue.EnQue(xLocal);

        xLocal = inQueue.DeQue<float>();

        LocalTensor<float> yLocal = outQueue.AllocTensor<float>();

        // LogSigmoid(x) = -log(1 + exp(-x))
        Muls(yLocal, xLocal, static_cast<float>(-1.0), count);
        Exp(yLocal, yLocal, count);
        Adds(yLocal, yLocal, static_cast<float>(1.0), count);
        Ln(yLocal, yLocal, count);
        Muls(yLocal, yLocal, static_cast<float>(-1.0), count);

        outQueue.EnQue<float>(yLocal);

        yLocal = outQueue.DeQue<float>();
        DataCopy(yGm[offset], yLocal, count);

        inQueue.FreeTensor(xLocal);
        outQueue.FreeTensor(yLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;

    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;

    uint32_t blockOffset = 0;
    uint32_t blockLength = 0;
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelLogSigmoidCustom<DTYPE_X> op;
    op.Init(x, y, tilingData.size, tilingData.blockDim);
    op.Process();
}

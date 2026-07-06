#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

using namespace AscendC;

constexpr uint32_t DTYPE_FLOAT32 = 0;
constexpr uint32_t DTYPE_FLOAT16 = 1;
constexpr uint32_t DTYPE_BFLOAT16 = 2;
constexpr uint32_t BUFFER_NUM = 2;

class KernelLogSigmoidFloat {
public:
    __aicore__ inline KernelLogSigmoidFloat() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength,
                                uint32_t blockLength, uint32_t tileLength)
    {
        this->totalLength = totalLength;
        this->blockLength = blockLength;
        this->tileLength = tileLength;

        this->blockStart = GetBlockIdx() * this->blockLength;

        if (this->blockStart >= this->totalLength) {
            this->curBlockLength = 0;
            return;
        }

        this->curBlockLength = this->blockLength;
        if (this->blockStart + this->curBlockLength > this->totalLength) {
            this->curBlockLength = this->totalLength - this->blockStart;
        }

        xGm.SetGlobalBuffer((__gm__ float*)x + this->blockStart, this->curBlockLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + this->blockStart, this->curBlockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->curBlockLength == 0) {
            return;
        }

        for (uint32_t offset = 0; offset < this->curBlockLength; offset += this->tileLength) {
            uint32_t calcLength = this->tileLength;
            if (offset + calcLength > this->curBlockLength) {
                calcLength = this->curBlockLength - offset;
            }

            CopyIn(offset, calcLength);
            Compute(calcLength);
            CopyOut(offset, calcLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t calcLength)
    {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[offset], calcLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t calcLength)
    {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // y = log(1 / (1 + exp(-x))) = -ln(1 + exp(-x))
        Muls(yLocal, xLocal, -1.0f, calcLength);
        Exp(yLocal, yLocal, calcLength);
        Adds(yLocal, yLocal, 1.0f, calcLength);
        Ln(yLocal, yLocal, calcLength);
        Muls(yLocal, yLocal, -1.0f, calcLength);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t calcLength)
    {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[offset], yLocal, calcLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;

    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t blockStart;
    uint32_t curBlockLength;
};

template <typename T, RoundMode CAST_MODE>
class KernelLogSigmoidCast {
public:
    __aicore__ inline KernelLogSigmoidCast() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength,
                                uint32_t blockLength, uint32_t tileLength)
    {
        this->totalLength = totalLength;
        this->blockLength = blockLength;
        this->tileLength = tileLength;

        this->blockStart = GetBlockIdx() * this->blockLength;

        if (this->blockStart >= this->totalLength) {
            this->curBlockLength = 0;
            return;
        }

        this->curBlockLength = this->blockLength;
        if (this->blockStart + this->curBlockLength > this->totalLength) {
            this->curBlockLength = this->totalLength - this->blockStart;
        }

        xGm.SetGlobalBuffer((__gm__ T*)x + this->blockStart, this->curBlockLength);
        yGm.SetGlobalBuffer((__gm__ T*)y + this->blockStart, this->curBlockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->curBlockLength == 0) {
            return;
        }

        for (uint32_t offset = 0; offset < this->curBlockLength; offset += this->tileLength) {
            uint32_t calcLength = this->tileLength;
            if (offset + calcLength > this->curBlockLength) {
                calcLength = this->curBlockLength - offset;
            }

            CopyIn(offset, calcLength);
            Compute(calcLength);
            CopyOut(offset, calcLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t calcLength)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopy(xLocal, xGm[offset], calcLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t calcLength)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        LocalTensor<float> tmpLocal = tmpBuffer.Get<float>();

        // Convert half/bfloat16 input to float32 for Exp/Ln computation.
        Cast(tmpLocal, xLocal, RoundMode::CAST_NONE, calcLength);

        // y = log(1 / (1 + exp(-x))) = -ln(1 + exp(-x))
        Muls(tmpLocal, tmpLocal, -1.0f, calcLength);
        Exp(tmpLocal, tmpLocal, calcLength);
        Adds(tmpLocal, tmpLocal, 1.0f, calcLength);
        Ln(tmpLocal, tmpLocal, calcLength);
        Muls(tmpLocal, tmpLocal, -1.0f, calcLength);

        Cast(yLocal, tmpLocal, CAST_MODE, calcLength);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t calcLength)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopy(yGm[offset], yLocal, calcLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmpBuffer;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;

    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t blockStart;
    uint32_t curBlockLength;
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    if (tilingData.dataType == DTYPE_FLOAT32) {
        KernelLogSigmoidFloat op;
        op.Init(x, y, tilingData.totalLength, tilingData.blockLength, tilingData.tileLength);
        op.Process();
    } else if (tilingData.dataType == DTYPE_FLOAT16) {
        KernelLogSigmoidCast<half, RoundMode::CAST_NONE> op;
        op.Init(x, y, tilingData.totalLength, tilingData.blockLength, tilingData.tileLength);
        op.Process();
    } else if (tilingData.dataType == DTYPE_BFLOAT16) {
        KernelLogSigmoidCast<bfloat16_t, RoundMode::CAST_RINT> op;
        op.Init(x, y, tilingData.totalLength, tilingData.blockLength, tilingData.tileLength);
        op.Process();
    }
}

#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t TILE_LENGTH = 2048;

template <typename T>
class KernelLogSigmoid {
public:
    __aicore__ inline KernelLogSigmoid() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalSize, uint32_t blockNum)
    {
        uint32_t blockIdx = GetBlockIdx();

        uint32_t avgLength = (totalSize + blockNum - 1) / blockNum;
        uint32_t startOffset = blockIdx * avgLength;

        if (startOffset >= totalSize) {
            this->blockLength = 0;
            return;
        }

        uint32_t remainLength = totalSize - startOffset;
        this->blockLength = remainLength < avgLength ? remainLength : avgLength;

        xGm.SetGlobalBuffer((__gm__ T*)x + startOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ T*)y + startOffset, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe.InitBuffer(tmpBuffer, TILE_LENGTH * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength == 0) {
            return;
        }

        for (uint32_t offset = 0; offset < this->blockLength; offset += TILE_LENGTH) {
            uint32_t len = TILE_LENGTH;

            if (offset + TILE_LENGTH > this->blockLength) {
                len = this->blockLength - offset;
            }

            CopyIn(offset, len);
            Compute(len);
            CopyOut(offset, len);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t len)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();

        DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(len * sizeof(T)),
            0,
            0,
            0
        };

        DataCopyPadExtParams<T> padParams{
            true,
            0,
            0,
            static_cast<T>(0)
        };

        DataCopyPad(xLocal, xGm[offset], copyParams, padParams);

        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        LocalTensor<T> tmpLocal = tmpBuffer.Get<T>();

        // tmp = -x
        Muls(tmpLocal, xLocal, static_cast<T>(-1.0), len);

        // tmp = exp(-x)
        Exp(tmpLocal, tmpLocal, len);

        // tmp = 1 + exp(-x)
        Adds(tmpLocal, tmpLocal, static_cast<T>(1.0), len);

        // tmp = log(1 + exp(-x))
        Ln(tmpLocal, tmpLocal, len);

        // y = -log(1 + exp(-x))
        Muls(yLocal, tmpLocal, static_cast<T>(-1.0), len);

        outQueueY.EnQue<T>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t len)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();

        DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(len * sizeof(T)),
            0,
            0,
            0
        };

        DataCopyPad(yGm[offset], yLocal, copyParams);

        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmpBuffer;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;

    uint32_t blockLength;
};

template <>
class KernelLogSigmoid<__bf16> {
public:
    __aicore__ inline KernelLogSigmoid() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalSize, uint32_t blockNum)
    {
        uint32_t blockIdx = GetBlockIdx();

        uint32_t avgLength = (totalSize + blockNum - 1) / blockNum;
        uint32_t startOffset = blockIdx * avgLength;

        if (startOffset >= totalSize) {
            this->blockLength = 0;
            return;
        }

        uint32_t remainLength = totalSize - startOffset;
        this->blockLength = remainLength < avgLength ? remainLength : avgLength;

        xGm.SetGlobalBuffer((__gm__ __bf16*)x + startOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ __bf16*)y + startOffset, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(__bf16));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(__bf16));

        pipe.InitBuffer(xFloatBuffer, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(tmpFloatBuffer, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(yFloatBuffer, TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength == 0) {
            return;
        }

        for (uint32_t offset = 0; offset < this->blockLength; offset += TILE_LENGTH) {
            uint32_t len = TILE_LENGTH;

            if (offset + TILE_LENGTH > this->blockLength) {
                len = this->blockLength - offset;
            }

            CopyIn(offset, len);
            Compute(len);
            CopyOut(offset, len);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t len)
    {
        LocalTensor<__bf16> xLocal = inQueueX.AllocTensor<__bf16>();

        DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(len * sizeof(__bf16)),
            0,
            0,
            0
        };

        DataCopyPadExtParams<__bf16> padParams{
            true,
            0,
            0,
            static_cast<__bf16>(0)
        };

        DataCopyPad(xLocal, xGm[offset], copyParams, padParams);

        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        LocalTensor<__bf16> xLocal = inQueueX.DeQue<__bf16>();
        LocalTensor<__bf16> yLocal = outQueueY.AllocTensor<__bf16>();

        LocalTensor<float> xFloat = xFloatBuffer.Get<float>();
        LocalTensor<float> tmpFloat = tmpFloatBuffer.Get<float>();
        LocalTensor<float> yFloat = yFloatBuffer.Get<float>();

        // BF16 -> float
        Cast(xFloat, xLocal, RoundMode::CAST_NONE, len);

        // tmp = -x
        Muls(tmpFloat, xFloat, -1.0f, len);

        // tmp = exp(-x)
        Exp(tmpFloat, tmpFloat, len);

        // tmp = 1 + exp(-x)
        Adds(tmpFloat, tmpFloat, 1.0f, len);

        // tmp = log(1 + exp(-x))
        Ln(tmpFloat, tmpFloat, len);

        // y = -log(1 + exp(-x))
        Muls(yFloat, tmpFloat, -1.0f, len);

        // float -> BF16
        Cast(yLocal, yFloat, RoundMode::CAST_RINT, len);

        outQueueY.EnQue<__bf16>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t len)
    {
        LocalTensor<__bf16> yLocal = outQueueY.DeQue<__bf16>();

        DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(len * sizeof(__bf16)),
            0,
            0,
            0
        };

        DataCopyPad(yGm[offset], yLocal, copyParams);

        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    TBuf<QuePosition::VECCALC> xFloatBuffer;
    TBuf<QuePosition::VECCALC> tmpFloatBuffer;
    TBuf<QuePosition::VECCALC> yFloatBuffer;

    GlobalTensor<__bf16> xGm;
    GlobalTensor<__bf16> yGm;

    uint32_t blockLength;
};


extern "C" __global__ __aicore__ void log_sigmoid_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelLogSigmoid<DTYPE_X> op;
    op.Init(x, y, tilingData.size, tilingData.blockNum);
    op.Process();
}

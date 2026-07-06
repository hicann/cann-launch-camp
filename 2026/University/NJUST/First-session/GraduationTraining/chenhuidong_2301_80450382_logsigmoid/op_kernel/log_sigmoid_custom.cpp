#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"
#include <type_traits>

constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class KernelLogSigmoid {
public:
    __aicore__ inline KernelLogSigmoid() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileLength)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileLength = tileLength;

        uint32_t offset = this->blockLength * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ T*)x + offset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ T*)y + offset, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(tmpBuf, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t loopCount = (this->blockLength + this->tileLength - 1) / this->tileLength;
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t offset = i * this->tileLength;
            uint32_t curLength = this->tileLength;
            if (offset + curLength > this->blockLength) {
                curLength = this->blockLength - offset;
            }

            CopyIn(i, curLength);
            Compute(curLength);
            CopyOut(i, curLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t length)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], length);
        inQueueX.EnQue<T>(xLocal);
    }

    __aicore__ inline void Compute(uint32_t length)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

        if constexpr (std::is_same_v<T, float>) {
            AscendC::Muls(yLocal, xLocal, -1.0f, length);
            AscendC::Exp(yLocal, yLocal, length);
            AscendC::Adds(yLocal, yLocal, 1.0f, length);
            AscendC::Ln(yLocal, yLocal, length);
            AscendC::Muls(yLocal, yLocal, -1.0f, length);
        } else {
            AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();
            AscendC::Cast(tmp, xLocal, AscendC::RoundMode::CAST_NONE, length);
            AscendC::Muls(tmp, tmp, -1.0f, length);
            AscendC::Exp(tmp, tmp, length);
            AscendC::Adds(tmp, tmp, 1.0f, length);
            AscendC::Ln(tmp, tmp, length);
            AscendC::Muls(tmp, tmp, -1.0f, length);

            if constexpr (std::is_same_v<T, half>) {
                AscendC::Cast(yLocal, tmp, AscendC::RoundMode::CAST_NONE, length);
            } else {
                AscendC::Cast(yLocal, tmp, AscendC::RoundMode::CAST_RINT, length);
            }
        }

        outQueueY.EnQue<T>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t length)
    {
        AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, length);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;

    uint32_t blockLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(1)) {
        KernelLogSigmoid<float> op;
        op.Init(x, y, tilingData.size, tilingData.tileLength);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelLogSigmoid<half> op;
        op.Init(x, y, tilingData.size, tilingData.tileLength);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelLogSigmoid<bfloat16_t> op;
        op.Init(x, y, tilingData.size, tilingData.tileLength);
        op.Process();
    }
}
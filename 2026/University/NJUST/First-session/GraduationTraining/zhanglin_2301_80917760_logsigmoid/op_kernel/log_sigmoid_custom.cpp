#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_LENGTH = 1024;

template <typename T>
__aicore__ inline void CastToFloat(LocalTensor<float> dst, LocalTensor<T> src, uint32_t len)
{
    Cast(dst, src, RoundMode::CAST_NONE, len);
}

template <>
__aicore__ inline void CastToFloat<float>(LocalTensor<float> dst, LocalTensor<float> src, uint32_t len)
{
    Adds(dst, src, 0.0f, len);
}

template <typename T>
__aicore__ inline void CastFromFloat(LocalTensor<T> dst, LocalTensor<float> src, uint32_t len)
{
    Cast(dst, src, RoundMode::CAST_RINT, len);
}

template <>
__aicore__ inline void CastFromFloat<float>(LocalTensor<float> dst, LocalTensor<float> src, uint32_t len)
{
    Adds(dst, src, 0.0f, len);
}

class KernelLogSigmoid {
public:
    __aicore__ inline KernelLogSigmoid() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength)
    {
        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();

        uint32_t perBlock = (totalLength + blockNum - 1) / blockNum;
        uint32_t startOffset = blockIdx * perBlock;

        if (startOffset >= totalLength) {
            this->blockLength = 0;
        } else {
            this->blockLength = totalLength - startOffset;
            if (this->blockLength > perBlock) {
                this->blockLength = perBlock;
            }
        }

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + startOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + startOffset, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(DTYPE_Y));

        // float workspace:
        // xFloat: cast 后的 x，也复用来放最终 float 结果
        // tmp1: -x 或 ln(1 + exp(-x))
        // tmp2: exp(-x)
        // tmp3: 1 + exp(-x)
        pipe.InitBuffer(tmpBuffer1, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(tmpBuffer2, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(tmpBuffer3, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(tmpBuffer4, TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t loopCount = (this->blockLength + TILE_LENGTH - 1) / TILE_LENGTH;

        for (uint32_t i = 0; i < loopCount; i++) {
            this->curLength = TILE_LENGTH;
            if ((i + 1) * TILE_LENGTH > this->blockLength) {
                this->curLength = this->blockLength - i * TILE_LENGTH;
            }

            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = this->curLength * sizeof(DTYPE_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        DataCopyPadExtParams<DTYPE_X> padParams;
        padParams.isPad = false;

        DataCopyPad(
            xLocal,
            xGm[progress * TILE_LENGTH],
            copyParams,
            padParams);

        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t progress)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        LocalTensor<float> xFloat = tmpBuffer1.Get<float>();
        LocalTensor<float> tmp1 = tmpBuffer2.Get<float>();
        LocalTensor<float> tmp2 = tmpBuffer3.Get<float>();
        LocalTensor<float> tmp3 = tmpBuffer4.Get<float>();

        // xLocal 可能是 half / float / bfloat16，先统一转成 float。
        CastToFloat<DTYPE_X>(xFloat, xLocal, this->curLength);
        PipeBarrier<PIPE_V>();

        // LogSigmoid(x) = -ln(1 + exp(-x))
        // 全部中间计算都使用 float，避免 half/bfloat16 和 float 混用。
        Muls(tmp1, xFloat, -1.0f, this->curLength);      // tmp1 = -x
        Exp(tmp2, tmp1, this->curLength);                // tmp2 = exp(-x)
        Adds(tmp3, tmp2, 1.0f, this->curLength);         // tmp3 = 1 + exp(-x)
        Ln(tmp1, tmp3, this->curLength);                 // tmp1 = ln(1 + exp(-x))
        Muls(xFloat, tmp1, -1.0f, this->curLength);      // xFloat = -ln(...)

        PipeBarrier<PIPE_V>();

        // 再把 float 结果 cast 回输出类型。
        CastFromFloat<DTYPE_Y>(yLocal, xFloat, this->curLength);

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = this->curLength * sizeof(DTYPE_Y);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        DataCopyPad(
            yGm[progress * TILE_LENGTH],
            yLocal,
            copyParams);

        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    TBuf<QuePosition::VECCALC> tmpBuffer1;
    TBuf<QuePosition::VECCALC> tmpBuffer2;
    TBuf<QuePosition::VECCALC> tmpBuffer3;
    TBuf<QuePosition::VECCALC> tmpBuffer4;

    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;

    uint32_t blockLength;
    uint32_t curLength;
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelLogSigmoid op;
    op.Init(x, y, tilingData.size);
    op.Process();
}

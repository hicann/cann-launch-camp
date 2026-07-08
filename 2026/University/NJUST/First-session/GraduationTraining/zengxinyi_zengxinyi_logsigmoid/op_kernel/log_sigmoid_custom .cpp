#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KernelLogSigmoid {
public:
    __aicore__ inline KernelLogSigmoid() {}

__aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const LogSigmoidCustomTilingData& tilingData)
{
    size = tilingData.size;
    tileLength = tilingData.tileLength;

    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t blockNum = tilingData.blockDim;

    // 普通 DataCopy 要求每个核处理的数据长度尽量 32B 对齐。
    // float32: 8 elements, float16/bfloat16: 16 elements.
    uint32_t alignNum = 32 / sizeof(T);

    uint32_t avgBlockLength = (size + blockNum - 1) / blockNum;
    avgBlockLength = ((avgBlockLength + alignNum - 1) / alignNum) * alignNum;

    uint32_t offset = blockIdx * avgBlockLength;

    if (offset >= size) {
        blockLength = 0;
    } else {
        blockLength = (offset + avgBlockLength > size) ? (size - offset) : avgBlockLength;
    }

    xGm.SetGlobalBuffer((__gm__ T*)x + offset, blockLength);
    yGm.SetGlobalBuffer((__gm__ T*)y + offset, blockLength);

    pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(T));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(T));
    pipe.InitBuffer(tmpBuf1, tileLength * sizeof(float));
    pipe.InitBuffer(tmpBuf2, tileLength * sizeof(float));
}

    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < blockLength; offset += tileLength) {
            uint32_t curLength = (offset + tileLength > blockLength) ? (blockLength - offset) : tileLength;
            CopyIn(offset, curLength);
            Compute(curLength);
            CopyOut(offset, curLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopy(xLocal, xGm[offset], length);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t length)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

        ComputeImpl(yLocal, xLocal, length);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length)
    {
        AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        AscendC::DataCopy(yGm[offset], yLocal, length);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void ComputeImpl(
        AscendC::LocalTensor<float>& yLocal,
        AscendC::LocalTensor<float>& xLocal,
        uint32_t length)
    {
        AscendC::LocalTensor<float> tmp = tmpBuf1.Get<float>();

        AscendC::Muls(tmp, xLocal, -1.0f, length);
        AscendC::Exp(tmp, tmp, length);
        AscendC::Adds(tmp, tmp, 1.0f, length);
        AscendC::Ln(tmp, tmp, length);
        AscendC::Muls(yLocal, tmp, -1.0f, length);
    }

    __aicore__ inline void ComputeImpl(
        AscendC::LocalTensor<half>& yLocal,
        AscendC::LocalTensor<half>& xLocal,
        uint32_t length)
    {
        AscendC::LocalTensor<float> xFloat = tmpBuf1.Get<float>();
        AscendC::LocalTensor<float> tmp = tmpBuf2.Get<float>();

        AscendC::Cast(xFloat, xLocal, AscendC::RoundMode::CAST_NONE, length);
        AscendC::Muls(tmp, xFloat, -1.0f, length);
        AscendC::Exp(tmp, tmp, length);
        AscendC::Adds(tmp, tmp, 1.0f, length);
        AscendC::Ln(tmp, tmp, length);
        AscendC::Muls(tmp, tmp, -1.0f, length);
        AscendC::Cast(yLocal, tmp, AscendC::RoundMode::CAST_NONE, length);
    }

__aicore__ inline void ComputeImpl(
    AscendC::LocalTensor<bfloat16_t>& yLocal,
    AscendC::LocalTensor<bfloat16_t>& xLocal,
    uint32_t length)
{
    AscendC::LocalTensor<float> xFloat = tmpBuf1.Get<float>();
    AscendC::LocalTensor<float> tmp = tmpBuf2.Get<float>();

    AscendC::Cast(xFloat, xLocal, AscendC::RoundMode::CAST_NONE, length);
    AscendC::Muls(tmp, xFloat, -1.0f, length);
    AscendC::Exp(tmp, tmp, length);
    AscendC::Adds(tmp, tmp, 1.0f, length);
    AscendC::Ln(tmp, tmp, length);
    AscendC::Muls(tmp, tmp, -1.0f, length);
    AscendC::Cast(yLocal, tmp, AscendC::RoundMode::CAST_RINT, length);
}

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf2;

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;

    uint32_t size;
    uint32_t blockLength;
    uint32_t tileLength;
};
extern "C" __global__ __aicore__ void log_sigmoid_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    if (tilingData.dataType == 0) {
        KernelLogSigmoid<float> op;
        op.Init(x, y, tilingData);
        op.Process();
    } else if (tilingData.dataType == 1) {
        KernelLogSigmoid<half> op;
        op.Init(x, y, tilingData);
        op.Process();
    } else {
        KernelLogSigmoid<bfloat16_t> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
}

#include "kernel_operator.h"
#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;

constexpr float HALF = 0.5f;
constexpr float ONE = 1.0f;
constexpr float TANH_RATIONAL_C0 = 0.7974154410732703f;
constexpr float TANH_RATIONAL_C2 = 0.0456776776271884f;
constexpr float TANH_RATIONAL_D2 = 0.0107061942549241f;
constexpr uint32_t COPY_ALIGN_BYTES = 32;

static __aicore__ inline bool IsAligned(int32_t count, int32_t typeSize) {
    return (count * typeSize) % COPY_ALIGN_BYTES == 0;
}

template <class T>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t totalLength,
                                uint32_t blockLength, uint32_t tileLength, TPipe* p)
    {
        this->pipe = p;
        int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        int64_t offset = static_cast<int64_t>(blockLength) * blockIdx;
        int64_t remaining = static_cast<int64_t>(totalLength) - offset;
        this->blockLen = remaining > static_cast<int64_t>(blockLength) ? blockLength : (remaining > 0 ? static_cast<uint32_t>(remaining) : 0);
        this->tileLen = tileLength;

        inputGm.SetGlobalBuffer((__gm__ T*)input + offset, this->blockLen);
        outputGm.SetGlobalBuffer((__gm__ T*)output + offset, this->blockLen);

        int32_t loopCount = (this->blockLen + tileLength - 1) / tileLength;
        if (loopCount <= 1) {
            pipe->InitBuffer(inputSingleBuf, tileLength * sizeof(T));
            pipe->InitBuffer(outputSingleBuf, tileLength * sizeof(T));
            pipe->InitBuffer(tmpSingleBuf, tileLength * sizeof(T));
        } else {
            uint8_t bufNum = (loopCount >= 2) ? 2 : 1;
            pipe->InitBuffer(inQueue,  bufNum, tileLength * sizeof(T));
            pipe->InitBuffer(outQueue, bufNum, tileLength * sizeof(T));
            pipe->InitBuffer(tmpBuf, tileLength * sizeof(T));
        }
        multiTile = (loopCount > 1);
    }

    __aicore__ inline void Process()
    {
        if (blockLen == 0) return;
        if (!multiTile) {
            ProcessSingleTile();
            return;
        }
        int32_t loopCount = (blockLen + tileLen - 1) / tileLen;
        for (int32_t i = 0; i < loopCount; i++) {
            int32_t offset = i * tileLen;
            int32_t count = (offset + tileLen > blockLen) ? (blockLen - offset) : tileLen;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void ProcessSingleTile()
    {
        int32_t count = blockLen;
        LocalTensor<T> inputLocal = inputSingleBuf.Get<T>();
        LocalTensor<T> outputLocal = outputSingleBuf.Get<T>();
        LocalTensor<T> tmpLocal = tmpSingleBuf.Get<T>();
        DataCopyPad(inputLocal, inputGm[0], {1, static_cast<uint16_t>(count * sizeof(T)), 0, 0}, {false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();

        GeluCompute(outputLocal, inputLocal, tmpLocal, count);
        PipeBarrier<PIPE_ALL>();

        if (IsAligned(count, sizeof(T))) {
            DataCopy(outputGm[0], outputLocal, count);
        } else {
            DataCopyPad(outputGm[0], outputLocal, {1, static_cast<uint16_t>(count * sizeof(T)), 0, 0});
        }
    }

    __aicore__ inline void CopyIn(int32_t offset, int32_t count)
    {
        LocalTensor<T> xLocal = inQueue.AllocTensor<T>();
        DataCopyPad(xLocal, inputGm[offset], {1, static_cast<uint16_t>(count * sizeof(T)), 0, 0}, {false, 0, 0, 0});
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t count)
    {
        LocalTensor<T> xLocal = inQueue.DeQue<T>();
        LocalTensor<T> yLocal = outQueue.AllocTensor<T>();
        GeluCompute(yLocal, xLocal, tmpBuf.Get<T>(), count);
        outQueue.EnQue<T>(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t offset, int32_t count)
    {
        LocalTensor<T> yLocal = outQueue.DeQue<T>();
        DataCopyPad(outputGm[offset], yLocal, {1, static_cast<uint16_t>(count * sizeof(T)), 0, 0});
        outQueue.FreeTensor(yLocal);
    }

    __aicore__ inline void GeluCompute(LocalTensor<T> y, LocalTensor<T> x,
                                        LocalTensor<T> tmp, int32_t count)
    {
        if constexpr (sizeof(T) == 2) {
            Mul(tmp, x, x, count);
            Adds(tmp, tmp, static_cast<T>(22.363860002236f), count);
            Mul(tmp, tmp, x, count);
            Muls(tmp, tmp, static_cast<T>(-0.0713557640966f), count);
            Exp(y, tmp, count);
            Adds(y, y, static_cast<T>(1.0f), count);
            Div(y, x, y, count);
        } else {
            Mul(tmp, x, x, count);
            Muls(y, tmp, static_cast<T>(TANH_RATIONAL_C2), count);
            Adds(y, y, static_cast<T>(TANH_RATIONAL_C0), count);
            Mul(y, y, x, count);
            Muls(tmp, tmp, static_cast<T>(TANH_RATIONAL_D2), count);
            Adds(tmp, tmp, static_cast<T>(ONE), count);
            Div(y, y, tmp, count);
            // GELU = x * sigmoid(2*z),  sigmoid(u) = 1/(1+exp(-u))
            Muls(y, y, static_cast<T>(-2.0f), count);
            Exp(y, y, count);
            Adds(y, y, static_cast<T>(1.0f), count);
            Div(y, x, y, count);
        }
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN,  1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<QuePosition::VECCALC> tmpBuf;
    TBuf<QuePosition::VECCALC> inputSingleBuf;
    TBuf<QuePosition::VECCALC> outputSingleBuf;
    TBuf<QuePosition::VECCALC> tmpSingleBuf;
    GlobalTensor<T> inputGm, outputGm;
    int32_t blockLen = 0;
    int32_t tileLen = 0;
    bool multiTile = false;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tilingData, tiling);
    TPipe pipe;
    KernelGelu<DT_INPUT_X> op;
    op.Init(input, output, tilingData.totalLength, tilingData.blockLength, tilingData.tileLength, &pipe);
    op.Process();
}

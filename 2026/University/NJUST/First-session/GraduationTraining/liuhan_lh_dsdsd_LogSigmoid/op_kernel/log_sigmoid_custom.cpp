%%writefile Sources/test/custom_op/op_kernel/log_sigmoid_custom.cpp
#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t BLOCK_NUM = 8;
constexpr uint32_t TILE_LENGTH = 1024;
constexpr uint32_t BUFFER_NUM = 1;

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

template <typename T>
class KernelLogSigmoid {
public:
    __aicore__ inline KernelLogSigmoid() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalSize)
    {
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t perCore = (totalSize + BLOCK_NUM - 1) / BLOCK_NUM;
        const uint32_t start = blockIdx * perCore;
        if (start >= totalSize) {
            this->blockLength = 0;
            return;
        }
        this->blockLength = MinU32(perCore, totalSize - start);

        xGm.SetGlobalBuffer((__gm__ T*)x + start, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ T*)y + start, this->blockLength);

        pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe.InitBuffer(outQueue, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe.InitBuffer(tmp1Buf, TILE_LENGTH * sizeof(T));
        pipe.InitBuffer(tmp2Buf, TILE_LENGTH * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < blockLength; offset += TILE_LENGTH) {
            const uint32_t count = MinU32(TILE_LENGTH, blockLength - offset);
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count)
    {
        LocalTensor<T> xLocal = inQueue.AllocTensor<T>();
        DataCopyParams copyParams{1, static_cast<uint16_t>(count * sizeof(T)), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<T> xLocal = inQueue.DeQue<T>();
        LocalTensor<T> yLocal = outQueue.AllocTensor<T>();
        LocalTensor<T> tmp1 = tmp1Buf.Get<T>();
        LocalTensor<T> tmp2 = tmp2Buf.Get<T>();

        // y = -log(1 + exp(-x))
        Muls(tmp1, xLocal, static_cast<T>(-1.0f), count);
        PipeBarrier<PIPE_V>();
        Exp(tmp2, tmp1, count);
        PipeBarrier<PIPE_V>();
        Adds(tmp1, tmp2, static_cast<T>(1.0f), count);
        PipeBarrier<PIPE_V>();
        Log(tmp2, tmp1, count);
        PipeBarrier<PIPE_V>();
        Muls(yLocal, tmp2, static_cast<T>(-1.0f), count);
        PipeBarrier<PIPE_V>();

        outQueue.EnQue<T>(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        LocalTensor<T> yLocal = outQueue.DeQue<T>();
        DataCopyParams copyParams{1, static_cast<uint16_t>(count * sizeof(T)), 0, 0};
        DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueue.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueue;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<TPosition::VECCALC> tmp1Buf;
    TBuf<TPosition::VECCALC> tmp2Buf;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    uint32_t blockLength = 0;
};

class KernelLogSigmoidBf16 {
public:
    __aicore__ inline KernelLogSigmoidBf16() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalSize)
    {
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t perCore = (totalSize + BLOCK_NUM - 1) / BLOCK_NUM;
        const uint32_t start = blockIdx * perCore;
        if (start >= totalSize) {
            this->blockLength = 0;
            return;
        }
        this->blockLength = MinU32(perCore, totalSize - start);

        xGm.SetGlobalBuffer((__gm__ bfloat16_t*)x + start, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ bfloat16_t*)y + start, this->blockLength);

        pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_LENGTH * sizeof(bfloat16_t));
        pipe.InitBuffer(outQueue, BUFFER_NUM, TILE_LENGTH * sizeof(bfloat16_t));
        pipe.InitBuffer(xFloatBuf, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(tmp1Buf, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(tmp2Buf, TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < blockLength; offset += TILE_LENGTH) {
            const uint32_t count = MinU32(TILE_LENGTH, blockLength - offset);
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count)
    {
        LocalTensor<bfloat16_t> xLocal = inQueue.AllocTensor<bfloat16_t>();
        DataCopyParams copyParams{1, static_cast<uint16_t>(count * sizeof(bfloat16_t)), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<bfloat16_t> xLocal = inQueue.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> yLocal = outQueue.AllocTensor<bfloat16_t>();
        LocalTensor<float> xFloat = xFloatBuf.Get<float>();
        LocalTensor<float> tmp1 = tmp1Buf.Get<float>();
        LocalTensor<float> tmp2 = tmp2Buf.Get<float>();

        Cast(xFloat, xLocal, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Muls(tmp1, xFloat, -1.0f, count);
        PipeBarrier<PIPE_V>();
        Exp(tmp2, tmp1, count);
        PipeBarrier<PIPE_V>();
        Adds(tmp1, tmp2, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Log(tmp2, tmp1, count);
        PipeBarrier<PIPE_V>();
        Muls(tmp1, tmp2, -1.0f, count);
        PipeBarrier<PIPE_V>();
        Cast(yLocal, tmp1, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();

        outQueue.EnQue<bfloat16_t>(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        LocalTensor<bfloat16_t> yLocal = outQueue.DeQue<bfloat16_t>();
        DataCopyParams copyParams{1, static_cast<uint16_t>(count * sizeof(bfloat16_t)), 0, 0};
        DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueue.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueue;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<TPosition::VECCALC> xFloatBuf;
    TBuf<TPosition::VECCALC> tmp1Buf;
    TBuf<TPosition::VECCALC> tmp2Buf;
    GlobalTensor<bfloat16_t> xGm;
    GlobalTensor<bfloat16_t> yGm;
    uint32_t blockLength = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(1)) {
        KernelLogSigmoid<float> op;
        op.Init(x, y, tilingData.size);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelLogSigmoid<half> op;
        op.Init(x, y, tilingData.size);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelLogSigmoidBf16 op;
        op.Init(x, y, tilingData.size);
        op.Process();
    }
}
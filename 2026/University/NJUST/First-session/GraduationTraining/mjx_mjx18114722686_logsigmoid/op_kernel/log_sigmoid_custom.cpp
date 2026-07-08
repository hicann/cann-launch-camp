#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t TILE_LENGTH = 1024;

template <typename T>
class KernelLogSigmoid {
public:
    __aicore__ inline KernelLogSigmoid() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength)
    {
        totalLength_ = totalLength;

        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();

        uint32_t perBlockLength = (totalLength_ + blockNum - 1) / blockNum;
        start_ = blockIdx * perBlockLength;

        if (start_ >= totalLength_) {
            length_ = 0;
        } else {
            uint32_t end = start_ + perBlockLength;
            if (end > totalLength_) {
                end = totalLength_;
            }
            length_ = end - start_;
        }

        xGm_.SetGlobalBuffer((__gm__ T*)x + start_, length_);
        yGm_.SetGlobalBuffer((__gm__ T*)y + start_, length_);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe_.InitBuffer(tmpBuf1_, TILE_LENGTH * sizeof(float));
        pipe_.InitBuffer(tmpBuf2_, TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (length_ == 0) {
            return;
        }

        for (uint32_t offset = 0; offset < length_; offset += TILE_LENGTH) {
            uint32_t calcLen = TILE_LENGTH;
            if (offset + TILE_LENGTH > length_) {
                calcLen = length_ - offset;
            }

            CopyIn(offset, calcLen);
            Compute(calcLen);
            CopyOut(offset, calcLen);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t calcLen)
    {
        LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = calcLen * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;

        DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t calcLen)
    {
        LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY_.AllocTensor<T>();

        LocalTensor<float> tmp1 = tmpBuf1_.Get<float>();
        LocalTensor<float> tmp2 = tmpBuf2_.Get<float>();

        // LogSigmoid(x) = -log(1 + exp(-x))
        // 等价于：先计算 Sigmoid，再取 log
        // 但这里直接计算 LogSigmoid

        if constexpr (sizeof(T) == sizeof(float)) {
            // float32: tmp1 = -x
            Muls(tmp1, xLocal, -1.0f, calcLen);
            PipeBarrier<PIPE_V>();

            // tmp2 = exp(-x)
            Exp(tmp2, tmp1, calcLen);
            PipeBarrier<PIPE_V>();

            // tmp2 = 1 + exp(-x)
            Adds(tmp2, tmp2, 1.0f, calcLen);
            PipeBarrier<PIPE_V>();

            // tmp1 = log(1 + exp(-x))
            Log(tmp1, tmp2, calcLen);
            PipeBarrier<PIPE_V>();

            // y = -log(1 + exp(-x))
            Muls(yLocal, tmp1, -1.0f, calcLen);
        } else {
            // float16 / bfloat16: cast到float做计算
            Cast(tmp1, xLocal, RoundMode::CAST_NONE, calcLen);
            PipeBarrier<PIPE_V>();

            // tmp1 = -x
            Muls(tmp1, tmp1, -1.0f, calcLen);
            PipeBarrier<PIPE_V>();

            // tmp2 = exp(-x)
            Exp(tmp2, tmp1, calcLen);
            PipeBarrier<PIPE_V>();

            // tmp2 = 1 + exp(-x)
            Adds(tmp2, tmp2, 1.0f, calcLen);
            PipeBarrier<PIPE_V>();

            // tmp1 = log(1 + exp(-x))
            Log(tmp1, tmp2, calcLen);
            PipeBarrier<PIPE_V>();

            // tmp1 = -log(1 + exp(-x))
            Muls(tmp1, tmp1, -1.0f, calcLen);
            PipeBarrier<PIPE_V>();

            // 转换回目标类型
            Cast(yLocal, tmp1, RoundMode::CAST_RINT, calcLen);
        }

        outQueueY_.EnQue(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t calcLen)
    {
        LocalTensor<T> yLocal = outQueueY_.DeQue<T>();

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = calcLen * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        DataCopyPad(yGm_[offset], yLocal, copyParams);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    TBuf<QuePosition::VECCALC> tmpBuf1_;
    TBuf<QuePosition::VECCALC> tmpBuf2_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    uint32_t totalLength_;
    uint32_t start_;
    uint32_t length_;
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    // dataType: 0=float16, 1=float32, 2=bfloat16
    if (tilingData.dataTypeSize == 2 && tilingData.dataType == 0) {
        // float16
        KernelLogSigmoid<half> op;
        op.Init(x, y, tilingData.totalElements);
        op.Process();
    } else if (tilingData.dataTypeSize == 4 && tilingData.dataType == 1) {
        // float32
        KernelLogSigmoid<float> op;
        op.Init(x, y, tilingData.totalElements);
        op.Process();
    } else if (tilingData.dataTypeSize == 2 && tilingData.dataType == 2) {
        // bfloat16
        KernelLogSigmoid<bfloat16_t> op;
        op.Init(x, y, tilingData.totalElements);
        op.Process();
    }
}

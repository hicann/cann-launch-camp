#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

using namespace AscendC;

// 数据类型枚举
constexpr uint32_t DTYPE_FLOAT32 = 0;
constexpr uint32_t DTYPE_FLOAT16 = 1;
constexpr uint32_t DTYPE_BFLOAT16 = 2;

// 缓冲区数量
constexpr uint32_t BUFFER_NUM = 2;

/**
 * @brief float类型LogSigmoid计算类
 * 使用向量化计算，适用于float32数据类型
 */
class KernelLogSigmoidFloat {
public:
    __aicore__ inline KernelLogSigmoidFloat() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength,
                                uint32_t blockLength, uint32_t tileLength) {
        this->totalLength = totalLength;
        this->blockLength = blockLength;
        this->tileLength = tileLength;

        // 计算当前Block的起始位置
        this->blockStart = GetBlockIdx() * this->blockLength;

        if (this->blockStart >= this->totalLength) {
            this->curBlockLength = 0;
            return;
        }

        // 计算当前Block的实际长度
        this->curBlockLength = this->blockLength;
        if (this->blockStart + this->curBlockLength > this->totalLength) {
            this->curBlockLength = this->totalLength - this->blockStart;
        }

        // 设置GlobalTensor
        xGm.SetGlobalBuffer((__gm__ float*)x + this->blockStart, this->curBlockLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + this->blockStart, this->curBlockLength);

        // 初始化Pipe和队列
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process() {
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
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t calcLength) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[offset], calcLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t calcLength) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // 计算 log_sigmoid: -log(1 + exp(-x))
        // 步骤: x -> -x -> exp(-x) -> 1+exp(-x) -> log(1+exp(-x)) -> -log(1+exp(-x))
        Muls(yLocal, xLocal, -1.0f, calcLength);           // y = -x
        Exp(yLocal, yLocal, calcLength);                   // y = exp(-x)
        Adds(yLocal, yLocal, 1.0f, calcLength);            // y = 1 + exp(-x)
        Ln(yLocal, yLocal, calcLength);                    // y = log(1 + exp(-x))
        Muls(yLocal, yLocal, -1.0f, calcLength);           // y = -log(1 + exp(-x))

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t calcLength) {
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

/**
 * @brief half/bfloat16类型LogSigmoid计算类
 * 先转换为float进行计算，再转换回原类型
 * @tparam T 数据类型 (half 或 bfloat16_t)
 * @tparam CAST_MODE 转换模式
 */
template <typename T, RoundMode CAST_MODE>
class KernelLogSigmoidCast {
public:
    __aicore__ inline KernelLogSigmoidCast() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength,
                                uint32_t blockLength, uint32_t tileLength) {
        this->totalLength = totalLength;
        this->blockLength = blockLength;
        this->tileLength = tileLength;

        // 计算当前Block的起始位置
        this->blockStart = GetBlockIdx() * this->blockLength;

        if (this->blockStart >= this->totalLength) {
            this->curBlockLength = 0;
            return;
        }

        // 计算当前Block的实际长度
        this->curBlockLength = this->blockLength;
        if (this->blockStart + this->curBlockLength > this->totalLength) {
            this->curBlockLength = this->totalLength - this->blockStart;
        }

        // 设置GlobalTensor
        xGm.SetGlobalBuffer((__gm__ T*)x + this->blockStart, this->curBlockLength);
        yGm.SetGlobalBuffer((__gm__ T*)y + this->blockStart, this->curBlockLength);

        // 初始化Pipe、队列和临时缓冲区
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process() {
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
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t calcLength) {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopy(xLocal, xGm[offset], calcLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t calcLength) {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        LocalTensor<float> tmpLocal = tmpBuffer.Get<float>();

        // 将half/bfloat16转换为float32进行计算
        Cast(tmpLocal, xLocal, RoundMode::CAST_NONE, calcLength);

        // 计算 log_sigmoid: -log(1 + exp(-x))
        Muls(tmpLocal, tmpLocal, -1.0f, calcLength);        // tmp = -x
        Exp(tmpLocal, tmpLocal, calcLength);                // tmp = exp(-x)
        Adds(tmpLocal, tmpLocal, 1.0f, calcLength);         // tmp = 1 + exp(-x)
        Ln(tmpLocal, tmpLocal, calcLength);                 // tmp = log(1 + exp(-x))
        Muls(tmpLocal, tmpLocal, -1.0f, calcLength);        // tmp = -log(1 + exp(-x))

        // 将float32转换回原类型
        Cast(yLocal, tmpLocal, CAST_MODE, calcLength);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t calcLength) {
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

/**
 * @brief 算子入口函数
 */
extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    // 根据数据类型选择对应的计算类
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

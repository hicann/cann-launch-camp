%%writefile  Sources/test/custom_op/op_kernel/log_sigmoid_custom.cpp
#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

// 双缓冲
constexpr int32_t BUFFER_NUM = 2;

// half / float 路径
template <class T>
class KernelLogSigmoid {
public:
    __aicore__ inline KernelLogSigmoid() {}
    __aicore__ inline void Init(GM_ADDR xBase, GM_ADDR yBase, uint32_t blockLength, uint32_t tileNum)
    {
        this->blockLength = blockLength;
        ASSERT(this->blockLength != 0);

        if (tileNum == 1) {
            this->tileLength = this->blockLength;
        } else {
            this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        }

        // tileLength 对齐到 32B（half: 16 元素, float: 8 元素）
        constexpr int32_t ALIGN = 32 / sizeof(T);
        this->tileLength = (this->tileLength / ALIGN) * ALIGN;
        if (this->tileLength == 0) this->tileLength = ALIGN;
        ASSERT(this->tileLength != 0);

        // blockLength 已在外部保证 32B 对齐，故 tailLength 自然 32B 对齐
        this->fullTiles = this->blockLength / this->tileLength;
        this->tailLength = this->blockLength % this->tileLength;

        xGm.SetGlobalBuffer((__gm__ T *)xBase, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ T *)yBase, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(calcQueueTmp1, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(calcQueueTmp2, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        constexpr int32_t ALIGN = 32 / sizeof(T);
        int32_t loopCount = this->fullTiles + (this->tailLength > 0 ? 1 : 0);
        for (int32_t i = 0; i < loopCount; i++) {
            int32_t offset = i * this->tileLength;
            bool isTail = (i == this->fullTiles && this->tailLength > 0);
            // dataLen 已保证 32B 对齐
            int32_t dataLen = isTail ? this->tailLength : this->tileLength;
            // compLen 补齐到 32B 整数倍用于向量计算
            int32_t compLen = isTail ? ((dataLen + ALIGN - 1) / ALIGN) * ALIGN : dataLen;
            CopyIn(offset, dataLen);
            Compute(compLen);
            CopyOut(offset, dataLen);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t offset, int32_t length)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopy(xLocal, xGm[offset], length);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t length)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> tmp1 = calcQueueTmp1.AllocTensor<T>();
        AscendC::LocalTensor<T> tmp2 = calcQueueTmp2.AllocTensor<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

        T negOne = -1.0;
        T one = 1.0;
        // y = -ln(1 + e^(-x))
        AscendC::Muls(tmp1, xLocal, negOne, length);   // -x
        AscendC::Exp(tmp2, tmp1, length);               // e^(-x)
        AscendC::Adds(tmp1, tmp2, one, length);         // 1 + e^(-x)
        AscendC::Log(yLocal, tmp1, length);             // ln(1+e^(-x))
        AscendC::Muls(yLocal, yLocal, negOne, length);  // -ln(1+e^(-x))

        outQueueY.EnQue<T>(yLocal);
        calcQueueTmp2.FreeTensor(tmp2);
        calcQueueTmp1.FreeTensor(tmp1);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t offset, int32_t length)
    {
        AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        AscendC::DataCopy(yGm[offset], yLocal, length);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> calcQueueTmp1;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> calcQueueTmp2;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t fullTiles;
    uint32_t tailLength;
};

// bfloat16 路径
class KernelLogSigmoidBf16 {
public:
    __aicore__ inline KernelLogSigmoidBf16() {}
    __aicore__ inline void Init(GM_ADDR xBase, GM_ADDR yBase, uint32_t blockLength, uint32_t tileNum)
    {
        this->blockLength = blockLength;
        ASSERT(this->blockLength != 0);

        if (tileNum == 1) {
            this->tileLength = this->blockLength;
        } else {
            this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        }

        // tileLength 对齐到 32B（bf16: 16 元素）
        constexpr int32_t ALIGN_BF16 = 32 / sizeof(bfloat16_t);
        this->tileLength = (this->tileLength / ALIGN_BF16) * ALIGN_BF16;
        if (this->tileLength == 0) this->tileLength = ALIGN_BF16;
        ASSERT(this->tileLength != 0);

        // blockLength 已在外部保证 32B 对齐，故 tailLength 自然对齐
        this->fullTiles = this->blockLength / this->tileLength;
        this->tailLength = this->blockLength % this->tileLength;

        xGmBf16.SetGlobalBuffer((__gm__ bfloat16_t *)xBase, this->blockLength);
        yGmBf16.SetGlobalBuffer((__gm__ bfloat16_t *)yBase, this->blockLength);

        pipe.InitBuffer(inQueueBf16, BUFFER_NUM, this->tileLength * sizeof(bfloat16_t));
        pipe.InitBuffer(outQueueBf16, BUFFER_NUM, this->tileLength * sizeof(bfloat16_t));
        pipe.InitBuffer(inQueueFloat, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(calcQueueTmp1, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(calcQueueTmp2, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueFloat, BUFFER_NUM, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        constexpr int32_t ALIGN_BF16 = 32 / sizeof(bfloat16_t);
        int32_t loopCount = this->fullTiles + (this->tailLength > 0 ? 1 : 0);
        for (int32_t i = 0; i < loopCount; i++) {
            int32_t offset = i * this->tileLength;
            bool isTail = (i == this->fullTiles && this->tailLength > 0);
            int32_t dataLen = isTail ? this->tailLength : this->tileLength;
            int32_t compLen = isTail ? ((dataLen + ALIGN_BF16 - 1) / ALIGN_BF16) * ALIGN_BF16 : dataLen;
            CopyIn(offset, dataLen);
            CastToFloat(compLen);
            Compute(compLen);
            CastToBf16(compLen);
            CopyOut(offset, dataLen);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t offset, int32_t length)
    {
        AscendC::LocalTensor<bfloat16_t> xLocal = inQueueBf16.AllocTensor<bfloat16_t>();
        AscendC::DataCopy(xLocal, xGmBf16[offset], length);
        inQueueBf16.EnQue(xLocal);
    }

    __aicore__ inline void CastToFloat(int32_t length)
    {
        AscendC::LocalTensor<bfloat16_t> xBf16 = inQueueBf16.DeQue<bfloat16_t>();
        AscendC::LocalTensor<float> xFloat = inQueueFloat.AllocTensor<float>();
        AscendC::Cast(xFloat, xBf16, AscendC::RoundMode::CAST_NONE, length);
        inQueueFloat.EnQue(xFloat);
        inQueueBf16.FreeTensor(xBf16);
    }

    __aicore__ inline void Compute(int32_t length)
    {
        AscendC::LocalTensor<float> xLocal = inQueueFloat.DeQue<float>();
        AscendC::LocalTensor<float> tmp1 = calcQueueTmp1.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp2 = calcQueueTmp2.AllocTensor<float>();
        AscendC::LocalTensor<float> yLocal = outQueueFloat.AllocTensor<float>();

        float negOne = -1.0f;
        float one = 1.0f;
        // y = -ln(1 + e^(-x))
        AscendC::Muls(tmp1, xLocal, negOne, length);
        AscendC::Exp(tmp2, tmp1, length);
        AscendC::Adds(tmp1, tmp2, one, length);
        AscendC::Log(yLocal, tmp1, length);
        AscendC::Muls(yLocal, yLocal, negOne, length);

        outQueueFloat.EnQue<float>(yLocal);
        calcQueueTmp2.FreeTensor(tmp2);
        calcQueueTmp1.FreeTensor(tmp1);
        inQueueFloat.FreeTensor(xLocal);
    }

    __aicore__ inline void CastToBf16(int32_t length)
    {
        AscendC::LocalTensor<float> yFloat = outQueueFloat.DeQue<float>();
        AscendC::LocalTensor<bfloat16_t> yBf16 = outQueueBf16.AllocTensor<bfloat16_t>();
        AscendC::Cast(yBf16, yFloat, AscendC::RoundMode::CAST_RINT, length);
        outQueueBf16.EnQue(yBf16);
        outQueueFloat.FreeTensor(yFloat);
    }

    __aicore__ inline void CopyOut(int32_t offset, int32_t length)
    {
        AscendC::LocalTensor<bfloat16_t> yLocal = outQueueBf16.DeQue<bfloat16_t>();
        AscendC::DataCopy(yGmBf16[offset], yLocal, length);
        outQueueBf16.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueBf16;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueBf16;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueFloat;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> calcQueueTmp1;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> calcQueueTmp2;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueFloat;

    AscendC::GlobalTensor<bfloat16_t> xGmBf16;
    AscendC::GlobalTensor<bfloat16_t> yGmBf16;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t fullTiles;
    uint32_t tailLength;
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    uint32_t totalLength = tilingData.size;
    uint32_t blockNum = AscendC::GetBlockNum();
    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t rawBlockLen = totalLength / blockNum;

    if (tilingData.dtype == 0) {
        // float16: 32B 对齐 = 16 个 half
        constexpr uint32_t ALIGN = 32 / sizeof(half);
        uint32_t alignedLen = (rawBlockLen / ALIGN) * ALIGN;
        uint32_t blockLength = (blockIdx == blockNum - 1)
            ? (totalLength - alignedLen * (blockNum - 1))
            : alignedLen;

        uint32_t tileNum = 1;
        if (blockLength > 22000) {
            tileNum = (blockLength + 22000 - 1) / 22000;
        }
        KernelLogSigmoid<half> op;
        op.Init((GM_ADDR)((__gm__ half *)x + alignedLen * blockIdx),
                (GM_ADDR)((__gm__ half *)y + alignedLen * blockIdx),
                blockLength, tileNum);
        op.Process();
    } else if (tilingData.dtype == 1) {
        // float32: 32B 对齐 = 8 个 float
        constexpr uint32_t ALIGN = 32 / sizeof(float);
        uint32_t alignedLen = (rawBlockLen / ALIGN) * ALIGN;
        uint32_t blockLength = (blockIdx == blockNum - 1)
            ? (totalLength - alignedLen * (blockNum - 1))
            : alignedLen;

        uint32_t tileNum = 1;
        if (blockLength > 11200) {
            tileNum = (blockLength + 11200 - 1) / 11200;
        }
        KernelLogSigmoid<float> op;
        op.Init((GM_ADDR)((__gm__ float *)x + alignedLen * blockIdx),
                (GM_ADDR)((__gm__ float *)y + alignedLen * blockIdx),
                blockLength, tileNum);
        op.Process();
    } else {
        // bf16: 32B 对齐 = 16 个 bf16
        constexpr uint32_t ALIGN = 32 / sizeof(bfloat16_t);
        uint32_t alignedLen = (rawBlockLen / ALIGN) * ALIGN;
        uint32_t blockLength = (blockIdx == blockNum - 1)
            ? (totalLength - alignedLen * (blockNum - 1))
            : alignedLen;

        uint32_t tileNum = 1;
        if (blockLength > 8800) {
            tileNum = (blockLength + 8800 - 1) / 8800;
        }
        KernelLogSigmoidBf16 op;
        op.Init((GM_ADDR)((__gm__ bfloat16_t *)x + alignedLen * blockIdx),
                (GM_ADDR)((__gm__ bfloat16_t *)y + alignedLen * blockIdx),
                blockLength, tileNum);
        op.Process();
    }
}

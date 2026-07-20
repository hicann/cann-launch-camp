// Kernel侧核函数实现
#include "kernel_operator.h"

#include <type_traits>

#include "addcmul_tiling.h"

#include "tiling_key_addcmul.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t BLOCK_SIZE = 32;

template <typename T>
class KernelAddcmul {
public:
    __aicore__ inline KernelAddcmul() {}

    __aicore__ inline void Init(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
                                const AddcmulTilingData &tiling)
    {
        this->totalLength = tiling.totalLength;
        this->tileLength = tiling.tileLength;
        this->isBroadcast = tiling.isBroadcast;
        this->dimNum = tiling.dimNum;
        this->alignNum = BLOCK_SIZE / static_cast<uint32_t>(sizeof(T));
        if (this->alignNum == 0) {
            this->alignNum = 1;
        }

        if (this->totalLength == 0) {
            this->blockLength = 0;
            return;
        }

        const uint32_t blockIdx = GetBlockIdx();
        if (blockIdx < tiling.formerNum) {
            this->blockOffset = blockIdx * tiling.formerLength;
            this->blockLength = tiling.formerLength;
        } else {
            this->blockOffset = tiling.formerNum * tiling.formerLength +
                                (blockIdx - tiling.formerNum) * tiling.tailBlockLength;
            this->blockLength = tiling.tailBlockLength;
        }
        if (this->blockLength == 0) {
            return;
        }

        for (uint32_t i = 0; i < ADDCMUL_MAX_DIMS; ++i) {
            this->outShape[i] = tiling.outShape[i];
            this->outStride[i] = tiling.outStride[i];
            this->inputStride[i] = tiling.inputStride[i];
            this->x1Stride[i] = tiling.x1Stride[i];
            this->x2Stride[i] = tiling.x2Stride[i];
        }

        inputGm.SetGlobalBuffer((__gm__ T *)input_data);
        x1Gm.SetGlobalBuffer((__gm__ T *)x1);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2);
        valueGm.SetGlobalBuffer((__gm__ T *)value);
        yGm.SetGlobalBuffer((__gm__ T *)y);

        this->mValue = valueGm.GetValue(0);
        if constexpr (std::is_same_v<T, half>) {
            this->fValue = static_cast<float>(this->mValue);
        }

        if constexpr (!std::is_same_v<T, int8_t>) {
            if (this->isBroadcast == 0 && this->tileLength > 0) {
                // tileLength 已在 host 侧保证 32B 对齐
                pipe.InitBuffer(inQueueInput, BUFFER_NUM, this->tileLength * sizeof(T));
                pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileLength * sizeof(T));
                pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileLength * sizeof(T));
                pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(T));
            }
        }
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength == 0) {
            return;
        }

        if constexpr (std::is_same_v<T, int8_t>) {
            ProcessScalar();
        } else {
            if (this->isBroadcast == 1 || this->tileLength == 0) {
                ProcessScalar();
            } else {
                ProcessVector();
            }
        }
    }

private:
    __aicore__ inline uint32_t CeilAlign(uint32_t len) const
    {
        return (len + this->alignNum - 1) / this->alignNum * this->alignNum;
    }

    __aicore__ inline uint32_t CalcOffset(uint32_t linearIdx, const uint32_t *stride) const
    {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < this->dimNum; ++i) {
            const uint32_t coord = (linearIdx / this->outStride[i]) % this->outShape[i];
            offset += coord * stride[i];
        }
        return offset;
    }

    __aicore__ inline T CalcResult(T inputVal, T x1Val, T x2Val) const
    {
        if constexpr (std::is_same_v<T, half>) {
            const float r = static_cast<float>(inputVal) +
                            static_cast<float>(x1Val) * static_cast<float>(x2Val) * this->fValue;
            return static_cast<half>(r);
        } else if constexpr (std::is_same_v<T, int8_t>) {
            const int32_t tmp = static_cast<int32_t>(inputVal) +
                                static_cast<int32_t>(x1Val) * static_cast<int32_t>(x2Val) *
                                    static_cast<int32_t>(this->mValue);
            return static_cast<T>(tmp);
        } else {
            return inputVal + x1Val * x2Val * this->mValue;
        }
    }

    __aicore__ inline void ProcessScalar()
    {
        for (uint32_t i = 0; i < this->blockLength; ++i) {
            const uint32_t outIdx = this->blockOffset + i;
            const T inputVal = inputGm.GetValue(CalcOffset(outIdx, this->inputStride));
            const T x1Val = x1Gm.GetValue(CalcOffset(outIdx, this->x1Stride));
            const T x2Val = x2Gm.GetValue(CalcOffset(outIdx, this->x2Stride));
            yGm.SetValue(outIdx, CalcResult(inputVal, x1Val, x2Val));
        }
    }

    __aicore__ inline void ProcessVector()
    {
        if constexpr (!std::is_same_v<T, int8_t>) {
            uint32_t remain = this->blockLength;
            uint32_t progress = 0;
            while (remain > 0) {
                this->processLength = (remain > this->tileLength) ? this->tileLength : remain;
                this->calcLength = CeilAlign(this->processLength);
                if (this->calcLength > this->tileLength) {
                    this->calcLength = this->tileLength;
                }
                CopyIn(progress);
                Compute();
                CopyOut(progress);
                remain -= this->processLength;
                ++progress;
            }
        }
    }

    __aicore__ inline void CopyIn(uint32_t progress)
    {
        if constexpr (!std::is_same_v<T, int8_t>) {
            LocalTensor<T> inputLocal = inQueueInput.AllocTensor<T>();
            LocalTensor<T> x1Local = inQueueX1.AllocTensor<T>();
            LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();

            const uint32_t offset = this->blockOffset + progress * this->tileLength;
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->processLength * sizeof(T)), 0, 0, 0};
            const uint32_t rightPad = this->calcLength - this->processLength;
            DataCopyPadExtParams<T> padParams{rightPad > 0, 0, static_cast<uint8_t>(rightPad), static_cast<T>(0)};
            DataCopyPad(inputLocal, inputGm[offset], copyParams, padParams);
            DataCopyPad(x1Local, x1Gm[offset], copyParams, padParams);
            DataCopyPad(x2Local, x2Gm[offset], copyParams, padParams);

            inQueueInput.EnQue(inputLocal);
            inQueueX1.EnQue(x1Local);
            inQueueX2.EnQue(x2Local);
        }
    }

    __aicore__ inline void Compute()
    {
        if constexpr (!std::is_same_v<T, int8_t>) {
            LocalTensor<T> inputLocal = inQueueInput.DeQue<T>();
            LocalTensor<T> x1Local = inQueueX1.DeQue<T>();
            LocalTensor<T> x2Local = inQueueX2.DeQue<T>();
            LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

            // 用对齐后的 calcLength 做矢量计算（右侧 pad 为 0，不影响有效结果）
            Mul(x1Local, x1Local, x2Local, this->calcLength);
            Muls(x1Local, x1Local, this->mValue, this->calcLength);
            Add(yLocal, x1Local, inputLocal, this->calcLength);

            outQueueY.EnQue(yLocal);
            inQueueInput.FreeTensor(inputLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        if constexpr (!std::is_same_v<T, int8_t>) {
            LocalTensor<T> yLocal = outQueueY.DeQue<T>();
            const uint32_t offset = this->blockOffset + progress * this->tileLength;
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->processLength * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm[offset], yLocal, copyParams);
            outQueueY.FreeTensor(yLocal);
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueInput, inQueueX1, inQueueX2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    GlobalTensor<T> inputGm, x1Gm, x2Gm, valueGm, yGm;

    T mValue;
    float fValue{0.0f};

    uint32_t totalLength{0};
    uint32_t tileLength{0};
    uint32_t blockOffset{0};
    uint32_t blockLength{0};
    uint32_t processLength{0};
    uint32_t calcLength{0};
    uint32_t alignNum{1};
    uint32_t isBroadcast{0};
    uint32_t dimNum{0};
    uint32_t outShape[ADDCMUL_MAX_DIMS]{};
    uint32_t outStride[ADDCMUL_MAX_DIMS]{};
    uint32_t inputStride[ADDCMUL_MAX_DIMS]{};
    uint32_t x1Stride[ADDCMUL_MAX_DIMS]{};
    uint32_t x2Stride[ADDCMUL_MAX_DIMS]{};
};

template <typename DT_INPUT_DATA>
__global__ __aicore__ void addcmul(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
                                   GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddcmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddcmulTilingData, tiling_data, tiling);
    KernelAddcmul<DT_INPUT_DATA> op;
    op.Init(input_data, x1, x2, value, y, tiling_data);
    op.Process();
}

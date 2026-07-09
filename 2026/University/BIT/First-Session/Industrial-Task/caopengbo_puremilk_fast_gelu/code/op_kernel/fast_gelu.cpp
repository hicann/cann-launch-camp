#include "kernel_operator.h"

#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr float FASTGELU_ALPHA = 1.702f;
constexpr uint32_t BLOCK_SIZE_BYTES = 32;
constexpr uint32_t SMALL_FLOAT32_SIGMOID_THRESHOLD = 24;

template <class DT_X>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, uint32_t tileDataNum)
    {
        this->totalDataNum = length;

        uint32_t coreNum = AscendC::GetBlockNum();
        if (coreNum == 0) {
            coreNum = 1;
        }

        uint32_t coreIdx = AscendC::GetBlockIdx();

        uint32_t alignElem = BLOCK_SIZE_BYTES / sizeof(DT_X);
        if (alignElem == 0) {
            alignElem = 1;
        }

        uint32_t perCore = CeilDiv(length, coreNum);
        uint32_t alignedPerCore = AlignUp(perCore, alignElem);
        uint32_t coreOffset = coreIdx * alignedPerCore;

        if (coreOffset >= length) {
            this->coreDataNum = 0;
            this->tileNum = 0;
            return;
        }

        uint32_t remain = length - coreOffset;
        this->coreDataNum = remain > alignedPerCore ? alignedPerCore : remain;
        this->tileDataNum = tileDataNum == 0 ? 1u : tileDataNum;

        if (this->tileDataNum > this->coreDataNum) {
            this->tileDataNum = this->coreDataNum;
        }

        this->tileNum = CeilDiv(this->coreDataNum, this->tileDataNum);
        this->tailDataNum = this->coreDataNum - (this->tileNum - 1) * this->tileDataNum;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + coreOffset, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + coreOffset, this->coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DT_X));
        pipe.InitBuffer(downBuf, this->tileDataNum * sizeof(float));

        if constexpr (!AscendC::IsSameType<DT_X, float>::value) {
            pipe.InitBuffer(castBuf, this->tileDataNum * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->tileNum; ++i) {
            uint32_t curNum = (i + 1 == this->tileNum) ? this->tailDataNum : this->tileDataNum;
            CopyIn(i, curNum);
            Compute(curNum);
            CopyOut(i, curNum);
        }
    }

private:
    __aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b)
    {
        return b == 0 ? 0 : (a + b - 1) / b;
    }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align)
    {
        return align == 0 ? value : ((value + align - 1) / align) * align;
    }

    __aicore__ inline void CopyIn(uint32_t progress, uint32_t curNum)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        uint32_t copyBytes = curNum * sizeof(DT_X);

        if ((copyBytes % BLOCK_SIZE_BYTES) == 0) {
            AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], curNum);
        } else {
            AscendC::DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(xLocal, xGm[progress * this->tileDataNum], copyParams, padParams);
        }

        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t curNum)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        AscendC::LocalTensor<float> down = downBuf.Get<float>();

        if constexpr (AscendC::IsSameType<DT_X, float>::value) {
            if (this->totalDataNum <= SMALL_FLOAT32_SIGMOID_THRESHOLD) {
                AscendC::Muls(yLocal, xLocal, FASTGELU_ALPHA, curNum);
                AscendC::Sigmoid(yLocal, yLocal, curNum);
                AscendC::Mul(yLocal, yLocal, xLocal, curNum);
            } else {
                AscendC::Muls(down, xLocal, -FASTGELU_ALPHA, curNum);
                AscendC::Exp(down, down, curNum);
                AscendC::Adds(down, down, 1.0f, curNum);
                AscendC::Div(yLocal, xLocal, down, curNum);
            }
        } else {
            AscendC::LocalTensor<float> xF = castBuf.Get<float>();

            AscendC::Cast(xF, xLocal, AscendC::RoundMode::CAST_NONE, curNum);
            AscendC::Muls(down, xF, -FASTGELU_ALPHA, curNum);
            AscendC::Exp(down, down, curNum);
            AscendC::Adds(down, down, 1.0f, curNum);
            AscendC::Div(xF, xF, down, curNum);
            AscendC::Cast(yLocal, xF, AscendC::RoundMode::CAST_RINT, curNum);
        }

        outQueueY.EnQue<DT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t curNum)
    {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        uint32_t copyBytes = curNum * sizeof(DT_X);

        if ((copyBytes % BLOCK_SIZE_BYTES) == 0) {
            AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, curNum);
        } else {
            AscendC::DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            AscendC::DataCopyPad(yGm[progress * this->tileDataNum], yLocal, copyParams);
        }

        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> downBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> castBuf;

    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;

    uint32_t totalDataNum{0};
    uint32_t coreDataNum{0};
    uint32_t tileDataNum{0};
    uint32_t tileNum{0};
    uint32_t tailDataNum{0};
};

template <typename DT_X>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tiling_data, tiling);

    KernelFastGelu<DT_X> op;
    op.Init(x, y, tiling_data.length, tiling_data.tileDataNum);
    op.Process();
}

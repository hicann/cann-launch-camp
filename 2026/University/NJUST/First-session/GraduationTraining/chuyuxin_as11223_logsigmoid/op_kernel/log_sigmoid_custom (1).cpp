#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

#include <type_traits>

constexpr int32_t BUFFER_NUM = 2;

template<typename TYPE_X, typename TYPE_Y>
class KernelLogSigmoid {
public:
    __aicore__ inline KernelLogSigmoid() {}

    __aicore__ inline void Init(GM_ADDR x,
                                GM_ADDR y,
                                uint32_t totalDataNum,
                                uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum,
                                uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum,
                                uint32_t tileDataNum,
                                uint32_t smallTailDataNum,
                                uint32_t bigTailDataNum,
                                uint32_t tailBlockNum)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * blockIdx;

        this->totalDataNum = totalDataNum;
        this->tileDataNum = tileDataNum;
        this->globalBufferIndex = globalBufferIndex;

        if (blockIdx < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;

            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (blockIdx - tailBlockNum);
            this->globalBufferIndex = globalBufferIndex;
        }

        if (this->globalBufferIndex >= this->totalDataNum) {
            this->coreDataNum = 0;
            this->tileNum = 0;
            this->tailDataNum = 0;
            return;
        }

        if (this->globalBufferIndex + this->coreDataNum > this->totalDataNum) {
            this->coreDataNum = this->totalDataNum - this->globalBufferIndex;
            this->tileNum = (this->coreDataNum + this->tileDataNum - 1) / this->tileDataNum;
            this->tailDataNum = this->coreDataNum - (this->tileNum - 1) * this->tileDataNum;
        }

        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + this->globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + this->globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));

        pipe.InitBuffer(tmpBuf0, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpBuf1, this->tileDataNum * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < this->tileNum; ++i) {
            this->processDataNum = this->tileDataNum;

            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
            }

            if (this->processDataNum == 0) {
                continue;
            }

            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();

        AscendC::LocalTensor<float> tmp0 = tmpBuf0.Get<float>();
        AscendC::LocalTensor<float> tmp1 = tmpBuf1.Get<float>();

        /*
         * LogSigmoid(x) = log(1 / (1 + exp(-x)))
         *               = -log(1 + exp(-x))
         */

        if constexpr (std::is_same<TYPE_X, float>::value) {
            AscendC::Muls(tmp0, xLocal, -1.0f, this->processDataNum);
        } else {
            AscendC::Cast(tmp0, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Muls(tmp0, tmp0, -1.0f, this->processDataNum);
        }

        AscendC::Exp(tmp1, tmp0, this->processDataNum);
        AscendC::Adds(tmp1, tmp1, 1.0f, this->processDataNum);
        AscendC::Ln(tmp0, tmp1, this->processDataNum);
        AscendC::Muls(tmp0, tmp0, -1.0f, this->processDataNum);

        if constexpr (std::is_same<TYPE_Y, float>::value) {
            AscendC::Muls(yLocal, tmp0, 1.0f, this->processDataNum);
        } else if constexpr (std::is_same<TYPE_Y, bfloat16_t>::value) {
            AscendC::Cast(yLocal, tmp0, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        } else {
            AscendC::Cast(yLocal, tmp0, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        }

        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;

    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;

    uint32_t totalDataNum;
    uint32_t globalBufferIndex;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x,
                                                          GM_ADDR y,
                                                          GM_ADDR workspace,
                                                          GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelLogSigmoid<DTYPE_X, DTYPE_Y> op;

    op.Init(x,
            y,
            tilingData.size,
            tilingData.smallCoreDataNum,
            tilingData.bigCoreDataNum,
            tilingData.finalBigTileNum,
            tilingData.finalSmallTileNum,
            tilingData.tileDataNum,
            tilingData.smallTailDataNum,
            tilingData.bigTailDataNum,
            tilingData.tailBlockNum);

    op.Process();
}

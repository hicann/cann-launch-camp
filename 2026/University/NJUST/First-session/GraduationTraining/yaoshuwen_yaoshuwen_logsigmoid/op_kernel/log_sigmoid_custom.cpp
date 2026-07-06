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
                                uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum,
                                uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum,
                                uint32_t tileDataNum,
                                uint32_t smallTailDataNum,
                                uint32_t bigTailDataNum,
                                uint32_t tailBlockNum)
    {
        uint32_t coreIdx = AscendC::GetBlockIdx();

        this->tileDataNum = tileDataNum;

        uint32_t globalBufferIndex = bigCoreDataNum * coreIdx;

        if (coreIdx < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;

            globalBufferIndex -=
                (bigCoreDataNum - smallCoreDataNum) * (coreIdx - tailBlockNum);
        }

        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));

        if constexpr (std::is_same<TYPE_X, bfloat16_t>::value) {
            pipe.InitBuffer(tmpFloatX, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmpFloatY, this->tileDataNum * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->tileNum; i++) {
            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
            } else {
                this->processDataNum = this->tileDataNum;
            }

            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<TYPE_X> xLocal =
            inQueueX.AllocTensor<TYPE_X>();

        AscendC::DataCopy(
            xLocal,
            xGm[progress * this->tileDataNum],
            this->processDataNum
        );

        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<TYPE_X> xLocal =
            inQueueX.DeQue<TYPE_X>();

        AscendC::LocalTensor<TYPE_Y> yLocal =
            outQueueY.AllocTensor<TYPE_Y>();

        if constexpr (std::is_same<TYPE_X, bfloat16_t>::value) {
            AscendC::LocalTensor<float> xFloat = tmpFloatX.Get<float>();
            AscendC::LocalTensor<float> yFloat = tmpFloatY.Get<float>();

            AscendC::Cast(
                xFloat,
                xLocal,
                AscendC::RoundMode::CAST_NONE,
                this->processDataNum
            );

            AscendC::Sigmoid(
                yFloat,
                xFloat,
                this->processDataNum
            );

            AscendC::Ln(
                yFloat,
                yFloat,
                this->processDataNum
            );

            AscendC::Cast(
                yLocal,
                yFloat,
                AscendC::RoundMode::CAST_RINT,
                this->processDataNum
            );
        } else {
            AscendC::Sigmoid(
                yLocal,
                xLocal,
                this->processDataNum
            );

            AscendC::Ln(
                yLocal,
                yLocal,
                this->processDataNum
            );
        }

        outQueueY.EnQue<TYPE_Y>(yLocal);

        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<TYPE_Y> yLocal =
            outQueueY.DeQue<TYPE_Y>();

        AscendC::DataCopy(
            yGm[progress * this->tileDataNum],
            yLocal,
            this->processDataNum
        );

        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpFloatX;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpFloatY;

    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelLogSigmoid<DTYPE_X, DTYPE_Y> op;

    op.Init(x,
            y,
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

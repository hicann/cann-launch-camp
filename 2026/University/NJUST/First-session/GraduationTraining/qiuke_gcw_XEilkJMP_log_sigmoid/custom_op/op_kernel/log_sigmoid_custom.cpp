%%writefile Sources/test/custom_op/op_kernel/log_sigmoid_custom.cpp
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
        uint32_t globalBufferIndex = bigCoreDataNum * coreIdx;

        this->tileDataNum = tileDataNum;

        if (coreIdx < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (coreIdx - tailBlockNum);
        }

        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));

        // float中间buffer：用于兼容BF16，并提高half计算稳定性
        pipe.InitBuffer(tmpXBuffer, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpYBuffer, this->tileDataNum * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;

        for (int32_t i = 0; i < loopCount; i++) {
            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
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

        AscendC::DataCopy(
            xLocal,
            xGm[progress * this->tileDataNum],
            this->processDataNum
        );

        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();

        AscendC::LocalTensor<float> tmpX = tmpXBuffer.Get<float>();
        AscendC::LocalTensor<float> tmpY = tmpYBuffer.Get<float>();

        // 1. 输入统一转成float计算
        if constexpr (std::is_same<TYPE_X, float>::value) {
            AscendC::Adds(tmpX, xLocal, 0.0f, this->processDataNum);
        } else {
            AscendC::Cast(tmpX, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // 2. tmpY = -x
        AscendC::Muls(tmpY, tmpX, -1.0f, this->processDataNum);
        AscendC::PipeBarrier<PIPE_V>();

        // 3. tmpY = exp(-x)
        AscendC::Exp(tmpY, tmpY, this->processDataNum);
        AscendC::PipeBarrier<PIPE_V>();

        // 4. tmpY = 1 + exp(-x)
        AscendC::Adds(tmpY, tmpY, 1.0f, this->processDataNum);
        AscendC::PipeBarrier<PIPE_V>();

        // 5. tmpY = log(1 + exp(-x))
        AscendC::Ln(tmpY, tmpY, this->processDataNum);
        AscendC::PipeBarrier<PIPE_V>();

        // 6. tmpY = -log(1 + exp(-x))
        AscendC::Muls(tmpY, tmpY, -1.0f, this->processDataNum);
        AscendC::PipeBarrier<PIPE_V>();

        // 7. 输出转回原类型
        if constexpr (std::is_same<TYPE_Y, float>::value) {
            AscendC::Adds(yLocal, tmpY, 0.0f, this->processDataNum);
        } else {
            AscendC::Cast(yLocal, tmpY, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        }
        AscendC::PipeBarrier<PIPE_V>();

        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();

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

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpXBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpYBuffer;

    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;

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
#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

#include <type_traits>

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X, typename TYPE_Y>
class KernelLogSigmoidCustom {
public:
    __aicore__ inline KernelLogSigmoidCustom() {}

    __aicore__ inline void Init(
        GM_ADDR x,
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
        uint32_t blockIdx = AscendC::GetBlockIdx();

        this->tileDataNum = tileDataNum;

        uint32_t globalBufferIndex = 0;

        if (blockIdx < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;

            globalBufferIndex = blockIdx * bigCoreDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;

            globalBufferIndex =
                tailBlockNum * bigCoreDataNum +
                (blockIdx - tailBlockNum) * smallCoreDataNum;
        }

        xGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ TYPE_X*>(x) + globalBufferIndex,
            this->coreDataNum);

        yGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ TYPE_Y*>(y) + globalBufferIndex,
            this->coreDataNum);

        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            this->tileDataNum * sizeof(TYPE_X));

        pipe.InitBuffer(
            outQueueY,
            BUFFER_NUM,
            this->tileDataNum * sizeof(TYPE_Y));

        if constexpr (std::is_same<TYPE_X, bfloat16_t>::value) {
            pipe.InitBuffer(tmpBuf0, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmpBuf1, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmpBuf2, this->tileDataNum * sizeof(float));
        } else {
            pipe.InitBuffer(tmpBuf0, this->tileDataNum * sizeof(TYPE_X));
            pipe.InitBuffer(tmpBuf1, this->tileDataNum * sizeof(TYPE_X));
        }
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->tileNum; ++i) {
            this->processDataNum = this->tileDataNum;

            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
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
            this->processDataNum);

        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<TYPE_X> xLocal =
            inQueueX.DeQue<TYPE_X>();

        AscendC::LocalTensor<TYPE_Y> yLocal =
            outQueueY.AllocTensor<TYPE_Y>();

        if constexpr (std::is_same<TYPE_X, bfloat16_t>::value) {
            AscendC::LocalTensor<float> xFloat =
                tmpBuf0.Get<float>();

            AscendC::LocalTensor<float> tmp0 =
                tmpBuf1.Get<float>();

            AscendC::LocalTensor<float> tmp1 =
                tmpBuf2.Get<float>();

            // xFloat = float(x)
            AscendC::Cast(
                xFloat,
                xLocal,
                AscendC::RoundMode::CAST_NONE,
                this->processDataNum);

            AscendC::PipeBarrier<PIPE_V>();

            // tmp0 = -x
            AscendC::Muls(
                tmp0,
                xFloat,
                -1.0f,
                this->processDataNum);

            AscendC::PipeBarrier<PIPE_V>();

            // tmp1 = exp(-x)
            AscendC::Exp(
                tmp1,
                tmp0,
                this->processDataNum);

            AscendC::PipeBarrier<PIPE_V>();

            // tmp0 = 1 + exp(-x)
            AscendC::Adds(
                tmp0,
                tmp1,
                1.0f,
                this->processDataNum);

            AscendC::PipeBarrier<PIPE_V>();

            // tmp1 = log(1 + exp(-x))
            AscendC::Ln(
                tmp1,
                tmp0,
                this->processDataNum);

            AscendC::PipeBarrier<PIPE_V>();

            // tmp0 = -log(1 + exp(-x))
            AscendC::Muls(
                tmp0,
                tmp1,
                -1.0f,
                this->processDataNum);

            AscendC::PipeBarrier<PIPE_V>();

            // y = cast(tmp0) to bf16
            AscendC::Cast(
                yLocal,
                tmp0,
                AscendC::RoundMode::CAST_RINT,
                this->processDataNum);
        } else {
            AscendC::LocalTensor<TYPE_X> tmp0 =
                tmpBuf0.Get<TYPE_X>();

            AscendC::LocalTensor<TYPE_X> tmp1 =
                tmpBuf1.Get<TYPE_X>();

            // tmp0 = -x
            AscendC::Muls(
                tmp0,
                xLocal,
                static_cast<TYPE_X>(-1.0f),
                this->processDataNum);

            AscendC::PipeBarrier<PIPE_V>();

            // tmp1 = exp(-x)
            AscendC::Exp(
                tmp1,
                tmp0,
                this->processDataNum);

            AscendC::PipeBarrier<PIPE_V>();

            // tmp0 = 1 + exp(-x)
            AscendC::Adds(
                tmp0,
                tmp1,
                static_cast<TYPE_X>(1.0f),
                this->processDataNum);

            AscendC::PipeBarrier<PIPE_V>();

            // tmp1 = log(1 + exp(-x))
            AscendC::Ln(
                tmp1,
                tmp0,
                this->processDataNum);

            AscendC::PipeBarrier<PIPE_V>();

            // y = -log(1 + exp(-x))
            AscendC::Muls(
                yLocal,
                tmp1,
                static_cast<TYPE_X>(-1.0f),
                this->processDataNum);
        }

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<TYPE_Y> yLocal =
            outQueueY.DeQue<TYPE_Y>();

        AscendC::DataCopy(
            yGm[progress * this->tileDataNum],
            yLocal,
            this->processDataNum);

        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf2;

    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;

    uint32_t coreDataNum = 0;
    uint32_t tileNum = 0;
    uint32_t tileDataNum = 0;
    uint32_t tailDataNum = 0;
    uint32_t processDataNum = 0;
};

extern "C" __global__ __aicore__
void log_sigmoid_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelLogSigmoidCustom<DTYPE_X, DTYPE_Y> op;

    op.Init(
        x,
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
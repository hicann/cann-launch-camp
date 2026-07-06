#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 1;

__aicore__ inline void CastInputToFloat(AscendC::LocalTensor<float> dst,
                                        AscendC::LocalTensor<float> src,
                                        uint32_t count)
{
    AscendC::Adds(dst, src, 0.0f, count);
}

__aicore__ inline void CastInputToFloat(AscendC::LocalTensor<float> dst,
                                        AscendC::LocalTensor<half> src,
                                        uint32_t count)
{
    AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
}

__aicore__ inline void CastInputToFloat(AscendC::LocalTensor<float> dst,
                                        AscendC::LocalTensor<bfloat16_t> src,
                                        uint32_t count)
{
    AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
}

__aicore__ inline void CastFloatToOutput(AscendC::LocalTensor<float> dst,
                                         AscendC::LocalTensor<float> src,
                                         uint32_t count)
{
    AscendC::Adds(dst, src, 0.0f, count);
}

__aicore__ inline void CastFloatToOutput(AscendC::LocalTensor<half> dst,
                                         AscendC::LocalTensor<float> src,
                                         uint32_t count)
{
    AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
}

__aicore__ inline void CastFloatToOutput(AscendC::LocalTensor<bfloat16_t> dst,
                                         AscendC::LocalTensor<float> src,
                                         uint32_t count)
{
    AscendC::Cast(dst, src, AscendC::RoundMode::CAST_RINT, count);
}

template <typename TYPE_X, typename TYPE_Y>
class KernelLogSigmoid {
public:
    __aicore__ inline KernelLogSigmoid() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
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

        uint32_t globalBufferIndex = 0;
        if (coreIdx < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
            globalBufferIndex = bigCoreDataNum * coreIdx;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex = bigCoreDataNum * tailBlockNum + smallCoreDataNum * (coreIdx - tailBlockNum);
        }

        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));

        pipe.InitBuffer(tmpBuf0, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpBuf1, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpBuf2, this->tileDataNum * sizeof(float));
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
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();

        AscendC::LocalTensor<float> tmp0 = tmpBuf0.Get<float>();
        AscendC::LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
        AscendC::LocalTensor<float> tmp2 = tmpBuf2.Get<float>();

        CastInputToFloat(tmp0, xLocal, this->processDataNum);

        // LogSigmoid(x) = log(1 / (1 + exp(-x)))
        //               = -ln(1 + exp(-x))
        AscendC::Muls(tmp1, tmp0, -1.0f, this->processDataNum);
        AscendC::Exp(tmp1, tmp1, this->processDataNum);
        AscendC::Adds(tmp2, tmp1, 1.0f, this->processDataNum);
        AscendC::Ln(tmp0, tmp2, this->processDataNum);
        AscendC::Muls(tmp0, tmp0, -1.0f, this->processDataNum);

        CastFloatToOutput(yLocal, tmp0, this->processDataNum);

        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf2;

    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelLogSigmoid<DTYPE_X, DTYPE_Y> op;
    op.Init(x, y,
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

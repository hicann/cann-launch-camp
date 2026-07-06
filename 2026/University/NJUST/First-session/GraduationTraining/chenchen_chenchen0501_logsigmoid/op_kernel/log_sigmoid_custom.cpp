#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 1;

// 判断当前模板类型是否为 bfloat16
template <typename T>
struct IsBfloat16 {
    static constexpr bool value = false;
};

template <>
struct IsBfloat16<bfloat16_t> {
    static constexpr bool value = true;
};

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

        // 非 BF16 直接用原 dtype 计算；BF16 使用 float 临时变量计算
        constexpr uint32_t calcTypeSize = IsBfloat16<TYPE_X>::value ? sizeof(float) : sizeof(TYPE_X);
        pipe.InitBuffer(tmpBuf0, this->tileDataNum * calcTypeSize);
        pipe.InitBuffer(tmpBuf1, this->tileDataNum * calcTypeSize);
        pipe.InitBuffer(tmpBuf2, this->tileDataNum * calcTypeSize);
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

    if constexpr (IsBfloat16<TYPE_X>::value) {
        // BF16 特殊处理：先 Cast 到 float 计算，最后再 Cast 回 BF16
        AscendC::LocalTensor<float> tmp0 = tmpBuf0.Get<float>();
        AscendC::LocalTensor<float> tmp1 = tmpBuf1.Get<float>();

        // tmp0 = float(x)
        AscendC::Cast(tmp0, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);

        // tmp0 = -x
        AscendC::Muls(tmp0, tmp0, -1.0f, this->processDataNum);

        // tmp0 = exp(-x)
        AscendC::Exp(tmp0, tmp0, this->processDataNum);

        // tmp1 = 1 + exp(-x)
        AscendC::Adds(tmp1, tmp0, 1.0f, this->processDataNum);

        // tmp1 = ln(1 + exp(-x))
        AscendC::Ln(tmp1, tmp1, this->processDataNum);

        // tmp1 = -ln(1 + exp(-x))
        AscendC::Muls(tmp1, tmp1, -1.0f, this->processDataNum);

        // y = cast(tmp1)
        AscendC::Cast(yLocal, tmp1, AscendC::RoundMode::CAST_RINT, this->processDataNum);
    } else {
        // float32 / float16 直接用原始 dtype 计算，避免无意义 Cast
        AscendC::LocalTensor<TYPE_X> tmp0 = tmpBuf0.Get<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> tmp1 = tmpBuf1.Get<TYPE_X>();

        // tmp0 = -x
        AscendC::Muls(tmp0, xLocal, static_cast<TYPE_X>(-1.0), this->processDataNum);

        // tmp0 = exp(-x)
        AscendC::Exp(tmp0, tmp0, this->processDataNum);

        // tmp1 = 1 + exp(-x)
        AscendC::Adds(tmp1, tmp0, static_cast<TYPE_X>(1.0), this->processDataNum);

        // tmp1 = ln(1 + exp(-x))
        AscendC::Ln(tmp1, tmp1, this->processDataNum);

        // y = -ln(1 + exp(-x))
        AscendC::Muls(yLocal, tmp1, static_cast<TYPE_X>(-1.0), this->processDataNum);
    }

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

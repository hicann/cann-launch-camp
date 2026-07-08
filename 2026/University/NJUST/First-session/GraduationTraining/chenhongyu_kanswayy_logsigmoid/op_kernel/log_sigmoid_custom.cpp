
#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
struct IsFloatType
{
    static constexpr bool value = false;
};

template <>
struct IsFloatType<float>
{
    static constexpr bool value = true;
};

template <typename TYPE_X, typename TYPE_Y>
class KernelLogSigmoid
{
public:
    __aicore__ inline KernelLogSigmoid() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t normalCoreElemNum,
                                uint32_t largeCoreElemNum,
                                uint32_t tileElemNum,
                                uint32_t normalTileNum,
                                uint32_t largeTileNum,
                                uint32_t normalTailElemNum,
                                uint32_t largeTailElemNum,
                                uint32_t largeCoreNum)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();

        this->tileElemNum = tileElemNum;

        uint32_t startOffset = 0;

        if (blockIdx < largeCoreNum)
        {
            this->coreElemNum = largeCoreElemNum;
            this->tileNum = largeTileNum;
            this->tailElemNum = largeTailElemNum;
            startOffset = blockIdx * largeCoreElemNum;
        }
        else
        {
            this->coreElemNum = normalCoreElemNum;
            this->tileNum = normalTileNum;
            this->tailElemNum = normalTailElemNum;
            startOffset = largeCoreNum * largeCoreElemNum +
                          (blockIdx - largeCoreNum) * normalCoreElemNum;
        }

        xGm.SetGlobalBuffer((__gm__ TYPE_X *)x + startOffset, this->coreElemNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y *)y + startOffset, this->coreElemNum);

        pipe.InitBuffer(inputQueue, BUFFER_NUM, this->tileElemNum * sizeof(TYPE_X));
        pipe.InitBuffer(outputQueue, BUFFER_NUM, this->tileElemNum * sizeof(TYPE_Y));

        if constexpr (IsFloatType<TYPE_X>::value)
        {
            // float32：一个 float 临时 tensor 保存 sigmoid 结果
            pipe.InitBuffer(calcBuf, this->tileElemNum * sizeof(float));
        }
        else
        {
            // half / bfloat16：
            // 一个 float tensor 保存 Cast 后的 x
            // 一个 float tensor 保存 sigmoid 结果
            pipe.InitBuffer(calcBuf, this->tileElemNum * sizeof(float) * 2);
        }
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->tileNum; i++)
        {
            if (i == this->tileNum - 1)
            {
                this->currentElemNum = this->tailElemNum;
            }
            else
            {
                this->currentElemNum = this->tileElemNum;
            }

            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t tileIdx)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inputQueue.AllocTensor<TYPE_X>();
        AscendC::DataCopy(xLocal,
                          xGm[tileIdx * this->tileElemNum],
                          this->currentElemNum);
        inputQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inputQueue.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> yLocal = outputQueue.AllocTensor<TYPE_Y>();

        if constexpr (IsFloatType<TYPE_X>::value)
        {
            AscendC::LocalTensor<float> tmpLocal = calcBuf.Get<float>();

            AscendC::Sigmoid(tmpLocal, xLocal, this->currentElemNum);
            AscendC::Ln(yLocal, tmpLocal, this->currentElemNum);
        }
        else
        {
            AscendC::LocalTensor<float> xFloat = calcBuf.Get<float>();
            AscendC::LocalTensor<float> yFloat = xFloat[this->tileElemNum];

            AscendC::Cast(xFloat,
                          xLocal,
                          AscendC::RoundMode::CAST_NONE,
                          this->currentElemNum);

            AscendC::Sigmoid(yFloat, xFloat, this->currentElemNum);
            AscendC::Ln(xFloat, yFloat, this->currentElemNum);

            AscendC::Cast(yLocal,
                          xFloat,
                          AscendC::RoundMode::CAST_RINT,
                          this->currentElemNum);
        }

        outputQueue.EnQue<TYPE_Y>(yLocal);
        inputQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx)
    {
        AscendC::LocalTensor<TYPE_Y> yLocal = outputQueue.DeQue<TYPE_Y>();
        AscendC::DataCopy(yGm[tileIdx * this->tileElemNum],
                          yLocal,
                          this->currentElemNum);
        outputQueue.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inputQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuf;

    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;

    uint32_t coreElemNum;
    uint32_t tileElemNum;
    uint32_t tileNum;
    uint32_t tailElemNum;
    uint32_t currentElemNum;
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelLogSigmoid<DTYPE_X, DTYPE_Y> op;

    op.Init(x, y,
            tilingData.normalCoreElemNum,
            tilingData.largeCoreElemNum,
            tilingData.tileElemNum,
            tilingData.normalTileNum,
            tilingData.largeTileNum,
            tilingData.normalTailElemNum,
            tilingData.largeTailElemNum,
            tilingData.largeCoreNum);

    op.Process();
}
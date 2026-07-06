#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"
#include <type_traits>

constexpr int32_t BUFFER_NUM = 2;

template<typename TYPE_X, typename TYPE_Y>
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
        uint32_t coreId = AscendC::GetBlockIdx();
        uint32_t globalOffset = bigCoreDataNum * coreId;
        this->tileDataNum = tileDataNum;

        // 区分大核（多承担1块数据）和普通核
        if (coreId < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            // 修正普通核的全局内存起始偏移
            globalOffset -= (bigCoreDataNum - smallCoreDataNum) * (coreId - tailBlockNum);
        }

        // 绑定输入输出全局内存
        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalOffset, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalOffset, this->coreDataNum);

        // 初始化流水线队列与临时缓冲区
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
        pipe.InitBuffer(tmpFloatX, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpFloatY, this->tileDataNum * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < this->tileNum; i++) {
            // 最后一个Tile处理尾块数据量
            this->processDataNum = (i == this->tileNum - 1) ? this->tailDataNum : this->tileDataNum;
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

        AscendC::LocalTensor<float> xFloat = tmpFloatX.Get<float>();
        AscendC::LocalTensor<float> yFloat = tmpFloatY.Get<float>();

        // 编译期类型派发：低精度转float32计算，float直接计算
        if constexpr (std::is_same<TYPE_X, float>::value) {
            ComputeLogSigmoid(yFloat, xLocal, this->processDataNum);
            AscendC::DataCopy(yLocal, yFloat, this->processDataNum);
        } else if constexpr (std::is_same<TYPE_X, half>::value) {
            AscendC::Cast(xFloat, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            ComputeLogSigmoid(yFloat, xFloat, this->processDataNum);
            AscendC::Cast(yLocal, yFloat, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        } else if constexpr (std::is_same<TYPE_X, __bf16>::value) {
            AscendC::Cast(xFloat, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            ComputeLogSigmoid(yFloat, xFloat, this->processDataNum);
            AscendC::Cast(yLocal, yFloat, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        }

        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeLogSigmoid(AscendC::LocalTensor<float>& output,
                                             AscendC::LocalTensor<float>& input,
                                             uint32_t dataNum)
    {
        // LogSigmoid(x) = -log(1 + exp(-x))
        AscendC::Muls(output, input, -1.0f, dataNum);   // -x
        AscendC::Exp(output, output, dataNum);           // exp(-x)
        AscendC::Adds(output, output, 1.0f, dataNum);    // 1 + exp(-x)
        AscendC::Log(output, output, dataNum);           // log(1 + exp(-x))
        AscendC::Muls(output, output, -1.0f, dataNum);   // -log(1 + exp(-x))
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

// 核函数入口：框架自动注入DTYPE_X/DTYPE_Y，按类型实例化模板
extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x,
                                                         GM_ADDR y,
                                                         GM_ADDR workspace,
                                                         GM_ADDR tiling)
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

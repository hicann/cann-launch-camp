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
        uint32_t coreIdx = AscendC::GetBlockIdx();
        this->tileDataNum = tileDataNum;

        // ========== 修正：直观方式计算全局起始偏移，避免公式错误 ==========
        uint32_t globalOffset;
        if (coreIdx < tailBlockNum) {
            // 大Core：前面有coreIdx个大Core，每个处理bigCoreDataNum个元素
            globalOffset = coreIdx * bigCoreDataNum;
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            // 普通Core：前面有tailBlockNum个大Core + (coreIdx - tailBlockNum)个普通Core
            globalOffset = tailBlockNum * bigCoreDataNum + (coreIdx - tailBlockNum) * smallCoreDataNum;
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
        }

        // 绑定全局内存地址与长度
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE_X*>(x) + globalOffset, this->coreDataNum);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE_Y*>(y) + globalOffset, this->coreDataNum);

        // 初始化管道缓冲区
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
        pipe.InitBuffer(tmpFloatX, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpFloatZ, this->tileDataNum * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < this->tileNum; i++) {
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
        AscendC::LocalTensor<TYPE_Y> zLocal = outQueueZ.AllocTensor<TYPE_Y>();

        AscendC::LocalTensor<float> xFloat = tmpFloatX.Get<float>();
        AscendC::LocalTensor<float> tmpFloat = tmpFloatZ.Get<float>();

        // 输入统一转float32
        if constexpr (std::is_same<TYPE_X, float>::value) {
            AscendC::DataCopy(xFloat, xLocal, this->processDataNum);
        } else if constexpr (std::is_same<TYPE_X, half>::value) {
            AscendC::Cast(xFloat, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        } else if constexpr (std::is_same<TYPE_X, __bf16>::value) {
            AscendC::Cast(xFloat, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        }

        // 数值稳定版 LogSigmoid 计算
        ComputeLogSigmoid(xFloat, tmpFloat, this->processDataNum);

        // 结果转成输出类型
        if constexpr (std::is_same<TYPE_Y, float>::value) {
            AscendC::DataCopy(zLocal, xFloat, this->processDataNum);
        } else if constexpr (std::is_same<TYPE_Y, half>::value) {
            AscendC::Cast(zLocal, xFloat, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        } else if constexpr (std::is_same<TYPE_Y, __bf16>::value) {
            AscendC::Cast(zLocal, xFloat, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        }

        outQueueZ.EnQue<TYPE_Y>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    // 数值稳定公式：log_sigmoid(x) = min(x, 0) - log(1 + exp(-|x|))
    __aicore__ inline void ComputeLogSigmoid(AscendC::LocalTensor<float>& inOut,
                                             AscendC::LocalTensor<float>& tmp,
                                             uint32_t count)
    {
        AscendC::Abs(tmp, inOut, count);           // tmp = |x|
        AscendC::Muls(tmp, tmp, -1.0f, count);     // tmp = -|x|
        AscendC::Exp(tmp, tmp, count);             // tmp = exp(-|x|)
        AscendC::Adds(tmp, tmp, 1.0f, count);      // tmp = 1 + exp(-|x|)
        AscendC::Log(tmp, tmp, count);             // tmp = log(1 + exp(-|x|))
        AscendC::Mins(inOut, inOut, 0.0f, count);  // inOut = min(x, 0)
        AscendC::Sub(inOut, inOut, tmp, count);    // inOut = min(x,0) - log(...)
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_Y> zLocal = outQueueZ.DeQue<TYPE_Y>();
        AscendC::DataCopy(zGm[progress * this->tileDataNum], zLocal, this->processDataNum);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpFloatX;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpFloatZ;

    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> zGm;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

// 核函数入口
extern "C" __global__ __aicore__ void log_sigmoid_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
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

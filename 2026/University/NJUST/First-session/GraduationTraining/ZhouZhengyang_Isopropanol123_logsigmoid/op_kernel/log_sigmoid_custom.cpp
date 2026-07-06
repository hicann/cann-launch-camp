#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2;

template<typename TYPE_X, typename TYPE_Y>
class KernelLogSigmoid {
public:
    __aicore__ inline KernelLogSigmoid() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum)
    {
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * coreNum;
        this->tileDataNum = tileDataNum;

        if (coreNum < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (coreNum - tailBlockNum);
        }

        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
        zGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
        pipe.InitBuffer(tmpQueue1, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(tmpQueue2, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(floatQueue, BUFFER_NUM, this->tileDataNum * sizeof(float));
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
        AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> zLocal = outQueueZ.AllocTensor<TYPE_Y>();
        AscendC::LocalTensor<TYPE_X> tmpLocal1 = tmpQueue1.AllocTensor<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> tmpLocal2 = tmpQueue2.AllocTensor<TYPE_X>();

        // 使用预处理器宏判断是否为 bfloat16
        // 注意：DTYPE_X 是框架定义的宏，表示实际类型名称
        // 方法1：使用 sizeof 判断（2字节且不是 half）
        constexpr bool isHalfType = std::is_same<TYPE_X, half>::value;
        constexpr bool isFloatType = std::is_same<TYPE_X, float>::value;
        constexpr bool isTwoByteType = (sizeof(TYPE_X) == 2);
        constexpr bool isBfloat16 = isTwoByteType && !isHalfType && !isFloatType;

        // 调试信息（编译时输出）
        // static_assert(isBfloat16 || isHalfType || isFloatType, "Unknown type");

        if constexpr (isBfloat16) {
            // bfloat16 路径：转换为 float 计算
            AscendC::LocalTensor<float> floatLocal = floatQueue.AllocTensor<float>();
            AscendC::LocalTensor<float> floatTmp1 = floatQueue.AllocTensor<float>();
            AscendC::LocalTensor<float> floatTmp2 = floatQueue.AllocTensor<float>();

            // bfloat16 -> float
            AscendC::Cast(floatLocal, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);

            // 在 float 上计算
            AscendC::Muls(floatTmp1, floatLocal, -1.0f, this->processDataNum);
            AscendC::Exp(floatTmp2, floatTmp1, this->processDataNum);
            AscendC::Adds(floatTmp1, floatTmp2, 1.0f, this->processDataNum);
            AscendC::Log(floatTmp2, floatTmp1, this->processDataNum);
            AscendC::Muls(floatTmp1, floatTmp2, -1.0f, this->processDataNum);

            // float -> bfloat16
            AscendC::Cast(zLocal, floatTmp1, AscendC::RoundMode::CAST_NONE, this->processDataNum);

            floatQueue.FreeTensor(floatTmp2);
            floatQueue.FreeTensor(floatTmp1);
            floatQueue.FreeTensor(floatLocal);
        } else {
            // float32 和 float16 (half)：直接计算
            AscendC::Muls(tmpLocal1, xLocal, (TYPE_X)(-1.0f), this->processDataNum);
            AscendC::Exp(tmpLocal2, tmpLocal1, this->processDataNum);
            AscendC::Adds(tmpLocal1, tmpLocal2, (TYPE_X)1.0f, this->processDataNum);
            AscendC::Log(tmpLocal2, tmpLocal1, this->processDataNum);
            AscendC::Muls(zLocal, tmpLocal2, (TYPE_Y)(-1.0f), this->processDataNum);
        }

        outQueueZ.EnQue<TYPE_Y>(zLocal);
        tmpQueue2.FreeTensor(tmpLocal2);
        tmpQueue1.FreeTensor(tmpLocal1);
        inQueueX.FreeTensor(xLocal);
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
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> tmpQueue1;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> tmpQueue2;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> floatQueue;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> zGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelLogSigmoid<DTYPE_X, DTYPE_Y> op;
    op.Init(x, y, tilingData.smallCoreDataNum, 
            tilingData.bigCoreDataNum, tilingData.finalBigTileNum, 
            tilingData.finalSmallTileNum, tilingData.tileDataNum, 
            tilingData.smallTailDataNum, tilingData.bigTailDataNum, 
            tilingData.tailBlockNum);
    op.Process();
}

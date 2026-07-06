#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"
#include <type_traits>

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t ALIGN_SIZE = 128;   // 改为 128 提高对齐稳定性

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
        uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();

        this->tileDataNum = tileDataNum;

        if (coreNum < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) *
                                 (AscendC::GetBlockIdx() - tailBlockNum);
        }

        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
        zGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
        pipe.InitBuffer(tmpFloatX, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpFloatZ, this->tileDataNum * sizeof(float));
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
        AscendC::DataCopy(xLocal,
                          xGm[progress * this->tileDataNum],
                          this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> zLocal = outQueueZ.AllocTensor<TYPE_Y>();

        AscendC::LocalTensor<float> xFloat = tmpFloatX.Get<float>();
        AscendC::LocalTensor<float> zFloat = tmpFloatZ.Get<float>();

        // 计算对齐后的元素个数（ALIGN_SIZE 的倍数）
        uint32_t alignedNum = ((this->processDataNum + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
        if (alignedNum > this->tileDataNum) alignedNum = this->tileDataNum;

        // 将输入转换为 float 并补齐对齐长度（补0）
        if constexpr (std::is_same<TYPE_X, float>::value) {
            for (uint32_t i = 0; i < this->processDataNum; ++i) {
                xFloat(i) = xLocal(i);
            }
            for (uint32_t i = this->processDataNum; i < alignedNum; ++i) {
                xFloat(i) = 0.0f;
            }
        } else if constexpr (std::is_same<TYPE_X, half>::value) {
            AscendC::Cast(xFloat, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            for (uint32_t i = this->processDataNum; i < alignedNum; ++i) {
                xFloat(i) = 0.0f;
            }
        } else if constexpr (std::is_same<TYPE_X, __bf16>::value) {
            AscendC::Cast(xFloat, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            for (uint32_t i = this->processDataNum; i < alignedNum; ++i) {
                xFloat(i) = 0.0f;
            }
        }

        // LogSigmoid 计算（使用对齐长度）
        ComputeLogSigmoid(zFloat, xFloat, alignedNum);

        // 结果转回输出类型并写入（只写有效部分）
        if constexpr (std::is_same<TYPE_Y, float>::value) {
            for (uint32_t i = 0; i < this->processDataNum; ++i) {
                zLocal(i) = zFloat(i);
            }
        } else if constexpr (std::is_same<TYPE_Y, half>::value) {
            AscendC::Cast(zLocal, zFloat, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        } else if constexpr (std::is_same<TYPE_Y, __bf16>::value) {
            AscendC::Cast(zLocal, zFloat, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        }

        outQueueZ.EnQue<TYPE_Y>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeLogSigmoid(AscendC::LocalTensor<float>& output,
                                             AscendC::LocalTensor<float>& input,
                                             uint32_t alignedNum)
    {
        // LogSigmoid(x) = -log(1 + exp(-x))
        AscendC::Muls(output, input, -1.0f, alignedNum);
        AscendC::Exp(output, output, alignedNum);
        AscendC::Adds(output, output, 1.0f, alignedNum);
        AscendC::Log(output, output, alignedNum);
        AscendC::Muls(output, output, -1.0f, alignedNum);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_Y> zLocal = outQueueZ.DeQue<TYPE_Y>();
        AscendC::DataCopy(zGm[progress * this->tileDataNum],
                          zLocal,
                          this->processDataNum);
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

extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x,
                                                         GM_ADDR y,
                                                         GM_ADDR workspace,
                                                         GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    // 根据 type 动态实例化模板
    if (tilingData.type == 0) {
        KernelLogSigmoid<float, float> op;
        op.Init(x, y,
                tilingData.smallCoreDataNum, tilingData.bigCoreDataNum,
                tilingData.finalBigTileNum, tilingData.finalSmallTileNum,
                tilingData.tileDataNum,
                tilingData.smallTailDataNum, tilingData.bigTailDataNum,
                tilingData.tailBlockNum);
        op.Process();
    } else if (tilingData.type == 1) {
        KernelLogSigmoid<half, half> op;
        op.Init(x, y,
                tilingData.smallCoreDataNum, tilingData.bigCoreDataNum,
                tilingData.finalBigTileNum, tilingData.finalSmallTileNum,
                tilingData.tileDataNum,
                tilingData.smallTailDataNum, tilingData.bigTailDataNum,
                tilingData.tailBlockNum);
        op.Process();
    } else if (tilingData.type == 2) {
        KernelLogSigmoid<__bf16, __bf16> op;
        op.Init(x, y,
                tilingData.smallCoreDataNum, tilingData.bigCoreDataNum,
                tilingData.finalBigTileNum, tilingData.finalSmallTileNum,
                tilingData.tileDataNum,
                tilingData.smallTailDataNum, tilingData.bigTailDataNum,
                tilingData.tailBlockNum);
        op.Process();
    }
}

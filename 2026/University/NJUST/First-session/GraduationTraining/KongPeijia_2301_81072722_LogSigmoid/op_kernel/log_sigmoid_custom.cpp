%%writefile  Sources/test/custom_op/op_kernel/log_sigmoid_custom.cpp
#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"
#include <type_traits>

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
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
        }

        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));

        // LogSigmoid只需要1个临时buffer来存储sigmoid结果
        // 对于bfloat16: 需要2个float buffer（一个用于cast，一个用于sigmoid）
        if constexpr (std::is_same<TYPE_X, bfloat16_t>::value) {
            pipe.InitBuffer(tmpBuf0, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmpBuf1, this->tileDataNum * sizeof(float));
        } else {
            pipe.InitBuffer(tmpBuf0, this->tileDataNum * sizeof(TYPE_X));
        }
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
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();

        // 手动实现 LogSigmoid(x) = log(sigmoid(x))
        // 使用更简单的方法：先计算sigmoid，再计算log
        if constexpr (std::is_same<TYPE_X, bfloat16_t>::value) {
            // bfloat16: 先cast到float，计算后再cast回来
            AscendC::LocalTensor<float> tmpFloatX = tmpBuf0.Get<float>();
            AscendC::LocalTensor<float> tmpFloat0 = tmpBuf1.Get<float>();
            
            // Step 1: Cast bfloat16 to float
            AscendC::Cast(tmpFloatX, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            
            // Step 2: Compute sigmoid(x) = 1/(1+exp(-x))
            AscendC::Sigmoid(tmpFloat0, tmpFloatX, this->processDataNum);
            
            // Step 3: Compute log(sigmoid(x))
            AscendC::Log(tmpFloat0, tmpFloat0, this->processDataNum);
            
            // Step 4: Cast float back to bfloat16
            AscendC::Cast(yLocal, tmpFloat0, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        } else {
            // float/half: 直接计算
            AscendC::LocalTensor<TYPE_X> tmp0 = tmpBuf0.Get<TYPE_X>();
            
            // Step 1: Compute sigmoid(x) = 1/(1+exp(-x))
            AscendC::Sigmoid(tmp0, xLocal, this->processDataNum);
            
            // Step 2: Compute log(sigmoid(x))
            AscendC::Log(yLocal, tmp0, this->processDataNum);
        }

        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;  // 用于bfloat16
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;
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
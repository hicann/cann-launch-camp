#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template<typename TYPE_X, typename TYPE_Y>
class KernelLogSigmoid {
public:
    __aicore__ inline KernelLogSigmoid() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, 
                                uint32_t smallCoreDataNum, uint32_t bigCoreDataNum,
                                uint32_t finalBigTileNum, uint32_t finalSmallTileNum,
                                uint32_t tileDataNum, uint32_t smallTailDataNum,
                                uint32_t bigTailDataNum, uint32_t tailBlockNum)
    {
        uint32_t coreIdx = GetBlockIdx();
        uint32_t globalOffset = 0;

        if (coreIdx < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
            globalOffset = bigCoreDataNum * coreIdx; 
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalOffset = bigCoreDataNum * tailBlockNum + smallCoreDataNum * (coreIdx - tailBlockNum);
        }

        this->tileDataNum = tileDataNum;

        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalOffset, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalOffset, this->coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
        pipe.InitBuffer(tmpBuf, 2 * this->tileDataNum * sizeof(float)); 
    }

    __aicore__ inline void Process()
    {
        this->progress = 0;
        for (int32_t i = 0; i < this->tileNum; i++) {
            this->processDataNum = (i == this->tileNum - 1) ? this->tailDataNum : this->tileDataNum;
            CopyIn();
            Compute();
            CopyOut();
        }
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        DataCopy(xLocal, xGm[this->progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();

        LocalTensor<float> xFloat = tmpBuf.Get<float>();
        LocalTensor<float> yFloat = tmpBuf.Get<float>(this->tileDataNum);

        Cast(xFloat, xLocal, RoundMode::CAST_NONE, this->processDataNum);

        // 使用 AscendC::math 命名空间
        AscendC::math::Neg(xFloat, xFloat, this->processDataNum);
        AscendC::math::Exp(xFloat, xFloat, this->processDataNum);
        AscendC::math::Adds(xFloat, xFloat, 1.0f, this->processDataNum);
        AscendC::math::Log(xFloat, xFloat, this->processDataNum);
        AscendC::math::Neg(xFloat, xFloat, this->processDataNum);

        Cast(yLocal, xFloat, RoundMode::CAST_NONE, this->processDataNum);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut()
    {
        LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
        DataCopy(yGm[this->progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
        this->progress++;
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmpBuf;

    GlobalTensor<TYPE_X> xGm;
    GlobalTensor<TYPE_Y> yGm;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    int32_t progress;
};

// --- 关键修复：确保入口函数有且仅有 4 个参数 ---
// 参数顺序：inputs, outputs, workspace, tiling
// 即使代码里没用到 workspace，也必须写上占位符
extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) 
{
    // 使用 GetTilingData 获取数据指针
    void* tilingDataPtr = GetTilingData(tiling);
    optiling::LogSigmoidCustomTilingData* tilingData = (optiling::LogSigmoidCustomTilingData*)tilingDataPtr;

    KernelLogSigmoid<DTYPE_X, DTYPE_Y> op;
    op.Init(x, y, 
            tilingData->small_core_data_num, 
            tilingData->big_core_data_num,
            tilingData->final_big_tile_num, 
            tilingData->final_small_tile_num,
            tilingData->tile_data_num, 
            tilingData->small_tail_data_num,
            tilingData->big_tail_data_num, 
            tilingData->tail_block_num);
    op.Process();
}

// Kernel侧核函数实现
#include "kernel_operator.h"

#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const FastGeluTilingData &tilingData)
    {
        uint32_t coreIdx = GetBlockIdx();
        uint32_t globalBufferIndex = tilingData.bigCoreDataNum * coreIdx;
        tileDataNum = tilingData.tileDataNum;

        if (coreIdx < tilingData.tailBlockNum) {
            coreDataNum = tilingData.bigCoreDataNum;
            tileNum = tilingData.finalBigTileNum;
            tailDataNum = tilingData.bigTailDataNum;
        } else {
            coreDataNum = tilingData.smallCoreDataNum;
            tileNum = tilingData.finalSmallTileNum;
            tailDataNum = tilingData.smallTailDataNum;
            globalBufferIndex -= (tilingData.bigCoreDataNum - tilingData.smallCoreDataNum) *
                                 (coreIdx - tilingData.tailBlockNum);
        }

        if (coreDataNum == 0) {
            return;
        }

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + globalBufferIndex, coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + globalBufferIndex, coreDataNum);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileDataNum * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileDataNum * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (coreDataNum == 0) {
            return;
        }

        processDataNum = tileDataNum;
        for (uint32_t i = 0; i < tileNum; ++i) {
            if (i == tileNum - 1) {
                processDataNum = tailDataNum;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, xGm[progress * tileDataNum], processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t progress)
    {
        (void)progress;
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        FasterGelu<DT_X, true, false>(yLocal, xLocal, processDataNum);
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        DataCopy(yGm[progress * tileDataNum], yLocal, processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    uint32_t coreDataNum = 0;
    uint32_t tileNum = 0;
    uint32_t tileDataNum = 0;
    uint32_t tailDataNum = 0;
    uint32_t processDataNum = 0;
};

template <typename DT_X>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tiling_data, tiling);
    KernelFastGelu<DT_X> op;
    op.Init(x, y, tiling_data);
    op.Process();
}

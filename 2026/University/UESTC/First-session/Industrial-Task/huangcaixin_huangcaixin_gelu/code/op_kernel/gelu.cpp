#include "kernel_operator.h"
#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

constexpr int32_t BUFFER_NUM = 2;

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

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
        uint32_t globalOffset = bigCoreDataNum * coreIdx;

        if (coreIdx < tailBlockNum) {
            coreDataNum = bigCoreDataNum;
            tileNum = finalBigTileNum;
            tailDataNum = bigTailDataNum;
        } else {
            coreDataNum = smallCoreDataNum;
            tileNum = finalSmallTileNum;
            tailDataNum = smallTailDataNum;
            globalOffset -= (bigCoreDataNum - smallCoreDataNum) * (coreIdx - tailBlockNum);
        }

        this->tileDataNum = tileDataNum;

        xGm.SetGlobalBuffer((__gm__ DT_INPUT_X*)x + globalOffset, coreDataNum);
        zGm.SetGlobalBuffer((__gm__ DT_INPUT_X*)y + globalOffset, coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileDataNum * sizeof(DT_INPUT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileDataNum * sizeof(DT_INPUT_X));
    }

    __aicore__ inline void Process()
    {
        if (tileNum == 0) return;

        // 第一块数据提前搬入，启动流水线
        uint32_t firstTileSize = (tileNum == 1) ? tailDataNum : tileDataNum;
        CopyIn(0, firstTileSize);

        // 流水线处理：当前块计算，同时搬运下一块
        for (uint32_t i = 0; i < tileNum; i++) {
            uint32_t currentSize = (i == tileNum - 1) ? tailDataNum : tileDataNum;

            Compute(currentSize);   // 计算第 i 块
            CopyOut(i, currentSize); // 搬出第 i 块结果

            // 如果不是最后一块，提前搬入下一块
            if (i + 1 < tileNum) {
                uint32_t nextSize = (i + 1 == tileNum - 1) ? tailDataNum : tileDataNum;
                CopyIn(i + 1, nextSize);
            }
        }
    }

private:
    // 常量（类内定义，避免重复构造）
    static constexpr DT_INPUT_X inv_sqrt2 = static_cast<DT_INPUT_X>(0.70710678f);
    static constexpr DT_INPUT_X one       = static_cast<DT_INPUT_X>(1.0f);
    static constexpr DT_INPUT_X half      = static_cast<DT_INPUT_X>(0.5f);

    __aicore__ inline void CopyIn(uint32_t progress, uint32_t currentTileSize)
    {
        auto xLocal = inQueueX.AllocTensor<DT_INPUT_X>();
        AscendC::DataCopy(xLocal, xGm[progress * tileDataNum], currentTileSize);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t currentTileSize)
    {
        auto xLocal = inQueueX.DeQue<DT_INPUT_X>();
        auto zLocal = outQueueZ.AllocTensor<DT_INPUT_X>();

        // GELU(x) = 0.5 * x * (1 + erf(x/√2))
        AscendC::Muls(zLocal, xLocal, inv_sqrt2, currentTileSize);   // x / √2
        AscendC::Erf(zLocal, zLocal, currentTileSize);               // erf
        AscendC::Adds(zLocal, zLocal, one, currentTileSize);         // 1 + erf
        AscendC::Mul(zLocal, zLocal, xLocal, currentTileSize);       // x * (1+erf)
        AscendC::Muls(zLocal, zLocal, half, currentTileSize);        // * 0.5

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t currentTileSize)
    {
        auto zLocal = outQueueZ.DeQue<DT_INPUT_X>();
        AscendC::DataCopy(zGm[progress * tileDataNum], zLocal, currentTileSize);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DT_INPUT_X> xGm;
    AscendC::GlobalTensor<DT_INPUT_X> zGm;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);

    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output,
            tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum,
            tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum,
            tiling_data.tileDataNum,
            tiling_data.smallTailDataNum,
            tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum);
    op.Process();
}
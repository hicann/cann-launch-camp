#include "kernel_operator.h"
#include "gelu_tiling.h"

template<typename TYPE_X, typename TYPE_Y>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output,
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

        if (coreIdx < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
            this->gmOffset = coreIdx * bigCoreDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            this->gmOffset = tailBlockNum * bigCoreDataNum +
                             (coreIdx - tailBlockNum) * smallCoreDataNum;
        }

        xGm.SetGlobalBuffer((__gm__ TYPE_X*)input_x + this->gmOffset, this->coreDataNum);
        zGm.SetGlobalBuffer((__gm__ TYPE_Y*)output + this->gmOffset, this->coreDataNum);

        pipe.InitBuffer(xBuf, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(zBuf, this->tileDataNum * sizeof(TYPE_Y));
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<TYPE_X> xLocal = xBuf.Get<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> zLocal = zBuf.Get<TYPE_Y>();

        for (uint32_t i = 0; i < this->tileNum; i++) {
            uint32_t curProcessNum = this->tileDataNum;
            if (i == this->tileNum - 1) {
                curProcessNum = this->tailDataNum;
            }

            uint32_t offset = i * this->tileDataNum;

            AscendC::DataCopy(xLocal, xGm[offset], curProcessNum);

            
            AscendC::PipeBarrier<PIPE_ALL>();

            
            AscendC::Gelu(zLocal, xLocal, curProcessNum);

            
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::DataCopy(zGm[offset], zLocal, curProcessNum);

            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> zBuf;

    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> zGm;

    uint32_t gmOffset;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
};

extern "C" __global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA(tilingData, tiling);

    if (tilingData.input_dtype_flag == 0) {
        KernelGelu<half, half> op;
        op.Init(input_x, output,
                tilingData.smallCoreDataNum,
                tilingData.bigCoreDataNum,
                tilingData.finalBigTileNum,
                tilingData.finalSmallTileNum,
                tilingData.tileDataNum,
                tilingData.smallTailDataNum,
                tilingData.bigTailDataNum,
                tilingData.tailBlockNum);
        op.Process();
    } else {
        KernelGelu<float, float> op;
        op.Init(input_x, output,
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
}
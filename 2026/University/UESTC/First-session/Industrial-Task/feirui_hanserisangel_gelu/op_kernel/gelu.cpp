#include "kernel_operator.h"
#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

constexpr int32_t BUFFER_NUM = 2;  // 双缓冲（TQue 硬件要求 ≥2）

constexpr float INV_SQRT2 = 0.7071067811865475f;  // 1 / sqrt(2)
constexpr float HALF      = 0.5f;
constexpr float ONE       = 1.0f;
constexpr float THRESHOLD = 6.0f;
constexpr float NEGATIVE_THRESHOLD = -6.0f;

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output,
        uint32_t smallCoreDataNum, uint32_t bigCoreDataNum,
        uint32_t finalBigTileNum,  uint32_t finalSmallTileNum,
        uint32_t tileDataNum,      uint32_t smallTailDataNum,
        uint32_t bigTailDataNum,   uint32_t tailBlockNum)
    {
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * coreIdx;
        this->tileDataNum = tileDataNum;

        if (coreIdx < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum     = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum     = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (coreIdx - tailBlockNum);
        }

        inputGm.SetGlobalBuffer((__gm__ DT_INPUT_X*)input_x + globalBufferIndex, this->coreDataNum);
        outputGm.SetGlobalBuffer((__gm__ DT_INPUT_X*)output + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueX,   BUFFER_NUM, this->tileDataNum * sizeof(DT_INPUT_X));
        pipe.InitBuffer(outQueueY,  BUFFER_NUM, this->tileDataNum * sizeof(DT_INPUT_X));

        pipe.InitBuffer(tmpBuf0, this->tileDataNum * sizeof(DT_INPUT_X));
        pipe.InitBuffer(tmpBuf1, this->tileDataNum * sizeof(DT_INPUT_X));     // f32时4B, f16时2B
        pipe.InitBuffer(maskBuf, this->tileDataNum * sizeof(uint8_t));
    }

    __aicore__ inline void Process() {
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < this->tileNum; i++) {
            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        AscendC::LocalTensor<DT_INPUT_X> xLocal = inQueueX.AllocTensor<DT_INPUT_X>();
        AscendC::DataCopy(xLocal, inputGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        AscendC::LocalTensor<DT_INPUT_X> yLocal = outQueueY.DeQue<DT_INPUT_X>();
        AscendC::DataCopy(outputGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        uint32_t n = this->processDataNum;

        AscendC::LocalTensor<DT_INPUT_X> xLocal = inQueueX.DeQue<DT_INPUT_X>();
        AscendC::LocalTensor<DT_INPUT_X> yLocal = outQueueY.AllocTensor<DT_INPUT_X>();

        AscendC::LocalTensor<DT_INPUT_X> t0 = tmpBuf0.Get<DT_INPUT_X>();
        AscendC::LocalTensor<DT_INPUT_X> t1 = tmpBuf1.Get<DT_INPUT_X>();
        AscendC::LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();

        AscendC::Muls(t0, xLocal, static_cast<DT_INPUT_X>(INV_SQRT2), n);
        AscendC::Erf(t1, t0, n);
        AscendC::Adds(t1, t1, static_cast<DT_INPUT_X>(ONE), n);
        AscendC::Muls(t1, t1, static_cast<DT_INPUT_X>(HALF), n);
        AscendC::Mul(t1, xLocal, t1, n);

        AscendC::Compares(mask, xLocal, static_cast<DT_INPUT_X>(THRESHOLD), AscendC::CMPMODE::GT, n);
        AscendC::Select(t1, mask, xLocal, t1, AscendC::SELMODE::VSEL_CMPMASK_SPR, n);

        AscendC::Compares(mask, xLocal, static_cast<DT_INPUT_X>(NEGATIVE_THRESHOLD), AscendC::CMPMODE::LT, n);
        AscendC::Duplicate(t0, static_cast<DT_INPUT_X>(0.0f), n);
        AscendC::Select(yLocal, mask, t0, t1, AscendC::SELMODE::VSEL_CMPMASK_SPR, n);

        outQueueY.EnQue<DT_INPUT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf;

    AscendC::GlobalTensor<DT_INPUT_X> inputGm;
    AscendC::GlobalTensor<DT_INPUT_X> outputGm;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);

    KernelGelu<DT_INPUT_X> op;
    op.Init(
        input_x, output,
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
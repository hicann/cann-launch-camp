#include "kernel_operator.h"

#include "gelu_tiling.h"

constexpr int32_t BUFFER_NUM = 2;

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, uint32_t totalDataNum,
                                uint32_t smallCoreDataNum, uint32_t bigCoreDataNum, uint32_t tileDataNum,
                                uint32_t tailCoreNum)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = 0;
        this->tileDataNum = tileDataNum;

        if (blockIdx < tailCoreNum) {
            this->coreDataNum = bigCoreDataNum;
            globalBufferIndex = blockIdx * bigCoreDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            globalBufferIndex = tailCoreNum * bigCoreDataNum + (blockIdx - tailCoreNum) * smallCoreDataNum;
        }

        if (globalBufferIndex >= totalDataNum) {
            this->coreDataNum = 0;
        } else if (globalBufferIndex + this->coreDataNum > totalDataNum) {
            this->coreDataNum = totalDataNum - globalBufferIndex;
        }

        this->tileNum = (this->coreDataNum + this->tileDataNum - 1) / this->tileDataNum;
        this->tailDataNum = this->coreDataNum % this->tileDataNum;
        if (this->tailDataNum == 0) {
            this->tailDataNum = this->tileDataNum;
        }

        xGm.SetGlobalBuffer((__gm__ DT_INPUT_X *)input_x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_INPUT_X *)output + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(DT_INPUT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DT_INPUT_X));
        pipe.InitBuffer(tmpBuffer, this->tileDataNum * sizeof(DT_INPUT_X));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->tileNum; ++i) {
            this->processDataNum = this->tileDataNum;
            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<DT_INPUT_X> xLocal = inQueueX.AllocTensor<DT_INPUT_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<DT_INPUT_X> xLocal = inQueueX.DeQue<DT_INPUT_X>();
        AscendC::LocalTensor<DT_INPUT_X> yLocal = outQueueY.AllocTensor<DT_INPUT_X>();
        AscendC::LocalTensor<DT_INPUT_X> tmpLocal = tmpBuffer.Get<DT_INPUT_X>();

        if constexpr (sizeof(DT_INPUT_X) == 2) {
            AscendC::Mul(tmpLocal, xLocal, xLocal, this->processDataNum);
            AscendC::Mul(tmpLocal, tmpLocal, xLocal, this->processDataNum);
            AscendC::Muls(tmpLocal, tmpLocal, static_cast<DT_INPUT_X>(0.0455399241), this->processDataNum);
            AscendC::Add(tmpLocal, tmpLocal, xLocal, this->processDataNum);
            AscendC::Muls(tmpLocal, tmpLocal, static_cast<DT_INPUT_X>(-1.595769122), this->processDataNum);
            AscendC::Exp(tmpLocal, tmpLocal, this->processDataNum);
            AscendC::Adds(tmpLocal, tmpLocal, static_cast<DT_INPUT_X>(1.0), this->processDataNum);
            AscendC::Div(yLocal, xLocal, tmpLocal, this->processDataNum);
        } else {
            AscendC::Muls(tmpLocal, xLocal, static_cast<DT_INPUT_X>(0.7071067811865475244), this->processDataNum);
            AscendC::Erf(tmpLocal, tmpLocal, this->processDataNum);
            AscendC::Adds(tmpLocal, tmpLocal, static_cast<DT_INPUT_X>(1.0), this->processDataNum);
            AscendC::Mul(yLocal, xLocal, tmpLocal, this->processDataNum);
            AscendC::Muls(yLocal, yLocal, static_cast<DT_INPUT_X>(0.5), this->processDataNum);
        }

        outQueueY.EnQue<DT_INPUT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<DT_INPUT_X> yLocal = outQueueY.DeQue<DT_INPUT_X>();
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer;
    AscendC::GlobalTensor<DT_INPUT_X> xGm;
    AscendC::GlobalTensor<DT_INPUT_X> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

extern "C" __global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DTYPE_INPUT_X> op;
    op.Init(input_x, output, tiling_data.totalDataNum, tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum, tiling_data.tileDataNum, tiling_data.tailCoreNum);
    op.Process();
}

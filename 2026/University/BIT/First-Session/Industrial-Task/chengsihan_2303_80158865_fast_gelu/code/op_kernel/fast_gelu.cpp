// Kernel侧核函数实现
#include "kernel_operator.h"

#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

using namespace AscendC;

template <class DT_X>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
        uint32_t totalLength, uint32_t tileNum,
        uint32_t tileLength, uint32_t lastTileLength) {

        uint32_t coreOffset = GetBlockIdx() * totalLength;

        this->x_gm.SetGlobalBuffer((__gm__ DT_X*)x + coreOffset, totalLength);
        this->y_gm.SetGlobalBuffer((__gm__ DT_X*)y + coreOffset, totalLength);

        this->totalLength = totalLength;
        this->tileNum = tileNum;
        this->tileLength = tileLength;
        this->lastTileLength = lastTileLength;

        this->pipe.InitBuffer(this->inQueueX, 2, this->tileLength * sizeof(DT_X));
        this->pipe.InitBuffer(this->outQueueY, 2, this->tileLength * sizeof(DT_X));

        this->pipe.InitBuffer(this->tmpBuf1, this->tileLength * sizeof(DT_X));
        this->pipe.InitBuffer(this->tmpBuf2, this->tileLength * sizeof(DT_X));
        this->pipe.InitBuffer(this->tmpBuf3, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (this->tileNum == 1) {
            uint32_t thisTileLen = this->lastTileLength;
            LocalTensor<DT_X> xLocal = this->inQueueX.template AllocTensor<DT_X>();
            AscendC::DataCopy(xLocal, this->x_gm[0], thisTileLen);
            this->inQueueX.EnQue(xLocal);

            Compute(0);
            CopyOut(0);
            return;
        }

        uint32_t progress = 0;

        LocalTensor<DT_X> xLocal = this->inQueueX.template AllocTensor<DT_X>();
        uint32_t thisTileLen = (progress == this->tileNum - 1) ? this->lastTileLength : this->tileLength;
        AscendC::DataCopy(xLocal, this->x_gm[progress * this->tileLength], thisTileLen);
        this->inQueueX.EnQue(xLocal);
        progress++;

        for (; progress < this->tileNum; progress++) {
            xLocal = this->inQueueX.template AllocTensor<DT_X>();
            thisTileLen = (progress == this->tileNum - 1) ? this->lastTileLength : this->tileLength;
            AscendC::DataCopy(xLocal, this->x_gm[progress * this->tileLength], thisTileLen);
            this->inQueueX.EnQue(xLocal);

            uint32_t computeIdx = progress - 1;
            Compute(computeIdx);
            CopyOut(computeIdx);
        }

        Compute(this->tileNum - 1);
        CopyOut(this->tileNum - 1);
    }

private:
    __aicore__ inline void Compute(uint32_t progress) {
        LocalTensor<DT_X> xLocal = this->inQueueX.template DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = this->outQueueY.template AllocTensor<DT_X>();

        LocalTensor<DT_X> absX = this->tmpBuf1.template Get<DT_X>();
        LocalTensor<DT_X> divDown = this->tmpBuf2.template Get<DT_X>();
        LocalTensor<DT_X> divUp = this->tmpBuf3.template Get<DT_X>();

        uint32_t thisTileLen = (progress == this->tileNum - 1) ? this->lastTileLength : this->tileLength;

        AscendC::Abs(absX, xLocal, thisTileLen);

        AscendC::Muls(divDown, absX, static_cast<DT_X>(-1.702f), thisTileLen);
        AscendC::Exp(divDown, divDown, thisTileLen);
        AscendC::Adds(divDown, divDown, static_cast<DT_X>(1.0f), thisTileLen);

        AscendC::Sub(divUp, xLocal, absX, thisTileLen);
        AscendC::Muls(divUp, divUp, static_cast<DT_X>(0.851f), thisTileLen);
        AscendC::Exp(divUp, divUp, thisTileLen);
        AscendC::Mul(divUp, divUp, xLocal, thisTileLen);

        AscendC::Div(yLocal, divUp, divDown, thisTileLen);

        this->inQueueX.FreeTensor(xLocal);
        this->outQueueY.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress) {
        LocalTensor<DT_X> yLocal = this->outQueueY.template DeQue<DT_X>();

        uint32_t thisTileLen = (progress == this->tileNum - 1) ? this->lastTileLength : this->tileLength;

        AscendC::DataCopy(this->y_gm[progress * this->tileLength], yLocal, thisTileLen);

        this->outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueueX;
    TQue<QuePosition::VECOUT, 2> outQueueY;

    TBuf<QuePosition::VECOUT> tmpBuf1, tmpBuf2, tmpBuf3;

    GlobalTensor<DT_X> x_gm;
    GlobalTensor<DT_X> y_gm;

    uint32_t totalLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;
};

template <typename DT_X>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tiling_data, tiling);

    KernelFastGelu<DT_X> op;
    op.Init(x, y,
        tiling_data.totalLength,
        tiling_data.tileNum,
        tiling_data.tileLength,
        tiling_data.lastTileLength);
    op.Process();
}

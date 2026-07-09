// Kernel侧核函数实现
#include "kernel_operator.h"

#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

constexpr int32_t BUFFER_NUM = 1;

template <class DT_X>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, uint32_t blockLength, uint32_t tileLength) {
        this->length = length;
        this->blockLength = blockLength;
        this->tileLength = tileLength;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t offset = blockIdx * this->blockLength;
        this->coreOffset = offset;
        if (offset >= this->length || this->blockLength == 0) {
            this->coreLength = 0;
        } else {
            uint32_t remainLength = this->length - offset;
            this->coreLength = remainLength < this->blockLength ? remainLength : this->blockLength;
        }

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + this->coreOffset, this->coreLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + this->coreOffset, this->coreLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(tmp, this->tileLength * sizeof(DT_X));

        this->fullTileNum = this->coreLength / this->tileLength;
        this->tailDataNum = this->coreLength - this->fullTileNum * this->tileLength;
        this->fullTileAligned =
            ((this->coreOffset * sizeof(DT_X)) % BLOCK_SIZE == 0) &&
            ((this->tileLength * sizeof(DT_X)) % BLOCK_SIZE == 0);
    }

    __aicore__ inline void Process() {
        this->processDataNum = this->tileLength;
        if (this->fullTileAligned) {
            ProcessFullTilesAligned();
        } else {
            ProcessFullTilesPad();
        }
        ProcessTail();
    }

private:
    __aicore__ inline void ProcessFullTilesAligned() {
        for (uint32_t i = 0, offset = 0; i < this->fullTileNum; i++, offset += this->tileLength) {
            CopyInAligned(offset);
            Compute();
            CopyOutAligned(offset);
        }
    }

    __aicore__ inline void ProcessFullTilesPad() {
        for (uint32_t i = 0, offset = 0; i < this->fullTileNum; i++, offset += this->tileLength) {
            CopyInPad(offset);
            Compute();
            CopyOutPad(offset);
        }
    }

    __aicore__ inline void ProcessTail() {
        if (this->tailDataNum == 0) {
            return;
        }

        uint32_t offset = this->fullTileNum * this->tileLength;
        this->processDataNum = this->tailDataNum;
        uint32_t gmOffsetBytes = (this->coreOffset + offset) * sizeof(DT_X);
        bool tailAligned =
            (gmOffsetBytes % BLOCK_SIZE == 0) &&
            ((this->tailDataNum * sizeof(DT_X)) % BLOCK_SIZE == 0);
        if (tailAligned) {
            CopyInAligned(offset);
            Compute();
            CopyOutAligned(offset);
        } else {
            CopyInPad(offset);
            Compute();
            CopyOutPad(offset);
        }
    }

    __aicore__ inline void CopyInAligned(uint32_t offset) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[offset], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyInPad(uint32_t offset) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(this->processDataNum * sizeof(DT_X)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> scaleLocal = tmp.Get<DT_X>();

        AscendC::Muls(scaleLocal, xLocal, static_cast<DT_X>(1.702f), this->processDataNum);
        AscendC::Sigmoid(yLocal, scaleLocal, this->processDataNum);
        AscendC::Mul(yLocal, yLocal, xLocal, this->processDataNum);
        outQueueY.EnQue<DT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutAligned(uint32_t offset) {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        AscendC::DataCopy(yGm[offset], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutPad(uint32_t offset) {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        AscendC::DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(this->processDataNum * sizeof(DT_X)), 0, 0, 0};
        AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    static constexpr uint32_t BLOCK_SIZE = 32;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t length;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t coreOffset;
    uint32_t coreLength;
    uint32_t fullTileNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    bool fullTileAligned;
};

template <typename DT_X>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tiling_data, tiling);
    KernelFastGelu<DT_X> op;
    op.Init(x, y, tiling_data.length, tiling_data.blockLength, tiling_data.tileLength);
    op.Process();
}

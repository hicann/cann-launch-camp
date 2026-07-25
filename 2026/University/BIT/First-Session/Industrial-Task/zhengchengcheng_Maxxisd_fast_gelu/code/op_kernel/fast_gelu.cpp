// Kernel侧核函数实现
#include "kernel_operator.h"

#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

template <class DT_X>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, uint32_t blockNum) {
        this->totalLength = length;
        this->blockIdx = AscendC::GetBlockIdx();
        this->blockNum = blockNum == 0 ? 1 : blockNum;

        constexpr uint32_t ALIGN_BYTES = 32;
        constexpr uint32_t ALIGN_NUM = ALIGN_BYTES / sizeof(DT_X);
        uint32_t alignedLength = length / ALIGN_NUM * ALIGN_NUM;
        uint32_t base = alignedLength / this->blockNum;
        base = base / ALIGN_NUM * ALIGN_NUM;
        uint32_t extraBlocks = (alignedLength - base * this->blockNum) / ALIGN_NUM;

        this->blockLength = base + (this->blockIdx < extraBlocks ? ALIGN_NUM : 0);
        uint32_t offset = this->blockIdx * base +
            (this->blockIdx < extraBlocks ? this->blockIdx * ALIGN_NUM : extraBlocks * ALIGN_NUM);
        if (this->blockIdx == this->blockNum - 1) {
            this->blockLength += length - alignedLength;
        }

        xGm.SetGlobalBuffer((__gm__ DT_X*)x + offset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + offset, this->blockLength);

        if (this->blockLength <= 512) {
            this->tileLength = 512;
        } else if (this->blockLength <= 1024) {
            this->tileLength = 1024;
        } else if (this->blockLength <= 2048) {
            this->tileLength = 2048;
        } else if (this->totalLength <= 8192) {
            this->tileLength = 2048;
        } else {
            this->tileLength = 4096;
        }
        this->tileNum = (this->blockLength + this->tileLength - 1) / this->tileLength;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        if (this->blockLength == 0) {
            return;
        }
        for (uint32_t i = 0; i < this->tileNum; ++i) {
            uint32_t offset = i * this->tileLength;
            uint32_t calcLength = this->tileLength;
            if (offset + calcLength > this->blockLength) {
                calcLength = this->blockLength - offset;
            }
            CopyIn(offset, calcLength);
            Compute(calcLength);
            CopyOut(offset, calcLength);
        }

    }
private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t calcLength) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[offset], calcLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t calcLength) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> tmp1 = tmpBuf1.Get<DT_X>();

        // FastGelu is algebraically equivalent to x / (1 + exp(-1.702 * x)).
        AscendC::Muls(tmp1, xLocal, (DT_X)-1.702f, calcLength);
        AscendC::Exp(tmp1, tmp1, calcLength);
        AscendC::Adds(tmp1, tmp1, (DT_X)1.0f, calcLength);
        AscendC::Div(yLocal, xLocal, tmp1, calcLength);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t calcLength) {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        AscendC::DataCopy(yGm[offset], yLocal, calcLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    static constexpr uint32_t BUFFER_NUM = 2;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t totalLength;
    uint32_t blockIdx;
    uint32_t blockNum;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t tileNum;

};

template <typename DT_X>
 __global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tiling_data, tiling);
    KernelFastGelu<DT_X> op;
    op.Init(x, y, tiling_data.length, tiling_data.blockNum);
    op.Process();
}

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

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, uint32_t blockLength, uint32_t tileLength) {
        this->length = length;
        this->tileLength = tileLength;
        this->blockOffset = GetBlockIdx() * blockLength;
        if (this->blockOffset >= length) {
            this->blockLength = 0;
        } else {
            uint32_t remain = length - this->blockOffset;
            this->blockLength = remain < blockLength ? remain : blockLength;
        }

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + this->blockOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + this->blockOffset, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(expBuf, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(denBuf, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (this->blockLength == 0 || this->length == 0) {
            return;
        }

        for (uint32_t offset = 0; offset < this->blockLength; offset += this->tileLength) {
            uint32_t calcLength = this->tileLength;
            uint32_t remain = this->blockLength - offset;
            if (remain < calcLength) {
                calcLength = remain;
            }
            CopyIn(offset, calcLength);
            Compute(calcLength);
            CopyOut(offset, calcLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t calcLength) {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(calcLength * sizeof(DT_X)), 0, 0, 0};
        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t calcLength) {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        LocalTensor<DT_X> expLocal = expBuf.Get<DT_X>();
        LocalTensor<DT_X> denLocal = denBuf.Get<DT_X>();

        Muls(expLocal, xLocal, static_cast<DT_X>(-1.702), calcLength);
        Exp(expLocal, expLocal, calcLength);
        Adds(denLocal, expLocal, static_cast<DT_X>(1.0), calcLength);
        Div(yLocal, xLocal, denLocal, calcLength);

        outQueueY.EnQue<DT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t calcLength) {
        LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(calcLength * sizeof(DT_X)), 0, 0, 0};
        DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> expBuf;
    TBuf<QuePosition::VECCALC> denBuf;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    uint32_t length;
    uint32_t blockOffset;
    uint32_t blockLength;
    uint32_t tileLength;
};

template <typename DT_X>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tiling_data, tiling);
    KernelFastGelu<DT_X> op;
    op.Init(x, y, tiling_data.length, tiling_data.blockLength, tiling_data.tileLength);
    op.Process();
}

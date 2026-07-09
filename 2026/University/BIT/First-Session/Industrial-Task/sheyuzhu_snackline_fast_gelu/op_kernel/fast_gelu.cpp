#include "kernel_operator.h"
#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

template <typename DT_X>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const FastGeluTilingData &tilingData)
    {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t coreOffset = 0;

        if (blockIdx < tilingData.formerNum) {
            coreLength_ = tilingData.formerLength;
            coreOffset = blockIdx * tilingData.formerLength;
        } else {
            coreLength_ = tilingData.tailLength;
            coreOffset = tilingData.formerNum * tilingData.formerLength +
                         (blockIdx - tilingData.formerNum) * tilingData.tailLength;
        }

        tileLength_ = (tilingData.tileLength == 0) ? 1 : tilingData.tileLength;
        tileNum_ = (coreLength_ == 0) ? 0 : (coreLength_ + tileLength_ - 1) / tileLength_;
        lastTileLength_ = (tileNum_ == 0) ? 0 : (coreLength_ - (tileNum_ - 1) * tileLength_);

        if (coreLength_ == 0) {
            return;
        }

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x + coreOffset, coreLength_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y + coreOffset, coreLength_);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileLength_ * sizeof(DT_X));

        pipe_.InitBuffer(absBuf_, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(numBuf_, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(denBuf_, tileLength_ * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < tileNum_; i++) {
            uint32_t curLength = (i == tileNum_ - 1) ? lastTileLength_ : tileLength_;
            uint32_t offset = i * tileLength_;
            CopyIn(offset, curLength);
            Compute(curLength);
            CopyOut(offset, curLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t curLength)
    {
        LocalTensor<DT_X> xLocal = inQueueX_.AllocTensor<DT_X>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(DT_X)), 0, 0, 0};
        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t curLength)
    {
        LocalTensor<DT_X> xLocal = inQueueX_.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueY_.AllocTensor<DT_X>();

        LocalTensor<DT_X> absF = absBuf_.Get<DT_X>();
        LocalTensor<DT_X> numF = numBuf_.Get<DT_X>();
        LocalTensor<DT_X> denF = denBuf_.Get<DT_X>();

        Abs(absF, xLocal, curLength);
        Sub(numF, xLocal, absF, curLength);
        Muls(numF, numF, static_cast<DT_X>(0.851f), curLength);
        Exp(numF, numF, curLength);
        Mul(numF, numF, xLocal, curLength);
        Muls(denF, absF, static_cast<DT_X>(-1.702f), curLength);
        Exp(denF, denF, curLength);
        Adds(denF, denF, static_cast<DT_X>(1.0f), curLength);
        Div(yLocal, numF, denF, curLength);

        outQueueY_.EnQue<DT_X>(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t curLength)
    {
        LocalTensor<DT_X> yLocal = outQueueY_.DeQue<DT_X>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(DT_X)), 0, 0, 0};
        DataCopyPad(yGm_[offset], yLocal, copyParams);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    TBuf<QuePosition::VECCALC> absBuf_, numBuf_, denBuf_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;

    uint32_t coreLength_ = 0;
    uint32_t tileLength_ = 0;
    uint32_t tileNum_ = 0;
    uint32_t lastTileLength_ = 0;
};

template <typename DT_X>
 __global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tiling_data, tiling);
    KernelFastGelu<DT_X> op;
    op.Init(x, y, tiling_data);
    op.Process();
}
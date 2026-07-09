#include "kernel_operator.h"

#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

using namespace AscendC;

namespace {
constexpr int32_t kBufferNum = 2;
constexpr float kFastGeluBeta = -1.702f;
constexpr float kOne = 1.0f;
constexpr float kHalf = 0.5f;
}  // namespace

template <class DT_X>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length,
                                uint32_t blockLength, uint32_t tileLength) {
        totalLength_ = length;
        tileLength_ = tileLength;
        const uint32_t blockIdx = GetBlockIdx();
        startOffset_ = blockIdx * blockLength;

        if (startOffset_ >= totalLength_) {
            coreLength_ = 0;
        } else {
            const uint32_t remain = totalLength_ - startOffset_;
            coreLength_ = remain < blockLength ? remain : blockLength;
        }

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x + startOffset_, coreLength_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y + startOffset_, coreLength_);

        pipe_.InitBuffer(inQueueX_, kBufferNum, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(outQueueY_, kBufferNum, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(absBuf_, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(expDenomBuf_, tileLength_ * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (coreLength_ == 0) {
            return;
        }
        if (coreLength_ == tileLength_) {
            CopyInAligned(0, tileLength_);
            Compute(tileLength_);
            CopyOutAligned(0, tileLength_);
            return;
        }

        const uint32_t loopCount = coreLength_ / tileLength_;
        for (uint32_t i = 0; i < loopCount; ++i) {
            const uint32_t offset = i * tileLength_;
            CopyInAligned(offset, tileLength_);
            Compute(tileLength_);
            CopyOutAligned(offset, tileLength_);
        }

        const uint32_t tailOffset = loopCount * tileLength_;
        const uint32_t tailLength = coreLength_ - tailOffset;
        if (tailLength > 0) {
            if (((tailLength * sizeof(DT_X)) & 31) == 0) {
                CopyInAligned(tailOffset, tailLength);
                Compute(tailLength);
                CopyOutAligned(tailOffset, tailLength);
            } else {
                CopyInPad(tailOffset, tailLength);
                Compute(tailLength);
                CopyOutPad(tailOffset, tailLength);
            }
        }
    }

private:
    __aicore__ inline void CopyInAligned(uint32_t offset, uint32_t calcLength) {
        LocalTensor<DT_X> xLocal = inQueueX_.AllocTensor<DT_X>();
        DataCopy(xLocal, xGm_[offset], calcLength);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInPad(uint32_t offset, uint32_t calcLength) {
        LocalTensor<DT_X> xLocal = inQueueX_.AllocTensor<DT_X>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(calcLength * sizeof(DT_X)), 0, 0, 0};
        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t calcLength) {
        LocalTensor<DT_X> xLocal = inQueueX_.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueY_.AllocTensor<DT_X>();
        LocalTensor<DT_X> absLocal = absBuf_.Get<DT_X>();
        LocalTensor<DT_X> expLocal = expDenomBuf_.Get<DT_X>();

        Abs(absLocal, xLocal, calcLength);

        Muls(expLocal, absLocal, static_cast<DT_X>(kFastGeluBeta), calcLength);
        Exp(expLocal, expLocal, calcLength);

        Sub(yLocal, xLocal, absLocal, calcLength);
        Mul(yLocal, yLocal, expLocal, calcLength);
        Add(absLocal, xLocal, absLocal, calcLength);
        Add(yLocal, yLocal, absLocal, calcLength);
        Adds(expLocal, expLocal, static_cast<DT_X>(kOne), calcLength);
        Div(yLocal, yLocal, expLocal, calcLength);
        Muls(yLocal, yLocal, static_cast<DT_X>(kHalf), calcLength);

        outQueueY_.EnQue(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutAligned(uint32_t offset, uint32_t calcLength) {
        LocalTensor<DT_X> yLocal = outQueueY_.DeQue<DT_X>();
        DataCopy(yGm_[offset], yLocal, calcLength);
        outQueueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutPad(uint32_t offset, uint32_t calcLength) {
        LocalTensor<DT_X> yLocal = outQueueY_.DeQue<DT_X>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(calcLength * sizeof(DT_X)), 0, 0, 0};
        DataCopyPad(yGm_[offset], yLocal, copyParams);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, kBufferNum> inQueueX_;
    TQue<QuePosition::VECOUT, kBufferNum> outQueueY_;
    TBuf<QuePosition::VECCALC> absBuf_;
    TBuf<QuePosition::VECCALC> expDenomBuf_;

    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    uint32_t totalLength_ = 0;
    uint32_t startOffset_ = 0;
    uint32_t coreLength_ = 0;
    uint32_t tileLength_ = 0;
};

template <typename DT_X>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tilingData, tiling);

    KernelFastGelu<DT_X> op;
    op.Init(x, y, tilingData.length, tilingData.blockLength, tilingData.tileLength);
    op.Process();
}

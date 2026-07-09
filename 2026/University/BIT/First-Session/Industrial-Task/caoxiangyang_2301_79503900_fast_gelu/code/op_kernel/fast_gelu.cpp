// Ascend C kernel implementation for FastGelu.
#include "kernel_operator.h"

#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

using namespace AscendC;

namespace {
constexpr uint32_t kBufferNum = 1;
constexpr float kNegBeta = -1.702f;
constexpr float kZero = 0.0f;
constexpr float kOne = 1.0f;
}  // namespace

template <class DT_X>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t length, uint32_t tileDataNum, uint32_t blockElements) {
        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        tileDataNum_ = tileDataNum;
        blockElements_ = blockElements;

        uint64_t fullBlockNum = length / blockElements_;
        uint32_t tailElements = static_cast<uint32_t>(length - fullBlockNum * blockElements_);
        uint64_t baseBlockNum = fullBlockNum / blockNum;
        uint64_t extraBlockNum = fullBlockNum - baseBlockNum * blockNum;
        uint64_t coreBlockNum = baseBlockNum + (blockIdx < extraBlockNum ? 1U : 0U);
        uint64_t startBlock = static_cast<uint64_t>(blockIdx) * baseBlockNum +
                              (blockIdx < extraBlockNum ? blockIdx : extraBlockNum);

        startOffset_ = startBlock * blockElements_;
        blockLength_ = coreBlockNum * blockElements_;
        if (blockIdx == blockNum - 1) {
            blockLength_ += tailElements;
        }
        alignedLength_ = (blockLength_ / blockElements_) * blockElements_;

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x + startOffset_, blockLength_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y + startOffset_, blockLength_);

        pipe_.InitBuffer(inQueueX_, kBufferNum, tileDataNum_ * sizeof(DT_X));
        pipe_.InitBuffer(outQueueY_, kBufferNum, tileDataNum_ * sizeof(DT_X));
        pipe_.InitBuffer(expBuf_, tileDataNum_ * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (blockLength_ == 0) {
            return;
        }

        for (uint64_t offset = 0; offset < alignedLength_; offset += tileDataNum_) {
            uint32_t count = static_cast<uint32_t>(
                (alignedLength_ - offset) > tileDataNum_ ? tileDataNum_ : (alignedLength_ - offset));
            CopyInAligned(offset, count);
            Compute(count);
            CopyOutAligned(offset, count);
        }

        if (alignedLength_ < blockLength_) {
            uint32_t tailCount = static_cast<uint32_t>(blockLength_ - alignedLength_);
            CopyInTail(alignedLength_, tailCount);
            Compute(tailCount);
            CopyOutTail(alignedLength_, tailCount);
        }
    }

private:
    __aicore__ inline void CopyInAligned(uint64_t offset, uint32_t count) {
        LocalTensor<DT_X> xLocal = inQueueX_.AllocTensor<DT_X>();
        DataCopy(xLocal, xGm_[offset], count);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInTail(uint64_t offset, uint32_t count) {
        LocalTensor<DT_X> xLocal = inQueueX_.AllocTensor<DT_X>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, static_cast<DT_X>(0)};
        DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count) {
        LocalTensor<DT_X> xLocal = inQueueX_.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueY_.AllocTensor<DT_X>();
        LocalTensor<DT_X> expLocal = expBuf_.Get<DT_X>();

        Abs(expLocal, xLocal, count);
        Muls(expLocal, expLocal, static_cast<DT_X>(kNegBeta), count);
        Exp(expLocal, expLocal, count);

        Mins(yLocal, xLocal, static_cast<DT_X>(kZero), count);
        Mul(yLocal, yLocal, expLocal, count);
        Maxs(xLocal, xLocal, static_cast<DT_X>(kZero), count);
        Add(yLocal, yLocal, xLocal, count);
        Adds(expLocal, expLocal, static_cast<DT_X>(kOne), count);
        Div(yLocal, yLocal, expLocal, count);

        outQueueY_.EnQue(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutAligned(uint64_t offset, uint32_t count) {
        LocalTensor<DT_X> yLocal = outQueueY_.DeQue<DT_X>();
        DataCopy(yGm_[offset], yLocal, count);
        outQueueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutTail(uint64_t offset, uint32_t count) {
        LocalTensor<DT_X> yLocal = outQueueY_.DeQue<DT_X>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
        DataCopyPad(yGm_[offset], yLocal, copyParams);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    TPipe pipe_;
    TQue<QuePosition::VECIN, kBufferNum> inQueueX_;
    TQue<QuePosition::VECOUT, kBufferNum> outQueueY_;
    TBuf<QuePosition::VECCALC> expBuf_;
    uint64_t startOffset_ = 0;
    uint64_t blockLength_ = 0;
    uint64_t alignedLength_ = 0;
    uint32_t tileDataNum_ = 1;
    uint32_t blockElements_ = 1;
};

template <typename DT_X>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    (void)workspace;
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tiling_data, tiling);
    KernelFastGelu<DT_X> op;
    op.Init(x, y, tiling_data.length, tiling_data.tileDataNum, tiling_data.blockElements);
    op.Process();
}

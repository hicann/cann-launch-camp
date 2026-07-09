// Kernel侧核函数实现
#include "kernel_operator.h"

#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

constexpr uint32_t BUFFER_NUM = 2;
constexpr float FAST_GELU_ATTR = 1.702f;
constexpr float SCALAR_NEG_ONE = -1.0f;
constexpr float SCALAR_ONE = 1.0f;

template <class DT_X>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const FastGeluTilingData &tilingData) {
        totalLength_ = tilingData.totalLength;
        alignedBlockLength_ = tilingData.blockLength;
        tileLength_ = tilingData.tileLength;
        coreNum_ = tilingData.coreNum;
        if (coreNum_ == 0) {
            coreNum_ = 1;
        }

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= coreNum_ || totalLength_ == 0) {
            blockLength_ = 0;
        } else {
            blockOffset_ = static_cast<uint64_t>(blockIdx) * alignedBlockLength_;
            if (blockOffset_ >= totalLength_) {
                blockLength_ = 0;
            } else {
                const uint64_t remain = totalLength_ - blockOffset_;
                blockLength_ = remain > alignedBlockLength_ ? alignedBlockLength_ : remain;
            }
        }

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, totalLength_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, totalLength_);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileLength_ * sizeof(DT_X));
        if constexpr (!AscendC::IsSameType<DT_X, float>::value) {
            pipe_.InitBuffer(xFp32Buffer_, tileLength_ * sizeof(float));
            pipe_.InitBuffer(yFp32Buffer_, tileLength_ * sizeof(float));
        }
    }

    __aicore__ inline void Process() {
        if (blockLength_ == 0) {
            return;
        }

        uint64_t processed = 0;
        while (processed < blockLength_) {
            const uint64_t remain = blockLength_ - processed;
            const uint32_t count = remain > tileLength_ ? tileLength_ : static_cast<uint32_t>(remain);
            const uint64_t gmOffset = blockOffset_ + processed;
            CopyIn(gmOffset, count);
            Compute(count);
            CopyOut(gmOffset, count);
            processed += count;
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t gmOffset, uint32_t count) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX_.AllocTensor<DT_X>();
        if (count == tileLength_) {
            AscendC::DataCopy(xLocal, xGm_[gmOffset], count);
        } else {
            const AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            const AscendC::DataCopyPadExtParams<DT_X> padParams{false, 0, 0, static_cast<DT_X>(0)};
            AscendC::DataCopyPad(xLocal, xGm_[gmOffset], copyParams, padParams);
        }
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX_.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY_.AllocTensor<DT_X>();

        if constexpr (AscendC::IsSameType<DT_X, float>::value) {
            ComputeFastGelu(yLocal, xLocal, count);
        } else {
            AscendC::LocalTensor<float> xFp32Local = xFp32Buffer_.Get<float>();
            AscendC::LocalTensor<float> yFp32Local = yFp32Buffer_.Get<float>();
            AscendC::Cast<float, DT_X>(xFp32Local, xLocal, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
            ComputeFastGelu(yFp32Local, xFp32Local, count);
            AscendC::Cast<DT_X, float>(yLocal, yFp32Local, AscendC::RoundMode::CAST_ROUND, count);
            AscendC::PipeBarrier<PIPE_V>();
        }

        outQueueY_.EnQue(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeFastGelu(const AscendC::LocalTensor<float> &yLocal,
        const AscendC::LocalTensor<float> &xLocal, uint32_t count) {
        const int32_t calCount = static_cast<int32_t>(count);

        // y = x / (1 + exp(-1.702 * x)), equivalent to the MindSpore FastGelu formula.
        AscendC::Muls<float>(yLocal, xLocal, SCALAR_NEG_ONE * FAST_GELU_ATTR, calCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp<float>(yLocal, yLocal, calCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds<float>(yLocal, yLocal, SCALAR_ONE, calCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div<float>(yLocal, xLocal, yLocal, calCount);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOut(uint64_t gmOffset, uint32_t count) {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY_.DeQue<DT_X>();
        if (count == tileLength_) {
            AscendC::DataCopy(yGm_[gmOffset], yLocal, count);
        } else {
            const AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPad(yGm_[gmOffset], yLocal, copyParams);
        }
        outQueueY_.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> xFp32Buffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> yFp32Buffer_;
    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> yGm_;
    uint64_t totalLength_{0};
    uint64_t blockOffset_{0};
    uint64_t blockLength_{0};
    uint64_t alignedBlockLength_{0};
    uint32_t tileLength_{1};
    uint32_t coreNum_{1};
};

template <typename DT_X>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    (void)workspace;
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tiling_data, tiling);
    KernelFastGelu<DT_X> op;
    op.Init(x, y, tiling_data);
    op.Process();
}

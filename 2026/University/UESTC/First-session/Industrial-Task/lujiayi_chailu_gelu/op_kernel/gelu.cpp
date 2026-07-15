// Kernel侧GELU实现：float32走exact erf，float16走AscendC高阶Gelu
#include "kernel_operator.h"
#include "gelu_tiling.h"
#include "tiling_key_gelu.h"
#include <type_traits>

constexpr int32_t BUFFER_NUM = 1;

namespace {
template <typename T>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR output, GM_ADDR workspace,
                                uint32_t totalLength, uint32_t tileLength, uint32_t tmpSize) {
        AscendC::SetSysWorkspace(workspace);
        totalLength_ = totalLength;
        tileLength_ = tileLength;
        tmpSize_ = tmpSize == 0 ? 32 : tmpSize;

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t tileCount = (totalLength_ + tileLength_ - 1) / tileLength_;
        const uint32_t tilesPerCore = (tileCount + blockNum - 1) / blockNum;
        const uint32_t firstTile = blockIdx * tilesPerCore;
        startOffset_ = firstTile * tileLength_;
        if (firstTile >= tileCount || startOffset_ >= totalLength_) {
            coreLength_ = 0;
        } else {
            uint32_t remain = totalLength_ - startOffset_;
            uint32_t work = tilesPerCore * tileLength_;
            coreLength_ = remain < work ? remain : work;
        }

        xGm_.SetGlobalBuffer((__gm__ T *)inputX + startOffset_, coreLength_);
        yGm_.SetGlobalBuffer((__gm__ T *)output + startOffset_, coreLength_);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileLength_ * sizeof(T));
        if constexpr (std::is_same_v<T, half>) {
            pipe_.InitBuffer(tmpBuf_, tmpSize_);
        } else {
            pipe_.InitBuffer(tmpBuf_, tileLength_ * sizeof(T));
        }
    }

    __aicore__ inline void Process() {
        if (coreLength_ == 0) {
            return;
        }
        uint32_t loopCount = (coreLength_ + tileLength_ - 1) / tileLength_;
        for (uint32_t i = 0; i < loopCount; ++i) {
            uint32_t offset = i * tileLength_;
            uint32_t validLength = tileLength_;
            if (offset + validLength > coreLength_) {
                validLength = coreLength_ - offset;
            }
            CopyIn(offset, validLength);
            Compute(validLength);
            CopyOut(offset, validLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t validLength) {
        AscendC::LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
        if (validLength == tileLength_) {
            AscendC::DataCopy(xLocal, xGm_[offset], tileLength_);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(validLength * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        }
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t validLength) {
        AscendC::LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY_.AllocTensor<T>();

        if constexpr (std::is_same_v<T, half>) {
            AscendC::LocalTensor<uint8_t> tmpLocal = tmpBuf_.Get<uint8_t>();
            constexpr uint32_t ELEMS_PER_32B = 32 / sizeof(T);
            uint32_t calLength = ((validLength + ELEMS_PER_32B - 1) / ELEMS_PER_32B) * ELEMS_PER_32B;
            if (calLength > tileLength_) {
                calLength = tileLength_;
            }
            AscendC::Gelu<T, false, false>(yLocal, xLocal, tmpLocal, calLength);
        } else {
            AscendC::LocalTensor<T> tmp = tmpBuf_.Get<T>();
            AscendC::Muls(tmp, xLocal, static_cast<T>(0.70710678118654752440f), validLength);
            AscendC::Erf<T, false>(yLocal, tmp, validLength);
            AscendC::Adds(tmp, yLocal, static_cast<T>(1.0f), validLength);
            AscendC::Muls(tmp, tmp, static_cast<T>(0.5f), validLength);
            AscendC::Mul(yLocal, xLocal, tmp, validLength);
        }

        outQueueY_.EnQue<T>(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t validLength) {
        AscendC::LocalTensor<T> yLocal = outQueueY_.DeQue<T>();
        if (validLength == tileLength_) {
            AscendC::DataCopy(yGm_[offset], yLocal, tileLength_);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(validLength * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(yGm_[offset], yLocal, copyParams);
        }
        outQueueY_.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;

    uint32_t totalLength_ = 0;
    uint32_t tileLength_ = 0;
    uint32_t tmpSize_ = 0;
    uint32_t startOffset_ = 0;
    uint32_t coreLength_ = 0;
};
}  // namespace

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tilingData, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, workspace, tilingData.totalLength, tilingData.tileLength, tilingData.tmpSize);
    op.Process();
}

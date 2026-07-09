// FastGelu Kernel 侧实现。
#include "kernel_operator.h"

#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

using namespace AscendC;

namespace {
constexpr uint32_t BUFFER_NUM = 2U;
constexpr float FAST_GELU_COEFFICIENT = -1.702f;
}

template <typename T>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint64_t totalLength,
                                uint64_t blockLength,
                                uint32_t tileLength) {
        const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        coreOffset_ = blockIdx * blockLength;
        tileLength_ = tileLength;

        if (coreOffset_ >= totalLength) {
            coreLength_ = 0U;
            return;
        }

        const uint64_t remaining = totalLength - coreOffset_;
        coreLength_ = remaining < blockLength ? remaining : blockLength;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x) + coreOffset_,
                             static_cast<uint32_t>(coreLength_));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y) + coreOffset_,
                             static_cast<uint32_t>(coreLength_));

        pipe_.InitBuffer(inputQueue_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(outputQueue_, BUFFER_NUM, tileLength_ * sizeof(T));
    }

    __aicore__ inline void Process() {
        if (coreLength_ == 0U) {
            return;
        }

        const uint64_t tileCount =
            (coreLength_ + static_cast<uint64_t>(tileLength_) - 1U) /
            static_cast<uint64_t>(tileLength_);

        for (uint64_t tileIdx = 0U; tileIdx < tileCount; ++tileIdx) {
            const uint64_t offset = tileIdx * static_cast<uint64_t>(tileLength_);
            const uint64_t remaining = coreLength_ - offset;
            const uint32_t validLength = static_cast<uint32_t>(
                remaining < tileLength_ ? remaining : tileLength_);

            CopyIn(offset, validLength);
            Compute(validLength);
            CopyOut(offset, validLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t validLength) {
        LocalTensor<T> xLocal = inputQueue_.AllocTensor<T>();

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1U;
        copyParams.blockLen = validLength * sizeof(T); // 单位：Byte
        copyParams.srcStride = 0U;
        copyParams.dstStride = 0U;
        copyParams.rsv = 0U;

        DataCopyPadExtParams<T> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0U;
        padParams.rightPadding = 0U;
        padParams.paddingValue = static_cast<T>(0);

        DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        inputQueue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t validLength) {
        LocalTensor<T> xLocal = inputQueue_.DeQue<T>();
        LocalTensor<T> yLocal = outputQueue_.AllocTensor<T>();

        // 题目公式可严格化简为：y = x / (1 + exp(-1.702 * x))。
        // 该写法只需 Muls + Exp + Adds + Div，减少中间 Tensor 和指令数。
        Muls(yLocal, xLocal, static_cast<T>(FAST_GELU_COEFFICIENT), validLength);
        PipeBarrier<PIPE_V>();
        Exp(yLocal, yLocal, validLength);
        PipeBarrier<PIPE_V>();
        Adds(yLocal, yLocal, static_cast<T>(1.0f), validLength);
        PipeBarrier<PIPE_V>();
        Div(yLocal, xLocal, yLocal, validLength);

        outputQueue_.EnQue(yLocal);
        inputQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t validLength) {
        LocalTensor<T> yLocal = outputQueue_.DeQue<T>();

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1U;
        copyParams.blockLen = validLength * sizeof(T); // 单位：Byte
        copyParams.srcStride = 0U;
        copyParams.dstStride = 0U;
        copyParams.rsv = 0U;

        // VECOUT -> GM 的 DataCopyPad 会在 UB 侧自动补齐，并只向 GM
        // 写入 blockLen 指定的真实字节数，因此不会覆盖张量尾部之外的内存。
        DataCopyPad(yGm_[offset], yLocal, copyParams);
        outputQueue_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    uint64_t coreOffset_ = 0U;
    uint64_t coreLength_ = 0U;
    uint32_t tileLength_ = 0U;
};

template <typename DT_X>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y,
                                     GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tilingData, tiling);

    KernelFastGelu<DT_X> kernel;
    kernel.Init(x, y,
                tilingData.totalLength,
                tilingData.blockLength,
                tilingData.tileLength);
    kernel.Process();
}

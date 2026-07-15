#include <cstdint>
#include "kernel_operator.h"
#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

namespace {

constexpr int64_t COPY_ALIGN_BYTES = 32;
constexpr float INV_SQRT2 = 0.7071067811865475244f;
constexpr float HALF = 0.5f;
constexpr float ONE  = 1.0f;

} // namespace

template <class T>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, const GeluTilingData* tiling) {
        totalLength_ = tiling->totalLength;
        blockLength_ = tiling->blockLength;
        tileLength_  = tiling->tileLength;

        int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
        int64_t offset = blockIdx * blockLength_;
        int64_t remaining = totalLength_ - offset;
        actualLength_ = (remaining > blockLength_) ? blockLength_ : remaining;
        if (actualLength_ < 0) actualLength_ = 0;

        inputGm_.SetGlobalBuffer((__gm__ T*)input + offset, actualLength_);
        outputGm_.SetGlobalBuffer((__gm__ T*)output + offset, actualLength_);

        pipe_.InitBuffer(inputQueue_, 1, tileLength_ * sizeof(T));
        pipe_.InitBuffer(outputQueue_, 1, tileLength_ * sizeof(T));
        pipe_.InitBuffer(tmpBuf_, tileLength_ * sizeof(T));
    }

    __aicore__ inline void Process() {
        if (actualLength_ <= 0) return;

        int64_t loopCount = (actualLength_ + tileLength_ - 1) / tileLength_;
        for (int64_t i = 0; i < loopCount; ++i) {
            int64_t offset = i * tileLength_;
            int64_t count = (offset + tileLength_ > actualLength_) ? (actualLength_ - offset) : tileLength_;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t count) {
        AscendC::LocalTensor<T> local = inputQueue_.AllocTensor<T>();
        AscendC::DataCopyParams params;
        params.blockCount = 1;
        params.blockLen   = static_cast<uint32_t>(count * sizeof(T));
        params.srcStride  = 0;
        params.dstStride  = 0;
        AscendC::DataCopyPad(local, inputGm_[offset], params, {false, 0, 0, 0});
        inputQueue_.EnQue(local);
    }

    __aicore__ inline void Compute(int64_t count) {
        AscendC::LocalTensor<T> xLocal = inputQueue_.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outputQueue_.AllocTensor<T>();
        AscendC::LocalTensor<T> tmp    = tmpBuf_.Get<T>();

        // GELU = 0.5 * x * (1 + erf(x / sqrt(2)))
        AscendC::Muls(tmp, xLocal, static_cast<T>(INV_SQRT2), static_cast<int32_t>(count));
        AscendC::Erf(tmp, tmp, static_cast<int32_t>(count));
        AscendC::Adds(tmp, tmp, static_cast<T>(ONE), static_cast<int32_t>(count));
        AscendC::Muls(tmp, tmp, static_cast<T>(HALF), static_cast<int32_t>(count));
        AscendC::Mul(yLocal, xLocal, tmp, static_cast<int32_t>(count));

        outputQueue_.EnQue(yLocal);
        inputQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t count) {
        AscendC::LocalTensor<T> local = outputQueue_.DeQue<T>();
        AscendC::DataCopyParams params;
        params.blockCount = 1;
        params.blockLen   = static_cast<uint32_t>(count * sizeof(T));
        params.srcStride  = 0;
        params.dstStride  = 0;
        AscendC::DataCopyPad(outputGm_[offset], local, params);
        outputQueue_.FreeTensor(local);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1>  inputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC>     tmpBuf_;

    AscendC::GlobalTensor<T> inputGm_;
    AscendC::GlobalTensor<T> outputGm_;

    int64_t totalLength_ = 0;
    int64_t blockLength_ = 0;
    int64_t tileLength_  = 0;
    int64_t actualLength_= 0;
};

// ---------- 模板 kernel 定义 ----------
// CANN 会根据 tiling_key_gelu.h 自动实例化 DT_INPUT_X=float 和 DT_INPUT_X=half
template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tilingData, tiling);

    KernelGelu<DT_INPUT_X> op;
    op.Init(input, output, &tilingData);
    op.Process();
}
// 无需显式实例化，CANN 自动处理
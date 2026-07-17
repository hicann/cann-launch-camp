#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t BLOCK_SIZE_BYTE = 32;
constexpr float INV_SQRT2 = 0.70710678118654752440f;

template <typename T>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t coreLength, uint32_t coreOffset,
                                uint32_t maxTileLength) {
        blockLength_ = coreLength;
        maxTileLength_ = (maxTileLength == 0) ? 1 : maxTileLength;
        if (blockLength_ == 0) { tileNum_ = 0; return; }

        tileNum_ = (blockLength_ + maxTileLength_ - 1) / maxTileLength_;
        lastTileLength_ = blockLength_ - (tileNum_ - 1) * maxTileLength_;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + coreOffset, blockLength_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + coreOffset, blockLength_);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, maxTileLength_ * sizeof(T));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, maxTileLength_ * sizeof(T));

        // mask (uint8_t) for Select
        pipe_.InitBuffer(tmpMaskBuf_, maxTileLength_ * sizeof(uint8_t));
        // buffer for -0 constant
        pipe_.InitBuffer(tmpNegZeroBuf_, maxTileLength_ * sizeof(T));

        // float16 needs extra float temporary for high precision
        if constexpr (std::is_same<T, half>::value) {
            pipe_.InitBuffer(tmpFloatBuf_, maxTileLength_ * sizeof(float));
        }
    }

    __aicore__ inline void Process() {
        if (blockLength_ == 0) return;
        for (uint32_t i = 0; i < tileNum_; ++i) {
            uint32_t curLength = (i == tileNum_ - 1) ? lastTileLength_ : maxTileLength_;
            uint32_t offset = i * maxTileLength_;
            CopyIn(offset, curLength);
            Compute(curLength);
            CopyOut(offset, curLength);
        }
    }

private:
    __aicore__ inline bool Is32BAligned(uint32_t elemNum) {
        return ((elemNum * sizeof(T)) % BLOCK_SIZE_BYTE) == 0;
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t curLength) {
        LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
        if (Is32BAligned(curLength)) {
            DataCopy(xLocal, xGm_[offset], curLength);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        }
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t curLength) {
        LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY_.AllocTensor<T>();

        if constexpr (std::is_same<T, float>::value) {
            // ---- GELU(x) = x * 0.5 * (1 + erf(x/√2)) ----
            Muls(yLocal, xLocal, INV_SQRT2, curLength);
            Erf(yLocal, yLocal, curLength);
            Adds(yLocal, yLocal, 1.0f, curLength);
            Muls(yLocal, yLocal, 0.5f, curLength);
            Mul(yLocal, xLocal, yLocal, curLength);

            // Correct -Inf -> -0.0
            FixNegInf(xLocal, yLocal, curLength);
        } else {
            // ---- float16: compute with float precision ----
            LocalTensor<float> fx = tmpFloatBuf_.Get<float>();
            Cast(fx, xLocal, RoundMode::CAST_NONE, curLength);

            Muls(fx, fx, INV_SQRT2, curLength);
            Erf(fx, fx, curLength);
            Adds(fx, fx, 1.0f, curLength);
            Muls(fx, fx, 0.5f, curLength);

            Cast(yLocal, fx, RoundMode::CAST_ROUND, curLength);
            Mul(yLocal, xLocal, yLocal, curLength);

            FixNegInf(xLocal, yLocal, curLength);
        }

        outQueueY_.EnQue<T>(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    // Helper: replace result with -0.0 where x < -1e10 (i.e. -Inf)
    __aicore__ inline void FixNegInf(const LocalTensor<T>& x,
                                     LocalTensor<T>& y,
                                     uint32_t curLength) {
        LocalTensor<T> negZero = tmpNegZeroBuf_.Get<T>();
        Duplicate(negZero, static_cast<T>(0.0f), curLength);
        Muls(negZero, negZero, static_cast<T>(-1.0f), curLength);  // -0.0

        LocalTensor<uint8_t> mask = tmpMaskBuf_.Get<uint8_t>();
        constexpr float NEG_INF_THRESH = -1e10f;
        CompareScalar(mask, x, static_cast<T>(NEG_INF_THRESH), CMPMODE::LT, curLength);

        Select(y, mask, negZero, y, SELMODE::VSEL_TENSOR_TENSOR_MODE, curLength);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t curLength) {
        LocalTensor<T> yLocal = outQueueY_.DeQue<T>();
        if (Is32BAligned(curLength)) {
            DataCopy(yGm_[offset], yLocal, curLength);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[offset], yLocal, copyParams);
        }
        outQueueY_.FreeTensor(yLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    TBuf<QuePosition::VECCALC> tmpMaskBuf_;
    TBuf<QuePosition::VECCALC> tmpNegZeroBuf_;
    TBuf<QuePosition::VECCALC> tmpFloatBuf_;  // used only for half

    GlobalTensor<T> xGm_, yGm_;
    uint32_t blockLength_ = 0, maxTileLength_ = 0;
    uint32_t tileNum_ = 0, lastTileLength_ = 0;
};

extern "C" __global__ __aicore__ void gelu(GM_ADDR x, GM_ADDR y,
                                           GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    uint32_t coreIdx = GetBlockIdx();
    uint32_t coreOff = 0, coreLen = 0;

    if (coreIdx < tilingData.formerNum) {
        coreLen = tilingData.formerLength;
        coreOff = coreIdx * tilingData.formerLength;
    } else if (coreIdx < tilingData.formerNum + tilingData.tailNum) {
        coreLen = tilingData.tailLength;
        coreOff = tilingData.formerNum * tilingData.formerLength;
    } else {
        return;
    }

    if (TILING_KEY_IS(1)) {
        KernelGelu<float> op;
        op.Init(x, y, coreLen, coreOff, tilingData.tileLength);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelGelu<half> op;
        op.Init(x, y, coreLen, coreOff, tilingData.tileLength);
        op.Process();
    }
}
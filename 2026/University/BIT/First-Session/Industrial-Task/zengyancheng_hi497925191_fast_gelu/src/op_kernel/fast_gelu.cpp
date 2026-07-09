// Kernel侧核函数实现
#include "kernel_operator.h"
#include <cstdint>
#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

using namespace AscendC;

template <class T>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength,
                                uint32_t, uint32_t blockLength) {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t offset = blockIdx * blockLength;
        if (offset >= totalLength) { coreLen_ = 0; return; }
        uint32_t remain = totalLength - offset;
        coreLen_ = remain < blockLength ? remain : blockLength;

        xGm_.SetGlobalBuffer((__gm__ T *)x + offset, coreLen_);
        yGm_.SetGlobalBuffer((__gm__ T *)y + offset, coreLen_);

        pipe_.InitBuffer(inQ_, BUFFER_NUM, TILE_LEN * sizeof(T));
        pipe_.InitBuffer(outQ_, BUFFER_NUM, TILE_LEN * sizeof(T));
        // f16: 需要 xF32_ 和 tmp_ 两个 float 缓冲区做 Cast→f32 计算
        // f32: 无需额外缓冲区，直接在 yLocal 中计算
        if constexpr (IsSameType<T, half>::value) {
            pipe_.InitBuffer(xF32_, TILE_LEN * sizeof(float));
            pipe_.InitBuffer(tmp_, TILE_LEN * sizeof(float));
        }
    }

    __aicore__ inline void Process() {
        if (coreLen_ == 0) return;
        const uint32_t tileLen = TILE_LEN;
        const uint32_t totalTiles = (coreLen_ + tileLen - 1) / tileLen;

        // 单tile：无流水线开销
        if (totalTiles == 1) {
            CopyIn(0, coreLen_);
            LocalTensor<T> xLocal = inQ_.DeQue<T>();
            LocalTensor<T> yLocal = outQ_.AllocTensor<T>();
            Compute(xLocal, yLocal, coreLen_);
            outQ_.EnQue<T>(yLocal);
            inQ_.FreeTensor(xLocal);
            LocalTensor<T> yOut = outQ_.DeQue<T>();
            CopyOut(0, coreLen_, yOut);
            return;
        }

        // 多tile：双缓冲流水线，拆分 Full/Pad 消除分支
        CopyInFull(0);
        for (uint32_t i = 0; i < totalTiles; i++) {
            uint32_t off = i * tileLen;
            uint32_t curLen = (i == totalTiles - 1)
                ? static_cast<uint32_t>(coreLen_ - off) : tileLen;

            if (i < totalTiles - 1) {
                uint32_t nextOff = off + tileLen;
                uint32_t nextLen = (nextOff + tileLen > coreLen_)
                    ? static_cast<uint32_t>(coreLen_ - nextOff) : tileLen;
                if (nextLen == tileLen) CopyInFull(nextOff);
                else CopyInPad(nextOff, nextLen);
            }

            LocalTensor<T> xLocal = inQ_.DeQue<T>();
            LocalTensor<T> yLocal = outQ_.AllocTensor<T>();
            Compute(xLocal, yLocal, curLen);
            outQ_.EnQue<T>(yLocal);
            inQ_.FreeTensor(xLocal);

            LocalTensor<T> yOut = outQ_.DeQue<T>();
            if (curLen == tileLen) CopyOutFull(off, yOut);
            else CopyOutPad(off, curLen, yOut);
        }
    }

private:
    // === compile-time helpers ===
    static constexpr uint32_t ALIGN_NUM = 32 / sizeof(T);

    __aicore__ inline uint32_t AlignUp(uint32_t v) const {
        return (v + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
    }

    // === CopyIn variants ===
    __aicore__ inline void CopyInFull(uint32_t offset) {
        LocalTensor<T> xLocal = inQ_.AllocTensor<T>();
        DataCopy(xLocal, xGm_[offset], TILE_LEN);
        inQ_.EnQue(xLocal);
    }
    __aicore__ inline void CopyInPad(uint32_t offset, uint32_t len) {
        LocalTensor<T> xLocal = inQ_.AllocTensor<T>();
        DataCopyExtParams cp{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pp{true, 0,
            static_cast<uint8_t>(AlignUp(len) - len), 0};
        DataCopyPad(xLocal, xGm_[offset], cp, pp);
        inQ_.EnQue(xLocal);
    }
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t len) {
        if (len == TILE_LEN) CopyInFull(offset);
        else CopyInPad(offset, len);
    }

    // === Compute (4-step simplified formula) ===
    __aicore__ inline void Compute(LocalTensor<T> &xLocal,
                                   LocalTensor<T> &yLocal, uint32_t len) {
        if constexpr (IsSameType<T, half>::value) {
            LocalTensor<float> xf = xF32_.Get<float>();
            LocalTensor<float> tmp = tmp_.Get<float>();
            Cast(xf, xLocal, RoundMode::CAST_NONE, len);
            Muls(tmp, xf, -1.702f, len);
            Exp(tmp, tmp, len);
            Adds(tmp, tmp, 1.0f, len);
            Div(xf, xf, tmp, len);
            Cast(yLocal, xf, RoundMode::CAST_RINT, len);
        } else {
            // f32: 直接在 yLocal 中计算，无需临时缓冲区
            Muls(yLocal, xLocal, -1.702f, len);
            Exp(yLocal, yLocal, len);
            Adds(yLocal, yLocal, 1.0f, len);
            Div(yLocal, xLocal, yLocal, len);
        }
    }

    // === CopyOut variants ===
    __aicore__ inline void CopyOutFull(uint32_t offset, LocalTensor<T> &yLocal) {
        DataCopy(yGm_[offset], yLocal, TILE_LEN);
        outQ_.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOutPad(uint32_t offset, uint32_t len,
                                       LocalTensor<T> &yLocal) {
        DataCopyExtParams cp{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[offset], yLocal, cp);
        outQ_.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t len,
                                    LocalTensor<T> &yLocal) {
        if (len == TILE_LEN) CopyOutFull(offset, yLocal);
        else CopyOutPad(offset, len, yLocal);
    }

    // === TILE_LEN: 卡满 192KB UB ===
    // f16: inQ×2(2B) + outQ×2(2B) + xF32(4B) + tmp(4B) = 16B×T → 192KB/16 = 12288
    // f32: inQ×2(4B) + outQ×2(4B)                         = 16B×T → 192KB/16 = 12288
    static constexpr uint32_t TILE_LEN = 12288;
    static constexpr uint32_t BUFFER_NUM = 2;

    TPipe pipe_;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQ_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQ_;
    TBuf<QuePosition::VECCALC> xF32_;
    TBuf<QuePosition::VECCALC> tmp_;
    GlobalTensor<T> xGm_, yGm_;
    uint32_t coreLen_ = 0;
};

template <typename DT_X>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y,
                                      GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tiling_data, tiling);
    KernelFastGelu<DT_X> op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.blockDim,
            tiling_data.blockLength);
    op.Process();
}

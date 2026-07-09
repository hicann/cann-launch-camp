#include "kernel_operator.h"
#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

using namespace AscendC;

template <typename T>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t length, uint32_t blockDim, uint64_t blockLength)
    {
        const uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t offset = coreIdx * blockLength;

        if (length == 0 || coreIdx >= blockDim || offset >= length) {
            coreLength_ = 0;
            return;
        }

        coreLength_ = ((length - offset) < blockLength) ? (length - offset) : blockLength;
        xGm_.SetGlobalBuffer((__gm__ T *)x + offset, coreLength_);
        yGm_.SetGlobalBuffer((__gm__ T *)y + offset, coreLength_);

        pipe_.InitBuffer(inQueue_, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe_.InitBuffer(outQueue_, BUFFER_NUM, TILE_LENGTH * sizeof(T));

        if constexpr (IsSameType<T, half>::value) {
            pipe_.InitBuffer(floatBuf_, TILE_LENGTH * sizeof(float));
        }
        pipe_.InitBuffer(tmpBuf_, TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (coreLength_ == 0) {
            return;
        }

        const uint64_t tileNum = (coreLength_ + TILE_LENGTH - 1) / TILE_LENGTH;

        uint32_t firstLen = static_cast<uint32_t>((coreLength_ < TILE_LENGTH) ? coreLength_ : TILE_LENGTH);
        CopyIn(0, firstLen);

        for (uint64_t i = 0; i < tileNum; ++i) {
            const uint64_t offset = i * TILE_LENGTH;
            const uint32_t curLen = static_cast<uint32_t>(
                (offset + TILE_LENGTH <= coreLength_) ? TILE_LENGTH : (coreLength_ - offset)
            );

            if (i + 1 < tileNum) {
                const uint64_t nextOffset = offset + TILE_LENGTH;
                const uint32_t nextLen = static_cast<uint32_t>(
                    (nextOffset + TILE_LENGTH <= coreLength_) ? TILE_LENGTH : (coreLength_ - nextOffset)
                );
                CopyIn(nextOffset, nextLen);
            }

            LocalTensor<T> xLocal = inQueue_.DeQue<T>();
            LocalTensor<T> yLocal = outQueue_.AllocTensor<T>();

            Compute(xLocal, yLocal, curLen);

            outQueue_.EnQue<T>(yLocal);
            inQueue_.FreeTensor(xLocal);

            LocalTensor<T> yOut = outQueue_.DeQue<T>();
            CopyOut(offset, curLen, yOut);
        }
    }

private:
    __aicore__ inline uint32_t AlignNum() const
    {
        return 32U / sizeof(T);
    }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const
    {
        return (value + align - 1U) / align * align;
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t len)
    {
        LocalTensor<T> xLocal = inQueue_.AllocTensor<T>();

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = len * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        DataCopyPadExtParams<T> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = static_cast<uint8_t>(AlignUp(len, AlignNum()) - len);
        padParams.paddingValue = static_cast<T>(0);

        DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        inQueue_.EnQue<T>(xLocal);
    }

    __aicore__ inline void Compute(LocalTensor<T> &xLocal, LocalTensor<T> &yLocal, uint32_t len)
    {
        LocalTensor<float> tmp = tmpBuf_.Get<float>();

        if constexpr (IsSameType<T, half>::value) {
            LocalTensor<float> xf = floatBuf_.Get<float>();

            Cast(xf, xLocal, RoundMode::CAST_NONE, len);
            Muls(tmp, xf, -1.702f, len);
            Exp(tmp, tmp, len);
            Adds(tmp, tmp, 1.0f, len);
            Div(xf, xf, tmp, len);
            Cast(yLocal, xf, RoundMode::CAST_NONE, len);
        } else {
            Muls(tmp, xLocal, -1.702f, len);
            Exp(tmp, tmp, len);
            Adds(tmp, tmp, 1.0f, len);
            Div(yLocal, xLocal, tmp, len);
        }
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t len, LocalTensor<T> &yLocal)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = len * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        DataCopyPad(yGm_[offset], yLocal, copyParams);
        outQueue_.FreeTensor(yLocal);
    }

private:
    static constexpr uint32_t BUFFER_NUM = 2;
    static constexpr uint32_t TILE_LENGTH = IsSameType<T, half>::value ? 10240 : 6144;

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    TBuf<QuePosition::VECCALC> floatBuf_;
    TBuf<QuePosition::VECCALC> tmpBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    uint64_t coreLength_ = 0;
};

template <typename DT_X>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tilingData, tiling);

    KernelFastGelu<DT_X> op;
    op.Init(x, y, tilingData.length, tilingData.blockDim, tilingData.blockLength);
    op.Process();
}

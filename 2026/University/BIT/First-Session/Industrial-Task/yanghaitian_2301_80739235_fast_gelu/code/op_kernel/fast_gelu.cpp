#include "kernel_operator.h"

#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

namespace {
constexpr int32_t kBufferNum = 2;
constexpr float kFastGeluFullCoef = -1.702f;
}  // namespace

template <class DT_X>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length, uint32_t tileSize, uint32_t blockDim)
    {
        this->length = length;
        this->tileSize = tileSize;
        this->blockDim = blockDim == 0 ? 1 : blockDim;
        this->coreOffset = 0;
        this->coreLength = 0;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x, length);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, length);

        if (length == 0 || tileSize == 0) {
            return;
        }

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t coreSliceLength = AlignUp(CeilDiv(length, this->blockDim), 32 / sizeof(DT_X));
        this->coreOffset = blockIdx * coreSliceLength;
        if (this->coreOffset >= length) {
            return;
        }

        this->coreLength = length - this->coreOffset;
        if (this->coreLength > coreSliceLength) {
            this->coreLength = coreSliceLength;
        }

        uint32_t queueBytes = AlignUp32(tileSize * sizeof(DT_X));
        pipe.InitBuffer(inQueueX, kBufferNum, queueBytes);
        pipe.InitBuffer(outQueueY, kBufferNum, queueBytes);
        InitCalcBuffer(static_cast<DT_X *>(nullptr), tileSize);
    }

    __aicore__ inline void Process()
    {
        if (coreLength == 0) {
            return;
        }

        uint32_t tileCount = CeilDiv(coreLength, tileSize);
        for (uint32_t i = 0; i < tileCount; ++i) {
            uint32_t offset = coreOffset + i * tileSize;
            uint32_t curLength = GetTileLength(i);

            CopyIn(offset, curLength);
            Compute(curLength);
            CopyOut(offset, curLength);
        }
    }

private:
    __aicore__ inline uint32_t AlignUp32(uint32_t value)
    {
        return (value + 31) / 32 * 32;
    }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align)
    {
        return align == 0 ? value : (value + align - 1) / align * align;
    }

    __aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t divisor)
    {
        return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
    }

    __aicore__ inline uint32_t GetTileLength(uint32_t tileIndex)
    {
        uint32_t processed = tileIndex * tileSize;
        uint32_t remain = coreLength - processed;
        return remain < tileSize ? remain : tileSize;
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t curLength)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        uint32_t copyBytes = curLength * static_cast<uint32_t>(sizeof(DT_X));
        if ((copyBytes & 31) == 0) {
            AscendC::DataCopy(xLocal, xGm[offset], curLength);
        } else {
            AscendC::DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> padParams{false, 0, 0, static_cast<DT_X>(0)};
            AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        }
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t curLength)
    {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        uint32_t copyBytes = curLength * static_cast<uint32_t>(sizeof(DT_X));
        if ((copyBytes & 31) == 0) {
            AscendC::DataCopy(yGm[offset], yLocal, curLength);
        } else {
            AscendC::DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
        }
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void Compute(uint32_t curLength)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();

        ComputeImpl(xLocal, yLocal, curLength);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void InitCalcBuffer(float *, uint32_t)
    {
    }

    __aicore__ inline void InitCalcBuffer(half *, uint32_t tileSize)
    {
        uint32_t tmpBytes = AlignUp32(tileSize * sizeof(float));
        pipe.InitBuffer(tmpX, tmpBytes);
        pipe.InitBuffer(tmpDen, tmpBytes);
    }

    __aicore__ inline void ComputeImpl(
        AscendC::LocalTensor<float> xLocal, AscendC::LocalTensor<float> yLocal, uint32_t curLength)
    {
        AscendC::FasterGelu<float, false, false>(yLocal, xLocal, curLength);
    }

    __aicore__ inline void ComputeImpl(
        AscendC::LocalTensor<half> xLocal, AscendC::LocalTensor<half> yLocal, uint32_t curLength)
    {
        AscendC::LocalTensor<float> xFloat = tmpX.Get<float>();
        AscendC::LocalTensor<float> denom = tmpDen.Get<float>();

        AscendC::Cast(xFloat, xLocal, AscendC::RoundMode::CAST_NONE, curLength);
        AscendC::Muls(denom, xFloat, kFastGeluFullCoef, curLength);
        AscendC::Exp(denom, denom, curLength);
        AscendC::Adds(denom, denom, 1.0f, curLength);
        AscendC::Div(xFloat, xFloat, denom, curLength);
        AscendC::Cast(yLocal, xFloat, AscendC::RoundMode::CAST_NONE, curLength);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, kBufferNum> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, kBufferNum> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpX;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpDen;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t length;
    uint32_t tileSize;
    uint32_t blockDim;
    uint32_t coreOffset;
    uint32_t coreLength;
};

template <typename DT_X>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tilingData, tiling);
    KernelFastGelu<DT_X> op;
    op.Init(x, y, tilingData.length, tilingData.tileSize, tilingData.blockDim);
    op.Process();
}

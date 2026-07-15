#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

#include <cstdint>

constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t TILE_LENGTH = 2048;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr float INV_SQRT_TWO = 0.70710678118654752440f;
constexpr float HALF = 0.5f;
constexpr float ONE = 1.0f;

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, uint32_t totalLength,
        uint32_t activeBlockNum, uint32_t coreLength)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t startOffset = blockIdx * coreLength;
        this->blockLength = 0;
        if (blockIdx < activeBlockNum && startOffset < totalLength) {
            this->blockLength = (blockIdx == activeBlockNum - 1) ? (totalLength - startOffset) : coreLength;
        }
        inputGm.SetGlobalBuffer((__gm__ DT_INPUT_X *)input_x + startOffset, this->blockLength);
        outputGm.SetGlobalBuffer((__gm__ DT_INPUT_X *)output + startOffset, this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(DT_INPUT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(DT_INPUT_X));
        pipe.InitBuffer(tmpBufferA, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(tmpBufferB, TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < this->blockLength; offset += TILE_LENGTH) {
            uint32_t curLen = Min(TILE_LENGTH, this->blockLength - offset);
            uint32_t alignedLen = AlignUp(curLen, ElementsPerBlock<DT_INPUT_X>());
            CopyIn(offset, curLen, alignedLen);
            Compute(alignedLen);
            CopyOut(offset, curLen);
        }
    }

private:
    template <class T>
    __aicore__ inline uint32_t ElementsPerBlock()
    {
        return BLOCK_BYTES / sizeof(T);
    }

    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b)
    {
        return a < b ? a : b;
    }

    __aicore__ inline uint32_t AlignUp(uint32_t len, uint32_t align)
    {
        return (len + align - 1) / align * align;
    }

    template <class T>
    __aicore__ inline bool IsBlockAligned(uint32_t len)
    {
        return len == AlignUp(len, ElementsPerBlock<T>());
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t curLen, uint32_t alignedLen)
    {
        AscendC::LocalTensor<DT_INPUT_X> xLocal = inQueueX.AllocTensor<DT_INPUT_X>();
        if (IsBlockAligned<DT_INPUT_X>(curLen)) {
            AscendC::DataCopy(xLocal, inputGm[offset], curLen);
        } else {
            AscendC::DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(curLen * sizeof(DT_INPUT_X)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_INPUT_X> padParams{
                true, 0, static_cast<uint8_t>(alignedLen - curLen), static_cast<DT_INPUT_X>(0)};
            AscendC::DataCopyPad(xLocal, inputGm[offset], copyParams, padParams);
        }
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t alignedLen)
    {
        AscendC::LocalTensor<DT_INPUT_X> xLocal = inQueueX.DeQue<DT_INPUT_X>();
        AscendC::LocalTensor<DT_INPUT_X> yLocal = outQueueY.AllocTensor<DT_INPUT_X>();
        ComputeGelu(yLocal, xLocal, alignedLen);
        outQueueY.EnQue<DT_INPUT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeGelu(AscendC::LocalTensor<half> yLocal,
        AscendC::LocalTensor<half> xLocal, uint32_t alignedLen)
    {
        AscendC::Gelu<half, true, false>(yLocal, xLocal, alignedLen);
    }

    __aicore__ inline void ComputeGelu(AscendC::LocalTensor<float> yLocal,
        AscendC::LocalTensor<float> xLocal, uint32_t alignedLen)
    {
        AscendC::LocalTensor<float> work = tmpBufferA.Get<float>();
        AscendC::LocalTensor<float> prod = tmpBufferB.Get<float>();
        AscendC::Muls(work, xLocal, INV_SQRT_TWO, alignedLen);
        AscendC::Erf(work, work, alignedLen);
        AscendC::Adds(work, work, ONE, alignedLen);
        AscendC::Mul(prod, xLocal, work, alignedLen);
        AscendC::Muls(yLocal, prod, HALF, alignedLen);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t curLen)
    {
        AscendC::LocalTensor<DT_INPUT_X> yLocal = outQueueY.DeQue<DT_INPUT_X>();
        if (IsBlockAligned<DT_INPUT_X>(curLen)) {
            AscendC::DataCopy(outputGm[offset], yLocal, curLen);
        } else {
            AscendC::DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(curLen * sizeof(DT_INPUT_X)), 0, 0, 0};
            AscendC::DataCopyPad(outputGm[offset], yLocal, copyParams);
        }
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufferA;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufferB;
    AscendC::GlobalTensor<DT_INPUT_X> inputGm;
    AscendC::GlobalTensor<DT_INPUT_X> outputGm;
    uint32_t blockLength;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data.totalLength, tiling_data.activeBlockNum, tiling_data.coreLength);
    op.Process();
}

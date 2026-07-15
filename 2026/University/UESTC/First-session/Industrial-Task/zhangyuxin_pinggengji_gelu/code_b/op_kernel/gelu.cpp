#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

#include <cstdint>

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_LENGTH = 2048;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr float INV_SQRT_TWO = 0.70710678118654752440f;
constexpr float HALF = 0.5f;
constexpr float ONE = 1.0f;
constexpr float NEG_ONE = -1.0f;
constexpr float ZERO = 0.0f;
constexpr float SIGN_EPS = 1.0e-20f;
constexpr float ERF_P = 0.3275911f;
constexpr float ERF_A1 = 0.254829592f;
constexpr float ERF_A2 = -0.284496736f;
constexpr float ERF_A3 = 1.421413741f;
constexpr float ERF_A4 = -1.453152027f;
constexpr float ERF_A5 = 1.061405429f;

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
        pipe.InitBuffer(tmpBufferC, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(tmpBufferD, TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength == 0) {
            return;
        }

        uint32_t tileCount = CeilDiv(this->blockLength, TILE_LENGTH);

        uint32_t firstLen = Min(TILE_LENGTH, this->blockLength);
        uint32_t firstAlignedLen = AlignUp(firstLen, ElementsPerBlock<DT_INPUT_X>());
        CopyIn(0, firstLen, firstAlignedLen);

        for (uint32_t tileIdx = 0; tileIdx < tileCount; ++tileIdx) {
            uint32_t offset = tileIdx * TILE_LENGTH;
            uint32_t curLen = Min(TILE_LENGTH, this->blockLength - offset);
            uint32_t alignedLen = AlignUp(curLen, ElementsPerBlock<DT_INPUT_X>());

            if (tileIdx + 1 < tileCount) {
                uint32_t nextOffset = (tileIdx + 1) * TILE_LENGTH;
                uint32_t nextLen = Min(TILE_LENGTH, this->blockLength - nextOffset);
                uint32_t nextAlignedLen = AlignUp(nextLen, ElementsPerBlock<DT_INPUT_X>());
                CopyIn(nextOffset, nextLen, nextAlignedLen);
            }

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

    __aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b)
    {
        return (a + b - 1) / b;
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
        AscendC::LocalTensor<float> absValue = tmpBufferA.Get<float>();
        AscendC::LocalTensor<float> tValue = tmpBufferB.Get<float>();
        AscendC::LocalTensor<float> poly = tmpBufferC.Get<float>();
        AscendC::LocalTensor<float> expValue = tmpBufferD.Get<float>();

        AscendC::Muls(yLocal, xLocal, INV_SQRT_TWO, alignedLen);
        AscendC::Abs(absValue, yLocal, alignedLen);

        AscendC::Muls(tValue, absValue, ERF_P, alignedLen);
        AscendC::Adds(tValue, tValue, ONE, alignedLen);
        AscendC::Muls(poly, absValue, ZERO, alignedLen);
        AscendC::Adds(poly, poly, ONE, alignedLen);
        AscendC::Div(tValue, poly, tValue, alignedLen);

        AscendC::Muls(poly, tValue, ERF_A5, alignedLen);
        AscendC::Adds(poly, poly, ERF_A4, alignedLen);
        AscendC::Mul(poly, poly, tValue, alignedLen);
        AscendC::Adds(poly, poly, ERF_A3, alignedLen);
        AscendC::Mul(poly, poly, tValue, alignedLen);
        AscendC::Adds(poly, poly, ERF_A2, alignedLen);
        AscendC::Mul(poly, poly, tValue, alignedLen);
        AscendC::Adds(poly, poly, ERF_A1, alignedLen);
        AscendC::Mul(poly, poly, tValue, alignedLen);

        AscendC::Mul(expValue, absValue, absValue, alignedLen);
        AscendC::Muls(expValue, expValue, NEG_ONE, alignedLen);
        AscendC::Exp(expValue, expValue, alignedLen);
        AscendC::Mul(poly, poly, expValue, alignedLen);
        AscendC::Muls(poly, poly, NEG_ONE, alignedLen);
        AscendC::Adds(poly, poly, ONE, alignedLen);

        AscendC::Adds(absValue, absValue, SIGN_EPS, alignedLen);
        AscendC::Div(yLocal, yLocal, absValue, alignedLen);
        AscendC::Mul(poly, poly, yLocal, alignedLen);
        AscendC::Adds(poly, poly, ONE, alignedLen);
        AscendC::Mul(yLocal, xLocal, poly, alignedLen);
        AscendC::Muls(yLocal, yLocal, HALF, alignedLen);
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufferC;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufferD;
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

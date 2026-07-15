// Kernel侧核函数实现
#include "kernel_operator.h"
#include <type_traits>

#include "gelu_tiling.h"

#include "tiling_key_gelu.h"

constexpr uint32_t BUFFER_NUM = 1;
constexpr float ERF_PARAM1 = -0.3512339572e-8f;
constexpr float ERF_PARAM2 = 0.2645266170e-6f;
constexpr float ERF_PARAM3 = -0.7929488134e-5f;
constexpr float ERF_PARAM4 = 0.1106123840e-3f;
constexpr float ERF_PARAM5 = 0.6518995814e-4f;
constexpr float ERF_PARAM6 = -0.7266616915e-1f;
constexpr float ERF_PARAM7 = -0.1595769883e1f;
constexpr float ERF_MIN = 5.75f;
constexpr float ERF_MAX = -13.15f;

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, uint32_t length, uint32_t tileLength) {
        this->totalLength = length;
        this->tileLength = tileLength;

        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t avgLength = (length + blockNum - 1) / blockNum;
        uint32_t alignNum = 32 / sizeof(DT_INPUT_X);
        avgLength = (avgLength + alignNum - 1) / alignNum * alignNum;
        uint32_t startOffset = blockIdx * avgLength;

        if (startOffset >= length) {
            this->blockLength = 0;
            return;
        }

        uint32_t remainLength = length - startOffset;
        this->blockLength = remainLength < avgLength ? remainLength : avgLength;

        inputGm.SetGlobalBuffer((__gm__ DT_INPUT_X *)input_x + startOffset, this->blockLength);
        outputGm.SetGlobalBuffer((__gm__ DT_INPUT_X *)output + startOffset, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_INPUT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_INPUT_X));
        pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(DT_INPUT_X));
    }
    __aicore__ inline void Process() {
        if (this->blockLength == 0) {
            return;
        }

        for (uint32_t offset = 0; offset < this->blockLength; offset += this->tileLength) {
            uint32_t calcLength = this->tileLength;
            if (offset + calcLength > this->blockLength) {
                calcLength = this->blockLength - offset;
            }

            CopyIn(offset, calcLength);
            Compute(calcLength);
            CopyOut(offset, calcLength);
        }

    }
private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t calcLength) {
        AscendC::LocalTensor<DT_INPUT_X> xLocal = inQueueX.AllocTensor<DT_INPUT_X>();

        if ((calcLength * sizeof(DT_INPUT_X)) % 32 == 0) {
            AscendC::DataCopy(xLocal, inputGm[offset], calcLength);
        } else {
            AscendC::DataCopyExtParams copyParams{
                1,
                static_cast<uint32_t>(calcLength * sizeof(DT_INPUT_X)),
                0,
                0,
                0
            };
            AscendC::DataCopyPadExtParams<DT_INPUT_X> padParams{
                true,
                0,
                0,
                static_cast<DT_INPUT_X>(0)
            };
            AscendC::DataCopyPad(xLocal, inputGm[offset], copyParams, padParams);
        }
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t calcLength) {
        AscendC::LocalTensor<DT_INPUT_X> xLocal = inQueueX.DeQue<DT_INPUT_X>();
        AscendC::LocalTensor<DT_INPUT_X> yLocal = outQueueY.AllocTensor<DT_INPUT_X>();
        AscendC::LocalTensor<DT_INPUT_X> tmpLocal = tmpBuffer.Get<DT_INPUT_X>();

        if constexpr (std::is_same<DT_INPUT_X, half>::value) {
            AscendC::Mul(yLocal, xLocal, xLocal, calcLength);
            AscendC::Mul(yLocal, yLocal, xLocal, calcLength);
            AscendC::Muls(yLocal, yLocal, static_cast<DT_INPUT_X>(0.044715f), calcLength);
            AscendC::Add(yLocal, yLocal, xLocal, calcLength);
            AscendC::Muls(yLocal, yLocal, static_cast<DT_INPUT_X>(0.7978845608028654f), calcLength);
            AscendC::Tanh(yLocal, yLocal, calcLength);
            AscendC::Adds(yLocal, yLocal, static_cast<DT_INPUT_X>(1.0f), calcLength);
            AscendC::Mul(yLocal, yLocal, xLocal, calcLength);
            AscendC::Muls(yLocal, yLocal, static_cast<DT_INPUT_X>(0.5f), calcLength);
        } else {
            AscendC::Maxs(xLocal, xLocal, ERF_MAX, calcLength);

            AscendC::Mul(tmpLocal, xLocal, xLocal, calcLength);

            AscendC::Muls(yLocal, tmpLocal, ERF_PARAM3, calcLength);
            AscendC::Adds(yLocal, yLocal, ERF_PARAM4, calcLength);

            AscendC::Mul(yLocal, yLocal, tmpLocal, calcLength);
            AscendC::Adds(yLocal, yLocal, ERF_PARAM5, calcLength);

            AscendC::Mul(yLocal, yLocal, tmpLocal, calcLength);
            AscendC::Adds(yLocal, yLocal, ERF_PARAM6, calcLength);

            AscendC::Mul(yLocal, yLocal, tmpLocal, calcLength);
            AscendC::Adds(yLocal, yLocal, ERF_PARAM7, calcLength);

            AscendC::Mul(yLocal, yLocal, xLocal, calcLength);

            AscendC::Exp(yLocal, yLocal, calcLength);
            AscendC::Adds(yLocal, yLocal, static_cast<DT_INPUT_X>(1.0f), calcLength);
            AscendC::Div(yLocal, xLocal, yLocal, calcLength);
        }

        outQueueY.EnQue<DT_INPUT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t calcLength) {
        AscendC::LocalTensor<DT_INPUT_X> yLocal = outQueueY.DeQue<DT_INPUT_X>();

        if ((calcLength * sizeof(DT_INPUT_X)) % 32 == 0) {
            AscendC::DataCopy(outputGm[offset], yLocal, calcLength);
        } else {
            AscendC::DataCopyExtParams copyParams{
                1,
                static_cast<uint32_t>(calcLength * sizeof(DT_INPUT_X)),
                0,
                0,
                0
            };
            AscendC::DataCopyPad(outputGm[offset], yLocal, copyParams);
        }
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer;
    AscendC::GlobalTensor<DT_INPUT_X> inputGm;
    AscendC::GlobalTensor<DT_INPUT_X> outputGm;
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileLength;
};

template <typename DT_INPUT_X>
 __global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data.length, tiling_data.tileLength);
    op.Process();
}

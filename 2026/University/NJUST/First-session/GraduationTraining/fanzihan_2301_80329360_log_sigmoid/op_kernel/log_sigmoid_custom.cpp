#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

#include <type_traits>

constexpr uint32_t kQueueDepth = 1U;

template <typename T>
class LogSigmoidKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR output,
        uint32_t totalElements, uint32_t tileElements)
    {
        elementsPerCore_ = totalElements / AscendC::GetBlockNum();
        tileElements_ = tileElements;

        const uint32_t coreOffset =
            AscendC::GetBlockIdx() * elementsPerCore_;

        inputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(input) + coreOffset,
            elementsPerCore_);

        outputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(output) + coreOffset,
            elementsPerCore_);

        pipe_.InitBuffer(
            inputQueue_, kQueueDepth, tileElements_ * sizeof(T));

        pipe_.InitBuffer(
            outputQueue_, kQueueDepth, tileElements_ * sizeof(T));

        pipe_.InitBuffer(
            floatWorkspace_, tileElements_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t tileCount =
            (elementsPerCore_ + tileElements_ - 1U) / tileElements_;

        for (uint32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
            const uint32_t start = tileIndex * tileElements_;
            const uint32_t remain = elementsPerCore_ - start;

            const uint32_t currentElements =
                (remain < tileElements_) ? remain : tileElements_;

            Load(tileIndex, currentElements);
            Calculate(currentElements);
            Store(tileIndex, currentElements);
        }
    }

private:
    __aicore__ inline void Load(uint32_t tileIndex, uint32_t count)
    {
        AscendC::LocalTensor<T> inputLocal =
            inputQueue_.AllocTensor<T>();

        AscendC::DataCopy(
            inputLocal,
            inputGm_[tileIndex * tileElements_],
            count);

        inputQueue_.EnQue<T>(inputLocal);
    }

    __aicore__ inline void RunFloatFormula(
        const AscendC::LocalTensor<float>& result,
        const AscendC::LocalTensor<float>& input,
        uint32_t count)
    {
        // LogSigmoid(x) = -ln(1 + exp(-x))
        AscendC::Muls(result, input, -1.0f, count);
        AscendC::Exp(result, result, count);
        AscendC::Adds(result, result, 1.0f, count);
        AscendC::Ln(result, result, count);
        AscendC::Muls(result, result, -1.0f, count);
    }

    __aicore__ inline void Calculate(uint32_t count)
    {
        AscendC::LocalTensor<T> inputLocal =
            inputQueue_.DeQue<T>();

        AscendC::LocalTensor<T> outputLocal =
            outputQueue_.AllocTensor<T>();

        if constexpr (std::is_same<T, float>::value) {
            RunFloatFormula(outputLocal, inputLocal, count);
        } else {
            AscendC::LocalTensor<float> floatLocal =
                floatWorkspace_.Get<float>();

            AscendC::Cast(
                floatLocal,
                inputLocal,
                AscendC::RoundMode::CAST_NONE,
                count);

            RunFloatFormula(floatLocal, floatLocal, count);

            if constexpr (std::is_same<T, half>::value) {
                AscendC::Cast(
                    outputLocal,
                    floatLocal,
                    AscendC::RoundMode::CAST_NONE,
                    count);
            } else {
                AscendC::Cast(
                    outputLocal,
                    floatLocal,
                    AscendC::RoundMode::CAST_RINT,
                    count);
            }
        }

        outputQueue_.EnQue<T>(outputLocal);
        inputQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void Store(uint32_t tileIndex, uint32_t count)
    {
        AscendC::LocalTensor<T> outputLocal =
            outputQueue_.DeQue<T>();

        AscendC::DataCopy(
            outputGm_[tileIndex * tileElements_],
            outputLocal,
            count);

        outputQueue_.FreeTensor(outputLocal);
    }

private:
    AscendC::TPipe pipe_;

    AscendC::TQue<AscendC::TPosition::VECIN, kQueueDepth> inputQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, kQueueDepth> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatWorkspace_;

    AscendC::GlobalTensor<T> inputGm_;
    AscendC::GlobalTensor<T> outputGm_;

    uint32_t elementsPerCore_;
    uint32_t tileElements_;
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(1)) {
        LogSigmoidKernel<float> kernel;
        kernel.Init(
            x, y,
            tilingData.elementCount,
            tilingData.tileElements);
        kernel.Process();

    } else if (TILING_KEY_IS(2)) {
        LogSigmoidKernel<half> kernel;
        kernel.Init(
            x, y,
            tilingData.elementCount,
            tilingData.tileElements);
        kernel.Process();

    } else if (TILING_KEY_IS(3)) {
        LogSigmoidKernel<bfloat16_t> kernel;
        kernel.Init(
            x, y,
            tilingData.elementCount,
            tilingData.tileElements);
        kernel.Process();
    }
}

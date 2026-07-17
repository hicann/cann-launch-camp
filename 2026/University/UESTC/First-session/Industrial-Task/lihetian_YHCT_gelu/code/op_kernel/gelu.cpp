#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

#include <type_traits>

constexpr uint32_t BUFFER_NUM = 2;
constexpr float INV_SQRT_2 = 0.7071067811865475f;
constexpr float HALF_VALUE = 0.5f;
constexpr float ONE_VALUE = 1.0f;

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR output, const GeluTilingData &tilingData)
    {
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = tilingData.bigCoreDataNum * coreIdx;

        this->tileLength = tilingData.tileLength;
        this->totalLength = tilingData.totalLength;

        if (coreIdx < tilingData.tailBlockNum) {
            this->coreDataNum = tilingData.bigCoreDataNum;
            this->tileNum = tilingData.finalBigTileNum;
            this->tailDataNum = tilingData.bigTailDataNum;
        } else {
            this->coreDataNum = tilingData.smallCoreDataNum;
            this->tileNum = tilingData.finalSmallTileNum;
            this->tailDataNum = tilingData.smallTailDataNum;
            globalBufferIndex -= (tilingData.bigCoreDataNum - tilingData.smallCoreDataNum) *
                (coreIdx - tilingData.tailBlockNum);
        }

        this->globalBufferIndex = globalBufferIndex;

        inputGm.SetGlobalBuffer((__gm__ DT_INPUT_X *)inputX + globalBufferIndex, this->coreDataNum);
        outputGm.SetGlobalBuffer((__gm__ DT_INPUT_X *)output + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueue, BUFFER_NUM, this->tileLength * sizeof(DT_INPUT_X));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(DT_INPUT_X));

        // 优化点：
        // float32 路径直接用 outputLocal 作为中间变量，不需要 tmpX/tmpErf/tmpScale。
        // 因此 float32 实例不再初始化这三个 float 临时 buffer。
        //
        // float16 路径仍然需要：
        // tmpX: half -> float 后的 x
        // tmpErf: erf 中间结果
        // tmpScale: 0.5 * (1 + erf) 中间结果
        if constexpr (!std::is_same<DT_INPUT_X, float>::value) {
            pipe.InitBuffer(tmpX, this->tileLength * sizeof(float));
            pipe.InitBuffer(tmpErf, this->tileLength * sizeof(float));
            pipe.InitBuffer(tmpScale, this->tileLength * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->tileNum; ++i) {
            uint32_t offset = i * this->tileLength;

            // processDataNum 用于 CopyIn / CopyOut。
            // 保持 32B 对齐后的拷贝长度。
            this->processDataNum = this->tileLength;
            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
            }

            // 当前 tile 起点超过真实输入长度，则跳过。
            if (this->globalBufferIndex + offset >= this->totalLength) {
                continue;
            }

            // validDataNum 用于 Compute。
            // 只对真实有效元素做 GELU。
            uint32_t remain = this->totalLength - this->globalBufferIndex - offset;
            this->validDataNum = remain > this->processDataNum ? this->processDataNum : remain;

            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<DT_INPUT_X> inputLocal = inQueue.AllocTensor<DT_INPUT_X>();

        // 注意：这里必须继续用 processDataNum，保持 32B 对齐拷贝。
        AscendC::DataCopy(inputLocal, inputGm[progress * this->tileLength], this->processDataNum);

        inQueue.EnQue(inputLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<DT_INPUT_X> inputLocal = inQueue.DeQue<DT_INPUT_X>();
        AscendC::LocalTensor<DT_INPUT_X> outputLocal = outQueue.AllocTensor<DT_INPUT_X>();

        if constexpr (std::is_same<DT_INPUT_X, float>::value) {
            // float32 路径：
            // 保持当前最优主线写法。
            // inputLocal 是 float，outputLocal 也是 float。
            // 直接用 outputLocal 作为中间变量，避免 tmpErf/tmpScale 读写。

            // outputLocal = x / sqrt(2)
            AscendC::Muls(outputLocal, inputLocal, INV_SQRT_2, this->validDataNum);

            // outputLocal = erf(x / sqrt(2))
            AscendC::Erf(outputLocal, outputLocal, this->validDataNum);

            // outputLocal = 1 + erf(x / sqrt(2))
            AscendC::Adds(outputLocal, outputLocal, ONE_VALUE, this->validDataNum);

            // outputLocal = 0.5 * (1 + erf(x / sqrt(2)))
            AscendC::Muls(outputLocal, outputLocal, HALF_VALUE, this->validDataNum);

            // outputLocal = x * 0.5 * (1 + erf(x / sqrt(2)))
            AscendC::Mul(outputLocal, inputLocal, outputLocal, this->validDataNum);
        } else {
            // float16 路径：
            // 保持稳定三临时 buffer 写法。
            // half 先转 float 计算，再 cast 回 half，保证精度。
            AscendC::LocalTensor<float> xFloat = tmpX.Get<float>();
            AscendC::LocalTensor<float> erfFloat = tmpErf.Get<float>();
            AscendC::LocalTensor<float> scaleFloat = tmpScale.Get<float>();

            // xFloat = float(inputLocal)
            AscendC::Cast(xFloat, inputLocal, AscendC::RoundMode::CAST_NONE, this->validDataNum);

            // erfFloat = x / sqrt(2)
            AscendC::Muls(erfFloat, xFloat, INV_SQRT_2, this->validDataNum);

            // erfFloat = erf(x / sqrt(2))
            AscendC::Erf(erfFloat, erfFloat, this->validDataNum);

            // scaleFloat = 1 + erf(x / sqrt(2))
            AscendC::Adds(scaleFloat, erfFloat, ONE_VALUE, this->validDataNum);

            // scaleFloat = 0.5 * (1 + erf(x / sqrt(2)))
            AscendC::Muls(scaleFloat, scaleFloat, HALF_VALUE, this->validDataNum);

            // xFloat = x * 0.5 * (1 + erf(x / sqrt(2)))
            AscendC::Mul(xFloat, xFloat, scaleFloat, this->validDataNum);

            // outputLocal = half(xFloat)
            AscendC::Cast(outputLocal, xFloat, AscendC::RoundMode::CAST_NONE, this->validDataNum);
        }

        outQueue.EnQue(outputLocal);
        inQueue.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<DT_INPUT_X> outputLocal = outQueue.DeQue<DT_INPUT_X>();

        // 注意：这里也必须继续用 processDataNum，保持 32B 对齐写回。
        AscendC::DataCopy(outputGm[progress * this->tileLength], outputLocal, this->processDataNum);

        outQueue.FreeTensor(outputLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpX;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpErf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpScale;

    AscendC::GlobalTensor<DT_INPUT_X> inputGm;
    AscendC::GlobalTensor<DT_INPUT_X> outputGm;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t tailDataNum;

    // CopyIn / CopyOut 使用，保持 32B 对齐后的元素数量。
    uint32_t processDataNum;

    // Compute 使用，只处理真实有效元素。
    uint32_t validDataNum;

    // 全局真实输入元素数量。
    uint32_t totalLength;

    // 当前 core 对应的全局起始元素位置。
    uint32_t globalBufferIndex;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tilingData, tiling);

    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tilingData);
    op.Process();
}
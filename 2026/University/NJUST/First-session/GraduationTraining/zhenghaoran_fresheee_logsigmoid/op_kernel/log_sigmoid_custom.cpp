#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    // 请完成Kernel侧代码实现
}

%%writefile Sources/test/custom_op/op_host/log_sigmoid_custom.cpp
#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    LogSigmoidCustomTilingData *tiling = context->GetTilingData<LogSigmoidCustomTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);

    // 1. 计算输入总元素数
    uint32_t totalSize = 1;
    int dimNum = x1_shape->GetStorageShape().GetDimNum();
    for (int i = 0; i < dimNum; ++i) {
        totalSize *= x1_shape->GetStorageShape().GetDim(i);
    }
    tiling->totalSize = totalSize;

    // 2. 直接链式调用获取数据类型，不声明 TensorDesc 变量，避开类型名问题
    ge::DataType inputType = context->GetInputDesc(0)->GetDataType();
    if (inputType == ge::DT_FLOAT) {
        tiling->dataType = 0;
    } else if (inputType == ge::DT_FLOAT16) {
        tiling->dataType = 1;
    } else if (inputType == ge::DT_BF16) {
        tiling->dataType = 2;
    } else {
        return ge::GRAPH_FAILED;
    }

    // 3. 配置单 Block 处理规模与总 Block 数
    const uint32_t blockSize = 4096;
    tiling->blockSize = blockSize;
    uint32_t blockDim = (totalSize + blockSize - 1) / blockSize;
    blockDim = blockDim > 0 ? blockDim : 1;
    context->SetBlockDim(blockDim);

    // 4. 无需额外工作空间
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class LogSigmoidCustom : public OpDef {
public:
    explicit LogSigmoidCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(LogSigmoidCustom);
}


11

%%writefile  Sources/test/custom_op/op_kernel/log_sigmoid_custom_tiling.h
#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H
#include <cstdint>

struct LogSigmoidCustomTilingData {
    uint32_t totalSize;   // 输入总元素个数
    uint32_t dataType;    // 数据类型：0-float32, 1-float16, 2-bfloat16
    uint32_t blockSize;   // 单个 AI Core 处理的元素数
};

#endif // LOG_SIGMOID_CUSTOM_TILING_H

11

%%writefile  Sources/test/custom_op/op_kernel/log_sigmoid_custom.cpp
#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

using namespace AscendC;

constexpr uint32_t CHUNK_SIZE = 256;

// 指针地址转32位偏移的辅助宏，显式截断避免编译报错
#define PTR_TO_U32(ptr) static_cast<uint32_t>(reinterpret_cast<unsigned long long>(ptr))

// 数值稳定的LogSigmoid计算，所有张量外部传入，避免内部构造触发类型限制
__aicore__ inline void ComputeLogSigmoid(
    LocalTensor<float>& yOut,
    LocalTensor<float>& absTmp,
    LocalTensor<float>& negAbsTmp,
    LocalTensor<float>& expTmp,
    LocalTensor<float>& onePlusExpTmp,
    LocalTensor<float>& logTmp,
    LocalTensor<float>& zeroTmp,
    LocalTensor<float>& minTmp,
    const LocalTensor<float>& xIn,
    uint32_t count)
{
    Abs(absTmp, xIn, count);
    Muls(negAbsTmp, absTmp, -1.0f, count);
    Exp(expTmp, negAbsTmp, count);
    Adds(onePlusExpTmp, expTmp, 1.0f, count);
    Log(logTmp, onePlusExpTmp, count);
    Muls(zeroTmp, xIn, 0.0f, count);
    Min(minTmp, xIn, zeroTmp, count);
    Sub(yOut, minTmp, logTmp, count);
}

extern "C" __global__ __aicore__ void log_sigmoid_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    uint32_t blockId = get_block_idx();
    uint32_t totalSize = tilingData.totalSize;
    uint32_t blockSize = tilingData.blockSize;
    uint32_t dataType = tilingData.dataType;

    uint32_t start = blockId * blockSize;
    if (start >= totalSize) {
        return;
    }
    uint32_t end = (start + blockSize > totalSize) ? totalSize : start + blockSize;
    uint32_t elemCount = end - start;

    if (dataType == 0) {
        // ========== float32 直接计算 ==========
        float xBuf[CHUNK_SIZE];
        float yBuf[CHUNK_SIZE];
        float absBuf[CHUNK_SIZE];
        float negAbsBuf[CHUNK_SIZE];
        float expBuf[CHUNK_SIZE];
        float onePlusExpBuf[CHUNK_SIZE];
        float logBuf[CHUNK_SIZE];
        float zeroBuf[CHUNK_SIZE];
        float minBuf[CHUNK_SIZE];

        LocalTensor<float> xLocal(TPosition::VECCALC, PTR_TO_U32(xBuf), CHUNK_SIZE);
        LocalTensor<float> yLocal(TPosition::VECCALC, PTR_TO_U32(yBuf), CHUNK_SIZE);
        LocalTensor<float> absLocal(TPosition::VECCALC, PTR_TO_U32(absBuf), CHUNK_SIZE);
        LocalTensor<float> negAbsLocal(TPosition::VECCALC, PTR_TO_U32(negAbsBuf), CHUNK_SIZE);
        LocalTensor<float> expLocal(TPosition::VECCALC, PTR_TO_U32(expBuf), CHUNK_SIZE);
        LocalTensor<float> onePlusExpLocal(TPosition::VECCALC, PTR_TO_U32(onePlusExpBuf), CHUNK_SIZE);
        LocalTensor<float> logLocal(TPosition::VECCALC, PTR_TO_U32(logBuf), CHUNK_SIZE);
        LocalTensor<float> zeroLocal(TPosition::VECCALC, PTR_TO_U32(zeroBuf), CHUNK_SIZE);
        LocalTensor<float> minLocal(TPosition::VECCALC, PTR_TO_U32(minBuf), CHUNK_SIZE);

        for (uint32_t i = 0; i < elemCount; i += CHUNK_SIZE) {
            uint32_t curSize = (i + CHUNK_SIZE > elemCount) ? elemCount - i : CHUNK_SIZE;
            GM_ADDR xCur = x + (start + i) * sizeof(float);
            GM_ADDR yCur = y + (start + i) * sizeof(float);

            GlobalTensor<float> xGm;
            GlobalTensor<float> yGm;
            xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(xCur), curSize);
            yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(yCur), curSize);

            DataCopy(xLocal, xGm, curSize);
            ComputeLogSigmoid(yLocal, absLocal, negAbsLocal, expLocal, onePlusExpLocal, logLocal, zeroLocal, minLocal, xLocal, curSize);
            DataCopy(yGm, yLocal, curSize);
        }
    } else if (dataType == 1) {
        // ========== float16：转float32计算再转回 ==========
        half xHalfBuf[CHUNK_SIZE];
        half yHalfBuf[CHUNK_SIZE];
        float xFloatBuf[CHUNK_SIZE];
        float yFloatBuf[CHUNK_SIZE];
        float absBuf[CHUNK_SIZE];
        float negAbsBuf[CHUNK_SIZE];
        float expBuf[CHUNK_SIZE];
        float onePlusExpBuf[CHUNK_SIZE];
        float logBuf[CHUNK_SIZE];
        float zeroBuf[CHUNK_SIZE];
        float minBuf[CHUNK_SIZE];

        LocalTensor<half> xHalfLocal(TPosition::VECCALC, PTR_TO_U32(xHalfBuf), CHUNK_SIZE);
        LocalTensor<half> yHalfLocal(TPosition::VECCALC, PTR_TO_U32(yHalfBuf), CHUNK_SIZE);
        LocalTensor<float> xFloatLocal(TPosition::VECCALC, PTR_TO_U32(xFloatBuf), CHUNK_SIZE);
        LocalTensor<float> yFloatLocal(TPosition::VECCALC, PTR_TO_U32(yFloatBuf), CHUNK_SIZE);
        LocalTensor<float> absLocal(TPosition::VECCALC, PTR_TO_U32(absBuf), CHUNK_SIZE);
        LocalTensor<float> negAbsLocal(TPosition::VECCALC, PTR_TO_U32(negAbsBuf), CHUNK_SIZE);
        LocalTensor<float> expLocal(TPosition::VECCALC, PTR_TO_U32(expBuf), CHUNK_SIZE);
        LocalTensor<float> onePlusExpLocal(TPosition::VECCALC, PTR_TO_U32(onePlusExpBuf), CHUNK_SIZE);
        LocalTensor<float> logLocal(TPosition::VECCALC, PTR_TO_U32(logBuf), CHUNK_SIZE);
        LocalTensor<float> zeroLocal(TPosition::VECCALC, PTR_TO_U32(zeroBuf), CHUNK_SIZE);
        LocalTensor<float> minLocal(TPosition::VECCALC, PTR_TO_U32(minBuf), CHUNK_SIZE);

        for (uint32_t i = 0; i < elemCount; i += CHUNK_SIZE) {
            uint32_t curSize = (i + CHUNK_SIZE > elemCount) ? elemCount - i : CHUNK_SIZE;
            GM_ADDR xCur = x + (start + i) * sizeof(half);
            GM_ADDR yCur = y + (start + i) * sizeof(half);

            GlobalTensor<half> xGm;
            GlobalTensor<half> yGm;
            xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(xCur), curSize);
            yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(yCur), curSize);

            DataCopy(xHalfLocal, xGm, curSize);
            // half -> float32：拓宽转换，无精度损失，使用CAST_NONE
            Cast(xFloatLocal, xHalfLocal, AscendC::RoundMode::CAST_NONE, curSize);

            ComputeLogSigmoid(yFloatLocal, absLocal, negAbsLocal, expLocal, onePlusExpLocal, logLocal, zeroLocal, minLocal, xFloatLocal, curSize);

            // float32 -> half：窄化转换，使用Von Neumann舍入CAST_ODD
            Cast(yHalfLocal, yFloatLocal, AscendC::RoundMode::CAST_ODD, curSize);
            DataCopy(yGm, yHalfLocal, curSize);
        }
    } else if (dataType == 2) {
        // ========== bfloat16：转float32计算再转回 ==========
        bfloat16_t xBf16Buf[CHUNK_SIZE];
        bfloat16_t yBf16Buf[CHUNK_SIZE];
        float xFloatBuf[CHUNK_SIZE];
        float yFloatBuf[CHUNK_SIZE];
        float absBuf[CHUNK_SIZE];
        float negAbsBuf[CHUNK_SIZE];
        float expBuf[CHUNK_SIZE];
        float onePlusExpBuf[CHUNK_SIZE];
        float logBuf[CHUNK_SIZE];
        float zeroBuf[CHUNK_SIZE];
        float minBuf[CHUNK_SIZE];

        LocalTensor<bfloat16_t> xBf16Local(TPosition::VECCALC, PTR_TO_U32(xBf16Buf), CHUNK_SIZE);
        LocalTensor<bfloat16_t> yBf16Local(TPosition::VECCALC, PTR_TO_U32(yBf16Buf), CHUNK_SIZE);
        LocalTensor<float> xFloatLocal(TPosition::VECCALC, PTR_TO_U32(xFloatBuf), CHUNK_SIZE);
        LocalTensor<float> yFloatLocal(TPosition::VECCALC, PTR_TO_U32(yFloatBuf), CHUNK_SIZE);
        LocalTensor<float> absLocal(TPosition::VECCALC, PTR_TO_U32(absBuf), CHUNK_SIZE);
        LocalTensor<float> negAbsLocal(TPosition::VECCALC, PTR_TO_U32(negAbsBuf), CHUNK_SIZE);
        LocalTensor<float> expLocal(TPosition::VECCALC, PTR_TO_U32(expBuf), CHUNK_SIZE);
        LocalTensor<float> onePlusExpLocal(TPosition::VECCALC, PTR_TO_U32(onePlusExpBuf), CHUNK_SIZE);
        LocalTensor<float> logLocal(TPosition::VECCALC, PTR_TO_U32(logBuf), CHUNK_SIZE);
        LocalTensor<float> zeroLocal(TPosition::VECCALC, PTR_TO_U32(zeroBuf), CHUNK_SIZE);
        LocalTensor<float> minLocal(TPosition::VECCALC, PTR_TO_U32(minBuf), CHUNK_SIZE);

        for (uint32_t i = 0; i < elemCount; i += CHUNK_SIZE) {
            uint32_t curSize = (i + CHUNK_SIZE > elemCount) ? elemCount - i : CHUNK_SIZE;
            GM_ADDR xCur = x + (start + i) * sizeof(bfloat16_t);
            GM_ADDR yCur = y + (start + i) * sizeof(bfloat16_t);

            GlobalTensor<bfloat16_t> xGm;
            GlobalTensor<bfloat16_t> yGm;
            xGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(xCur), curSize);
            yGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(yCur), curSize);

            DataCopy(xBf16Local, xGm, curSize);
            // bf16 -> float32：拓宽转换
            Cast(xFloatLocal, xBf16Local, AscendC::RoundMode::CAST_NONE, curSize);

            ComputeLogSigmoid(yFloatLocal, absLocal, negAbsLocal, expLocal, onePlusExpLocal, logLocal, zeroLocal, minLocal, xFloatLocal, curSize);

            // float32 -> bf16：截断转换
            Cast(yBf16Local, yFloatLocal, AscendC::RoundMode::CAST_NONE, curSize);
            DataCopy(yGm, yBf16Local, curSize);
        }
    }
}

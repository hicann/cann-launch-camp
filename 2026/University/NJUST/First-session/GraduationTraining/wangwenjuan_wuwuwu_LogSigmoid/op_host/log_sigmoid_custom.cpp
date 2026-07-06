
#include <algorithm>
#include <cstdint>

#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t BLOCK_SIZE = 32;

static uint32_t CeilDiv(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

static uint32_t AlignUp(uint32_t a, uint32_t b) {
    return CeilDiv(a, b) * b;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t maxCoreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNum());

    // 获取输入数据总量（使用uint32_t，与之前BF16实现一致）
    uint32_t inputNum = static_cast<uint32_t>(context->GetInputShape(0)->GetStorageShape().GetShapeSize());
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);

    // 按 32 字节对齐
    uint32_t alignNum = BLOCK_SIZE / typeLength;
    if (alignNum == 0) {
        alignNum = 1;
    }
    uint32_t inputNumAlign = AlignUp(inputNum, alignNum);
    uint32_t totalBlockNum = inputNumAlign / alignNum;

    // 核数分配
    uint32_t coreNum = std::min(maxCoreNum, totalBlockNum);
    coreNum = std::max(coreNum, static_cast<uint32_t>(1));
    context->SetBlockDim(coreNum);

    // 获取 UB 大小
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // 每个元素需占用：输入 + 输出 + 3 个 float 临时缓冲区
    uint32_t perElementBytes = typeLength + typeLength + 3 * sizeof(float);
    uint32_t tileDataNum = static_cast<uint32_t>(ubSize / perElementBytes);

    // 对齐到 alignNum
    if (tileDataNum < alignNum) {
        tileDataNum = alignNum;
    }
    tileDataNum = (tileDataNum / alignNum) * alignNum;

    // 每个核负责的 block 数
    uint32_t everyCoreBlockNum = totalBlockNum / coreNum;
    uint32_t tailBlockNum = totalBlockNum % coreNum;
    uint32_t smallCoreDataNum = everyCoreBlockNum * alignNum;
    uint32_t bigCoreDataNum = (everyCoreBlockNum + 1) * alignNum;

    // 计算每个核的 tile 数及尾块大小
    uint32_t finalSmallTileNum = CeilDiv(smallCoreDataNum, tileDataNum);
    uint32_t finalBigTileNum = CeilDiv(bigCoreDataNum, tileDataNum);
    uint32_t smallTailDataNum = smallCoreDataNum - (finalSmallTileNum - 1) * tileDataNum;
    uint32_t bigTailDataNum = bigCoreDataNum - (finalBigTileNum - 1) * tileDataNum;

    // 安全检查
    if (smallTailDataNum == 0) smallTailDataNum = tileDataNum;
    if (bigTailDataNum == 0) bigTailDataNum = tileDataNum;

    // 填充 Tiling 结构体（参考之前BF16实现，不包含size成员）
    LogSigmoidCustomTilingData* tiling = context->GetTilingData<LogSigmoidCustomTilingData>();
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->tailBlockNum = tailBlockNum;

    // 不需要额外 workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class LogSigmoidCustom : public OpDef {
public:
    explicit LogSigmoidCustom(const char* name) : OpDef(name) {
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

        this->AICore()
            .SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(LogSigmoidCustom);
} // namespace ops

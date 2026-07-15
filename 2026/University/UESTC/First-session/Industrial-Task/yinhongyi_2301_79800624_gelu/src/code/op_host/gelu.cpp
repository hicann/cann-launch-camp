#include "../op_kernel/gelu_tiling.h"

#include <algorithm>

#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platform.GetCoreNum();

    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);

    const uint32_t blockSize = 32;
    uint32_t inputLengthAlign32 = ((inputNum * typeLength + blockSize - 1) / blockSize) * blockSize;
    uint32_t inputBlockNum = inputLengthAlign32 / blockSize;

    coreNum = std::min(coreNum, inputBlockNum);
    coreNum = std::max(coreNum, static_cast<uint32_t>(1));
    context->SetBlockDim(coreNum);

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint32_t bufferNum = 2;
    const uint32_t ubTensorNum = 3;
    uint32_t tileBlockNum = (ubSize / blockSize / bufferNum) / ubTensorNum;
    tileBlockNum = std::max(tileBlockNum, static_cast<uint32_t>(1));
    uint32_t tileDataNum = tileBlockNum * blockSize / typeLength;
    tileDataNum = std::min(tileDataNum, static_cast<uint32_t>(4096));
    tileDataNum = std::max(tileDataNum, static_cast<uint32_t>(1));

    uint32_t smallCoreBlockNum = inputBlockNum / coreNum;
    uint32_t tailCoreNum = inputBlockNum % coreNum;
    uint32_t bigCoreBlockNum = smallCoreBlockNum + 1;
    uint32_t smallCoreDataNum = smallCoreBlockNum * blockSize / typeLength;
    uint32_t bigCoreDataNum = bigCoreBlockNum * blockSize / typeLength;

    auto *tiling = context->GetTilingData<GeluTilingData>();
    tiling->totalDataNum = inputNum;
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tileDataNum = tileDataNum;
    tiling->tailCoreNum = tailCoreNum;

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Gelu : public OpDef {
public:
    explicit Gelu(const char *name) : OpDef(name)
    {
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Gelu);
}  // namespace ops

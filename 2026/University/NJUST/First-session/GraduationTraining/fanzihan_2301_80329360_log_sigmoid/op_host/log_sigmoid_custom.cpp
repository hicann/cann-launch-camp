#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
namespace {

constexpr uint32_t kMaxCoreNum = 8U;
constexpr uint32_t kTileElements = 2048U;

static uint32_t GetElementCount(const gert::StorageShape* inputShape)
{
    const auto& shape = inputShape->GetStorageShape();
    uint32_t elementCount = 1U;

    for (int dim = 0; dim < shape.GetDimNum(); ++dim) {
        elementCount *= static_cast<uint32_t>(shape.GetDim(dim));
    }
    return elementCount;
}

static bool GetTypeTilingConfig(
    ge::DataType dataType, uint32_t& alignment, uint32_t& tilingKey)
{
    if (dataType == ge::DT_FLOAT) {
        alignment = 8U;
        tilingKey = 1U;
        return true;
    }

    if (dataType == ge::DT_FLOAT16) {
        alignment = 16U;
        tilingKey = 2U;
        return true;
    }

    if (dataType == ge::DT_BF16) {
        alignment = 16U;
        tilingKey = 3U;
        return true;
    }

    return false;
}

static uint32_t FindCoreNum(uint32_t elementCount, uint32_t alignment)
{
    for (uint32_t coreNum = kMaxCoreNum; coreNum >= 1U; --coreNum) {
        if (elementCount % coreNum != 0U) {
            continue;
        }

        const uint32_t elementsPerCore = elementCount / coreNum;
        if (elementsPerCore % alignment == 0U) {
            return coreNum;
        }
    }
    return 1U;
}

} // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto* tilingData =
        context->GetTilingData<LogSigmoidCustomTilingData>();

    const gert::StorageShape* inputShape = context->GetInputShape(0);
    const uint32_t elementCount = GetElementCount(inputShape);

    uint32_t alignment = 0U;
    uint32_t tilingKey = 0U;
    const ge::DataType inputType = context->GetInputDesc(0)->GetDataType();

    if (!GetTypeTilingConfig(inputType, alignment, tilingKey)) {
        return ge::GRAPH_FAILED;
    }

    tilingData->elementCount = elementCount;
    tilingData->tileElements = kTileElements;

    context->SetTilingKey(tilingKey);
    context->SetBlockDim(FindCoreNum(elementCount, alignment));

    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    workspaceSize[0] = 0U;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);

    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class LogSigmoidCustom : public OpDef {
public:
    explicit LogSigmoidCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(LogSigmoidCustom);
} // namespace ops

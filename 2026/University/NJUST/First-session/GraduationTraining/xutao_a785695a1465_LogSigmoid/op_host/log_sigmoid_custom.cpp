#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    LogSigmoidCustomTilingData *tiling = context->GetTilingData<LogSigmoidCustomTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    if (tiling == nullptr || x1_shape == nullptr || context->GetInputDesc(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint64_t dataSize = 1;
    const gert::Shape storageShape = x1_shape->GetStorageShape();
    for (int i = 0; i < storageShape.GetDimNum(); ++i) {
        dataSize *= static_cast<uint64_t>(storageShape.GetDim(i));
    }
    tiling->size = static_cast<uint32_t>(dataSize);

    const auto inputDataType = context->GetInputDesc(0)->GetDataType();
    if (inputDataType == ge::DT_FLOAT) {
        context->SetTilingKey(1);  // float32
    } else if (inputDataType == ge::DT_FLOAT16) {
        context->SetTilingKey(2);  // float16
    } else if (inputDataType == ge::DT_BF16) {
        context->SetTilingKey(3);  // bfloat16
    } else {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(8);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

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
}  // namespace ge

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
}  // namespace ops
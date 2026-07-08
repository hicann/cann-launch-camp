#include "../op_kernel/log_sigmoid_custom_tiling.h"   // ★ 改这里
#include "register/op_def_registry.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto* tiling = context->GetTilingData<LogSigmoidCustomTilingData>();
    const gert::StorageShape* xShape = context->GetInputShape(0);

    uint32_t data_sz = 1;
    for (int i = 0; i < xShape->GetStorageShape().GetDimNum(); ++i) {
        data_sz *= xShape->GetStorageShape().GetDim(i);
    }

    tiling->size = data_sz;

    auto inputDtype = context->GetInputDesc(0)->GetDataType();
    if (inputDtype == ge::DT_FLOAT16) {
        tiling->dataType = 0;
    } else if (inputDtype == ge::DT_FLOAT) {
        tiling->dataType = 1;
    } else if (inputDtype == ge::DT_BF16) {
        tiling->dataType = 2;
    } else {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(8);
    context->GetWorkspaceSizes(1)[0] = 0;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* ctx)
{
    *ctx->GetOutputShape(0) = *ctx->GetInputShape(0);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* ctx)
{
    ctx->SetOutputDataType(0, ctx->GetInputDataType(0));
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
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
             .SetInferDataType(ge::InferDataType);

        this->AICore()
             .SetTiling(optiling::TilingFunc)
             .AddConfig("ascend910b");
    }
};
OP_ADD(LogSigmoidCustom);
}

#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"

#include <cstdint>
#include <limits>

namespace optiling {
constexpr uint32_t BLOCK_DIM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    LogSigmoidCustomTilingData* tiling =
        context->GetTilingData<LogSigmoidCustomTilingData>();

    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* x_shape = context->GetInputShape(0);

    if (x_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t data_size = 1;

    for (int32_t i = 0; i < x_shape->GetStorageShape().GetDimNum(); ++i) {
        int64_t dim = x_shape->GetStorageShape().GetDim(i);

        if (dim <= 0) {
            return ge::GRAPH_FAILED;
        }

        data_size *= dim;
    }

    if (data_size <= 0 ||
        data_size > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        return ge::GRAPH_FAILED;
    }

    tiling->size = static_cast<uint32_t>(data_size);
    tiling->core_num = BLOCK_DIM;

    context->SetBlockDim(BLOCK_DIM);

    auto input_dtype = context->GetInputDesc(0)->GetDataType();

    if (input_dtype == ge::DT_FLOAT) {
        context->SetTilingKey(1);
    } else if (input_dtype == ge::DT_FLOAT16) {
        context->SetTilingKey(2);
    } else if (input_dtype == ge::DT_BF16) {
        context->SetTilingKey(3);
    } else {
        return ge::GRAPH_FAILED;
    }

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);

    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }

    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);

    if (x_shape == nullptr || y_shape == nullptr) {
        return GRAPH_FAILED;
    }

    *y_shape = *x_shape;

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);

    return ge::GRAPH_SUCCESS;
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
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(LogSigmoidCustom);
}

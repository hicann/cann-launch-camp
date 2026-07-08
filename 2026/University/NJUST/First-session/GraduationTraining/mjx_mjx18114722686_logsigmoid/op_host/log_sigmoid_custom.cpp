#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    LogSigmoidCustomTilingData *tiling = context->GetTilingData<LogSigmoidCustomTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);

    // 1. 计算总元素个数
    uint32_t totalElements = 1;
    for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
        totalElements *= x1_shape->GetStorageShape().GetDim(i);
    }

    // 2. 获取数据类型
    const gert::CompileTimeTensorDesc* inputDesc = context->GetInputDesc(0);
    auto dataType = inputDesc->GetDataType();

    uint32_t dataTypeSize = 0;
    uint32_t dataTypeEnum = 0;
    switch (dataType) {
        case ge::DT_FLOAT:
            dataTypeSize = 4;
            dataTypeEnum = 1;  // float32
            break;
        case ge::DT_FLOAT16:
            dataTypeSize = 2;
            dataTypeEnum = 0;  // float16
            break;
        case ge::DT_BF16:
            dataTypeSize = 2;
            dataTypeEnum = 2;  // bfloat16
            break;
        default:
            return ge::GRAPH_FAILED;
    }

    // 3. 填充Tiling数据
    tiling->totalElements = totalElements;
    tiling->dataTypeSize = dataTypeSize;
    tiling->dataType = dataTypeEnum;

    // 4. 设置Block维度
    context->SetBlockDim(8);

    // 5. 设置Workspace大小
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace != nullptr) {
        currentWorkspace[0] = 0;
    }

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

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(LogSigmoidCustom);
}

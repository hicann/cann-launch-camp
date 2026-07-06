#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static uint32_t SelectBlockDim(uint32_t totalLength, uint32_t alignNum)
{
    for (uint32_t block = 8; block > 0; --block) {
        if (totalLength % block == 0 && (totalLength / block) % alignNum == 0) {
            return block;
        }
    }
    return 1;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    LogSigmoidCustomTilingData *tiling = context->GetTilingData<LogSigmoidCustomTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);

    uint32_t dataSize = 1;
    for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
        dataSize *= x1_shape->GetStorageShape().GetDim(i);
    }

    ge::DataType dtype = context->GetInputDesc(0)->GetDataType();

    uint32_t alignNum = 16;
    if (dtype == ge::DT_FLOAT) {
        alignNum = 8;
        context->SetTilingKey(1);
    } else if (dtype == ge::DT_FLOAT16) {
        context->SetTilingKey(2);
    } else {
        context->SetTilingKey(3);
    }

    uint32_t blockDim = SelectBlockDim(dataSize, alignNum);
    tiling->size = dataSize;
    tiling->tileLength = 2048;

    context->SetBlockDim(blockDim);

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
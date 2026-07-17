#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus CalcTilingInfo(gert::TilingContext* ctx)
{
    LogSigmoidCustomTilingData *tileInfo = ctx->GetTilingData<LogSigmoidCustomTilingData>();
    const gert::StorageShape* inputXShape = ctx->GetInputShape(0);

    uint32_t totalElemNum = 1;
    for (int dimIdx = 0; dimIdx < inputXShape->GetStorageShape().GetDimNum(); dimIdx++)
    {
        totalElemNum *= inputXShape->GetStorageShape().GetDim(dimIdx);
    }
    tileInfo->elem_total = totalElemNum;

    ge::DataType dtypeTag = ctx->GetInputDesc(0)->GetDataType();
    if (dtypeTag == ge::DT_FLOAT16)
    {
        tileInfo->dtype_code = 0;
    }
    else if (dtypeTag == ge::DT_FLOAT)
    {
        tileInfo->dtype_code = 1;
    }
    else if (dtypeTag == ge::DT_BF16)
    {
        tileInfo->dtype_code = 2;
    }
    else
    {
        return ge::GRAPH_FAILED;
    }

    ctx->SetBlockDim(8);
    size_t* wsBuf = ctx->GetWorkspaceSizes(1);
    wsBuf[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferOutputShape(gert::InferShapeContext* ctx)
{
    const gert::Shape* xShape = ctx->GetInputShape(0);
    gert::Shape* yShape = ctx->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferOutputDtype(gert::InferDataTypeContext* ctx)
{
    ge::DataType inDtype = ctx->GetInputDataType(0);
    ctx->SetOutputDataType(0, inDtype);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class LogSigmoidCustom : public OpDef {
public:
    explicit LogSigmoidCustom(const char* opLabel) : OpDef(opLabel)
    {
        Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        SetInferShape(ge::InferOutputShape).SetInferDataType(ge::InferOutputDtype);
        AICore().SetTiling(optiling::CalcTilingInfo);
        AICore().AddConfig("ascend910b");
    }
};
OP_ADD(LogSigmoidCustom);
}

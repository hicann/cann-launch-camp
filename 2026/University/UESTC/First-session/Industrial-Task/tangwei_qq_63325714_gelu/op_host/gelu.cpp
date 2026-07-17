#include "register/op_def_registry.h"
#include <cstdint>
#include <limits>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t coreNumAiv = platform.GetCoreNumAiv();
    if (coreNumAiv <= 0) {
        coreNumAiv = 1;
    }

    const gert::Tensor *tensorInputX = context->GetRequiredInputTensor(0);
    ge::DataType dtypeInputX = tensorInputX->GetDataType();

    int64_t shapeSize = tensorInputX->GetShapeSize();
    if (shapeSize <= 0) {
        shapeSize = 1;
    }

    uint32_t lengthInputX = static_cast<uint32_t>(shapeSize);
    uint32_t DT_INPUT_X = static_cast<uint32_t>(dtypeInputX);
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

    uint32_t TILE_LENGTH = 4096;
    uint32_t blockNum = static_cast<uint32_t>(coreNumAiv);

    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
    tiling->length = lengthInputX;
    tiling->blockNum = blockNum;
    tiling->tileLength = TILE_LENGTH;

    context->SetBlockDim(blockNum);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const auto inputDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDtype);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Gelu : public OpDef {
public:
    explicit Gelu(const char *name) : OpDef(name) {
        this->Input("input_x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(Gelu);
}
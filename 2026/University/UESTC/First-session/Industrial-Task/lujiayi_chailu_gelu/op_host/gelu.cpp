// Host侧Tiling、shape推导和算子原型注册
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
static constexpr uint32_t TILE_LENGTH = 4096;  // 元素个数

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t numCoresAiv = platform.GetCoreNumAiv();

    const gert::Tensor *tensorInputX = context->GetRequiredInputTensor(0);
    ge::DataType dtypeInputX = tensorInputX->GetDataType();
    uint32_t totalLength = static_cast<uint32_t>(tensorInputX->GetShapeSize());

    // 配置tiling key：kernel侧按float16/float32实例化
    uint32_t DT_INPUT_X = static_cast<uint32_t>(dtypeInputX);
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

    uint32_t selectedTmpSize = 131072U;

    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
    tiling->totalLength = totalLength;
    tiling->tileLength = TILE_LENGTH;
    tiling->tmpSize = selectedTmpSize;

    // 小shape不浪费核，大shape尽量铺满AIV核
    uint32_t needCore = (totalLength + TILE_LENGTH - 1) / TILE_LENGTH;
    uint32_t usedCore = needCore == 0 ? 1 : needCore;
    if (usedCore > static_cast<uint32_t>(numCoresAiv)) {
        usedCore = static_cast<uint32_t>(numCoresAiv);
    }
    context->SetBlockDim(usedCore);

    // 使用Erf高阶API，给系统workspace预留空间
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Gelu : public OpDef {
public:
    explicit Gelu(const char *name) : OpDef(name) {
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Gelu);
}  // namespace ops

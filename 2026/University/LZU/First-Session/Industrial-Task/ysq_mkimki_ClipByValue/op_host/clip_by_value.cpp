#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/clip_by_value_tiling.h"
#include "../op_kernel/tiling_key_clip_by_value.h"

namespace optiling {
constexpr uint32_t TILE_LENGTH = 4096U;

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    const gert::Tensor *tensorMin = context->GetRequiredInputTensor(1);
    const gert::Tensor *tensorMax = context->GetRequiredInputTensor(2);
    if (tensorX == nullptr || tensorMin == nullptr || tensorMax == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtypeX = tensorX->GetDataType();
    if (tensorMin->GetDataType() != dtypeX || tensorMax->GetDataType() != dtypeX) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t length64 = tensorX->GetShapeSize();
    const uint64_t minLength64 = tensorMin->GetShapeSize();
    const uint64_t maxLength64 = tensorMax->GetShapeSize();
    if (length64 == 0U || length64 > 0xFFFFFFFFULL) {
        return ge::GRAPH_FAILED;
    }

    const bool minIsScalar = (minLength64 == 1U);
    const bool maxIsScalar = (maxLength64 == 1U);
    if ((!minIsScalar && minLength64 != length64) ||
        (!maxIsScalar && maxLength64 != length64)) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t numCoresAiv = platform.GetCoreNumAiv();
    if (numCoresAiv <= 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t length = static_cast<uint32_t>(length64);
    uint32_t blockDim = static_cast<uint32_t>(numCoresAiv);
    if (blockDim > length) {
        blockDim = length;
    }

    ClipByValueTilingData *tiling =
        context->GetTilingData<ClipByValueTilingData>();
    tiling->length = length;
    tiling->tileLength = TILE_LENGTH;
    tiling->minIsScalar = minIsScalar ? 1U : 0U;
    tiling->maxIsScalar = maxIsScalar ? 1U : 0U;

    const uint32_t DT_X = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    context->SetBlockDim(blockDim);
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class ClipByValue : public OpDef {
public:
    explicit ClipByValue(const char *name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("clip_value_min")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("clip_value_max")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(ClipByValue);
}  // namespace ops

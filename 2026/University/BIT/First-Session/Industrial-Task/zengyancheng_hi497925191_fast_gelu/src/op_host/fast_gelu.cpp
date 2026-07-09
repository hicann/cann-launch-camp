// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"
#include "graph/utils/type_utils.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t numCores = platform.GetCoreNumAiv();
    if (numCores == 0) numCores = 1;

    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_x = tensor_x->GetDataType();
    uint32_t typeSize = ge::GetSizeByDataType(dtype_x);
    uint32_t totalLength = static_cast<uint32_t>(tensor_x->GetShapeSize());

    // MIN_ELEMENTS_PER_CORE = 64, avoid too many cores
    constexpr uint32_t MIN_ELEM = 64;
    uint32_t blockDim = numCores;
    if (totalLength > 0) {
        uint32_t maxCores = (totalLength + MIN_ELEM - 1) / MIN_ELEM;
        if (blockDim > maxCores) blockDim = maxCores;
        if (blockDim > totalLength) blockDim = totalLength;
    }
    if (blockDim == 0) blockDim = 1;

    uint32_t alignNum = 32 / typeSize;
    uint32_t blockLength = (totalLength + blockDim - 1) / blockDim;
    blockLength = ((blockLength + alignNum - 1) / alignNum) * alignNum;

    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    FastGeluTilingData *tiling = context->GetTilingData<FastGeluTilingData>();
    tiling->totalLength = totalLength;
    tiling->blockLength = blockLength;
    tiling->blockDim = blockDim;

    context->SetBlockDim(blockDim);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class FastGelu : public OpDef {
public:
    explicit FastGelu(const char *name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(FastGelu);
}  // namespace ops

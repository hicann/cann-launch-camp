// Host侧：算子注册 + shape/dtype推导 + tiling参数生成
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const gert::Tensor *xTensor = context->GetRequiredInputTensor(0);
    if (xTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtype = xTensor->GetDataType();
    if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    // 根据输入dtype选择Kernel模板实例：float16 或 float32。
    uint32_t dtX = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, dtX);

    const uint32_t totalLength = static_cast<uint32_t>(xTensor->GetShapeSize());

    // 多核优化：小shape用1核减少调度开销；大shape启用多核，避免单核在大case上很慢。
    uint32_t blockDim = 1;
    if (totalLength >= 262144) {
        blockDim = 20;
    } else if (totalLength >= 65536) {
        blockDim = 16;
    } else if (totalLength >= 16384) {
        blockDim = 8;
    } else if (totalLength >= 4096) {
        blockDim = 4;
    }

    FastGeluTilingData *tiling = context->GetTilingData<FastGeluTilingData>();
    tiling->length = totalLength;
    tiling->blockDim = blockDim;

    context->SetBlockDim(blockDim);

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr) {
        return GRAPH_FAILED;
    }
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    ge::DataType xType = context->GetInputDataType(0);
    if (xType != ge::DT_FLOAT16 && xType != ge::DT_FLOAT) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, xType);
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

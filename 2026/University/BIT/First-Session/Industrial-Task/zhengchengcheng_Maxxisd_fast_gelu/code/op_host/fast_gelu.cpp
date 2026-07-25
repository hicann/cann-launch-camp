// Host tiling implementation
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t num_cores_aiv = platform.GetCoreNumAiv();

    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_x = tensor_x->GetDataType();
    uint32_t length_x = tensor_x->GetShapeSize();

    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    FastGeluTilingData *tiling = context->GetTilingData<FastGeluTilingData>();
    uint32_t block_num = static_cast<uint32_t>(num_cores_aiv);
    if (length_x <= 2048) {
        block_num = 1;
    } else if (length_x <= 8192) {
        block_num = 2;
    } else if (length_x <= 32768) {
        block_num = 4;
    } else if (length_x <= 131072) {
        block_num = 8;
    }
    if (block_num > static_cast<uint32_t>(num_cores_aiv)) {
        block_num = static_cast<uint32_t>(num_cores_aiv);
    }
    if (block_num == 0) {
        block_num = 1;
    }

    tiling->length = length_x;
    tiling->blockNum = block_num;

    context->SetBlockDim(block_num);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *x_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const ge::DataType x_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, x_dtype);
    return ge::GRAPH_SUCCESS;
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

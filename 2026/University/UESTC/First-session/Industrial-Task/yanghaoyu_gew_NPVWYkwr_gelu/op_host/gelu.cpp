#include "register/op_def_registry.h"

#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"

#include "../op_kernel/tiling_key_gelu.h"


namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    int32_t num_cores_aiv = platform.GetCoreNumAiv();

    uint64_t ub_size = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    const gert::Tensor *tensor_input_x = context->GetRequiredInputTensor(0);

    ge::DataType dtype_input_x = tensor_input_x->GetDataType();

    int dtype_size_input_x = ge::GetSizeByDataType(dtype_input_x);

    uint32_t length_input_x = tensor_input_x->GetShapeSize();

    uint32_t size_input_x = tensor_input_x->GetSize();

    (void)ub_size;
    (void)size_input_x;

    uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);

    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();

    tiling->length = length_input_x;

uint32_t core_num = static_cast<uint32_t>(num_cores_aiv);
uint32_t block_dim = 1;

uint32_t min_data_per_core = 256;

if (dtype_size_input_x == 2) {
    min_data_per_core = 512;
}

if (length_input_x > 0 && core_num > 0) {
    block_dim = (length_input_x + min_data_per_core - 1) / min_data_per_core;

    if (block_dim > core_num) {
        block_dim = core_num;
    }

    if (block_dim == 0) {
        block_dim = 1;
    }
}

context->SetBlockDim(block_dim);

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
    const auto input_data_type = context->GetInputDataType(0);

    context->SetOutputDataType(0, input_data_type);

    return ge::GRAPH_SUCCESS;
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

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Gelu);

}  // namespace ops
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

namespace optiling {
static ge::graphStatus BuildTiling(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t available_cores = static_cast<uint32_t>(platform.GetCoreNumAiv());

    const gert::Tensor *input_tensor = context->GetRequiredInputTensor(0);
    ge::DataType input_type = input_tensor->GetDataType();
    uint64_t elements = static_cast<uint64_t>(input_tensor->GetShapeSize());
    uint32_t bytes_per_element = static_cast<uint32_t>(ge::GetSizeByDataType(input_type));

    uint32_t dt_selector = static_cast<uint32_t>(input_type);
    ASCENDC_TPL_SEL_PARAM(context, dt_selector);

    constexpr uint32_t kMinElementsPerCore = 64;
    uint32_t active_cores = available_cores;

    if (elements > 0) {
        uint32_t useful_cores = static_cast<uint32_t>(
            (elements + kMinElementsPerCore - 1) / kMinElementsPerCore);
        active_cores = std::min(active_cores, useful_cores);
        active_cores = std::min(active_cores, static_cast<uint32_t>(elements));
    }

    if (active_cores == 0) {
        active_cores = 1;
    }

    uint64_t align_factor = 32u / bytes_per_element;
    uint64_t chunk_size = (elements + active_cores - 1) / active_cores;
    chunk_size = ((chunk_size + align_factor - 1) / align_factor) * align_factor;

    FastGeluTilingData *tiling_data = context->GetTilingData<FastGeluTilingData>();
    tiling_data->length = elements;
    tiling_data->blockDim = active_cores;
    tiling_data->blockLength = chunk_size;

    context->SetBlockDim(active_cores);
    size_t *workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *shape_x = context->GetInputShape(0);
    gert::Shape *shape_y = context->GetOutputShape(0);
    *shape_y = *shape_x;
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
        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::BuildTiling)
            .AddConfig("ascend910b");
    }
};

OP_ADD(FastGelu);
}  // namespace ops

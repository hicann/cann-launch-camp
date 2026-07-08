#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t num_cores_aiv = platform.GetCoreNumAiv();

    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_x = tensor_x->GetDataType();
    uint64_t length_x = static_cast<uint64_t>(tensor_x->GetShapeSize());
    uint32_t dtype_size = static_cast<uint32_t>(ge::GetSizeByDataType(dtype_x));

    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // ── Block-dim tuning ──────────────────────────────────────────────
    // Avoid launching more cores than beneficial: each core has scheduling
    // + pipeline-setup overhead.  Keep at least 64 elements per core.
    constexpr uint32_t MIN_ELEMENTS_PER_CORE = 64;
    uint32_t block_dim = static_cast<uint32_t>(num_cores_aiv);
    if (length_x != 0) {
        uint32_t max_useful_cores =
            static_cast<uint32_t>((length_x + MIN_ELEMENTS_PER_CORE - 1)
                                  / MIN_ELEMENTS_PER_CORE);
        if (block_dim > max_useful_cores) {
            block_dim = max_useful_cores;
        }
        if (block_dim > length_x) {
            block_dim = static_cast<uint32_t>(length_x);
        }
    }
    if (block_dim == 0) {
        block_dim = 1;
    }

    // ── Even-split block length (aligned for DMA) ─────────────────────
    uint64_t align_num    = 32 / dtype_size;
    uint64_t block_length = (length_x + block_dim - 1) / block_dim;
    block_length = ((block_length + align_num - 1) / align_num) * align_num;

    FastGeluTilingData *tiling = context->GetTilingData<FastGeluTilingData>();
    tiling->length      = length_x;
    tiling->blockDim    = block_dim;
    tiling->blockLength = block_length;

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
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(FastGelu);
}  // namespace ops

// Host侧Tiling实现：v52 GitCode top-architecture adaptation
// 借鉴公开优秀提交的核心工程思路：按每核最少64元素限制有效核数，
// 并在Host侧下发32B对齐后的blockLength，避免Kernel侧平均切分导致非对齐起点。
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
    if (num_cores_aiv <= 0) {
        num_cores_aiv = 1;
    }

    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    if (tensor_x == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType dtype_x = tensor_x->GetDataType();
    uint64_t length_x = static_cast<uint64_t>(tensor_x->GetShapeSize());
    uint32_t dtype_size = static_cast<uint32_t>(ge::GetSizeByDataType(dtype_x));
    if (dtype_size == 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 公开优秀提交的关键点：小shape不要盲目满核，但也不要固定1核；
    // 以64 elements/core控制调度开销和并行度平衡。
    constexpr uint32_t MIN_ELEMENTS_PER_CORE = 64;
    uint32_t block_dim = static_cast<uint32_t>(num_cores_aiv);
    if (length_x != 0) {
        uint32_t max_useful_cores = static_cast<uint32_t>(
            (length_x + MIN_ELEMENTS_PER_CORE - 1) / MIN_ELEMENTS_PER_CORE);
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

    // Host侧计算每core处理长度，并上调到32B对应的元素倍数。
    // Kernel只使用 blockIdx * blockLength 取offset，确保core起点对齐。
    uint64_t align_num = 32 / dtype_size;
    if (align_num == 0) {
        align_num = 1;
    }
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
    if (x_shape == nullptr || y_shape == nullptr) {
        return GRAPH_FAILED;
    }
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
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(FastGelu);
}  // namespace ops

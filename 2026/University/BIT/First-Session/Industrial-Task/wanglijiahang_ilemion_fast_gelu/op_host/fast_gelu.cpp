// Host侧Tiling实现
#include <algorithm>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

namespace optiling {
namespace {
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t FLOAT_TMP_NUM = 3;
constexpr uint32_t SMALL_TENSOR_LENGTH = 8192;
constexpr uint32_t SMALL_TENSOR_CORE_NUM = 8;

uint32_t CeilDiv(uint32_t value, uint32_t divisor) {
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

uint32_t AlignDown(uint32_t value, uint32_t align) {
    return align == 0 ? value : value / align * align;
}
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t num_cores_aiv = static_cast<uint32_t>(platform.GetCoreNumAiv());
    uint64_t ub_size;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_x = tensor_x->GetDataType();
    int dtype_size_x = ge::GetSizeByDataType(dtype_x);
    uint32_t length_x = static_cast<uint32_t>(tensor_x->GetShapeSize());
    if ((dtype_x != ge::DT_FLOAT16 && dtype_x != ge::DT_FLOAT) || dtype_size_x <= 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    const uint32_t elements_per_block = std::max<uint32_t>(
        1, BLOCK_SIZE / static_cast<uint32_t>(dtype_size_x));
    uint32_t block_dim = 1;
    if (length_x > 0) {
        uint32_t max_core_by_data = std::max<uint32_t>(1, CeilDiv(length_x, elements_per_block));
        block_dim = std::min(std::max<uint32_t>(1, num_cores_aiv), max_core_by_data);
        if (length_x <= SMALL_TENSOR_LENGTH) {
            block_dim = std::min(block_dim, SMALL_TENSOR_CORE_NUM);
        }
    }

    uint32_t block_length = length_x == 0 ? 0 : CeilDiv(length_x, block_dim);
    uint64_t bytes_per_element =
        static_cast<uint64_t>(dtype_size_x) * BUFFER_NUM * 2 + sizeof(float) * FLOAT_TMP_NUM;
    uint32_t tile_length = elements_per_block;
    if (bytes_per_element > 0 && ub_size >= BLOCK_SIZE) {
        uint64_t raw_tile_length = (ub_size / 2) / bytes_per_element;
        raw_tile_length = std::min<uint64_t>(
            raw_tile_length, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
        tile_length = AlignDown(static_cast<uint32_t>(raw_tile_length), elements_per_block);
        tile_length = std::max<uint32_t>(tile_length, elements_per_block);
    }

    FastGeluTilingData *tiling = context->GetTilingData<FastGeluTilingData>();
    tiling->length = length_x;
    tiling->blockLength = block_length;
    tiling->tileLength = tile_length;

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
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(FastGelu);
}  // namespace ops

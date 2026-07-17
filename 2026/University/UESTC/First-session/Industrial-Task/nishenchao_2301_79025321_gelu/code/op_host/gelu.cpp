// Host侧Tiling实现
#include "register/op_def_registry.h"

#include "tiling/platform/platform_ascendc.h"

#include <algorithm>

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t num_cores_aiv = platform.GetCoreNumAiv();
    if (num_cores_aiv <= 0) {
        num_cores_aiv = 1;
    }

    uint64_t ub_size = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    const gert::Tensor *tensor_input_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_input_x = tensor_input_x->GetDataType();
    int32_t dtype_size_input_x = ge::GetSizeByDataType(dtype_input_x);
    uint32_t length_input_x = tensor_input_x->GetShapeSize();

    uint32_t dt_input_x = static_cast<uint32_t>(dtype_input_x);
    ASCENDC_TPL_SEL_PARAM(context, dt_input_x);

    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
    tiling->length = length_input_x;

    context->SetBlockDim(num_cores_aiv);

    size_t *workspace_sizes = context->GetWorkspaceSizes(1);
    workspace_sizes[0] = 0;

    constexpr uint32_t kBlockSize = 32;
    const uint32_t aligned_length = ((length_input_x + kBlockSize - 1) / kBlockSize) * kBlockSize;
    const uint32_t block_count = aligned_length / kBlockSize;
    const uint32_t base_blocks_per_core = block_count / num_cores_aiv;
    const uint32_t extra_blocks = block_count % num_cores_aiv;

    const uint32_t small_core_data_num = base_blocks_per_core * kBlockSize;
    const uint32_t big_core_data_num = (base_blocks_per_core + 1) * kBlockSize;


    uint32_t available_ub = ub_size * 0.75;
    uint32_t tile_data_num = static_cast<uint32_t>(available_ub) / (2 * 2 * dtype_size_input_x);
    if (tile_data_num < kBlockSize) {
        tile_data_num = kBlockSize;
    }
    tile_data_num = std::min(tile_data_num, aligned_length);
    tile_data_num = ((tile_data_num + kBlockSize - 1) / kBlockSize) * kBlockSize;

    tiling->smallCoreDataNum = small_core_data_num;
    tiling->bigCoreDataNum = big_core_data_num;
    tiling->finalSmallTileNum = (small_core_data_num == 0) ? 0 : ((small_core_data_num + tile_data_num - 1) / tile_data_num);
    tiling->finalBigTileNum = (big_core_data_num == 0) ? 0 : ((big_core_data_num + tile_data_num - 1) / tile_data_num);
    tiling->smallTailDataNum = (small_core_data_num % tile_data_num == 0) ? tile_data_num : (small_core_data_num % tile_data_num);
    tiling->bigTailDataNum = (big_core_data_num % tile_data_num == 0) ? tile_data_num : (big_core_data_num % tile_data_num);
    tiling->tileDataNum = tile_data_num;
    tiling->tailBlockNum = extra_blocks;
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
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Gelu);
}  // namespace ops
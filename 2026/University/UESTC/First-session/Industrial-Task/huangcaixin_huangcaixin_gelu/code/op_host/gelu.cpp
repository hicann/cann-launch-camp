#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/tiling_key_gelu.h"
#include "graph/utils/type_utils.h"
#include "../op_kernel/gelu_tiling.h"
#include <algorithm>

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores = platform.GetCoreNumAiv();
        if (num_cores <= 0) num_cores = 1;

        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_input = context->GetRequiredInputTensor(0);
        uint32_t length_input = tensor_input->GetShapeSize();
        uint32_t dtype_size = ge::GetSizeByDataType(tensor_input->GetDataType());

        // 模板选择（虽然 Host 侧不直接使用，但保留以匹配 TilingKey）
        uint32_t dt_input = static_cast<uint32_t>(tensor_input->GetDataType());
        ASCENDC_TPL_SEL_PARAM(context, dt_input);

        constexpr uint32_t kBlockSize = 32;
        const uint32_t aligned_length = ((length_input + kBlockSize - 1) / kBlockSize) * kBlockSize;
        const uint32_t block_count = aligned_length / kBlockSize;
        const uint32_t base_blocks_per_core = block_count / num_cores;
        const uint32_t extra_blocks = block_count % num_cores;

        const uint32_t small_core_data_num = base_blocks_per_core * kBlockSize;
        const uint32_t big_core_data_num = (base_blocks_per_core + 1) * kBlockSize;

        // 动态计算 tile 大小
        uint32_t available_ub = ub_size * 0.75;
        uint32_t tile_data_num = static_cast<uint32_t>(available_ub) / (2 * 2 * dtype_size);
        if (tile_data_num < kBlockSize) tile_data_num = kBlockSize;
        tile_data_num = std::min(tile_data_num, aligned_length);
        tile_data_num = ((tile_data_num + kBlockSize - 1) / kBlockSize) * kBlockSize;

        auto tiling = context->GetTilingData<GeluTilingData>();
        tiling->smallCoreDataNum = small_core_data_num;
        tiling->bigCoreDataNum = big_core_data_num;
        tiling->finalSmallTileNum = (small_core_data_num == 0) ? 0 :
            ((small_core_data_num + tile_data_num - 1) / tile_data_num);
        tiling->finalBigTileNum = (big_core_data_num == 0) ? 0 :
            ((big_core_data_num + tile_data_num - 1) / tile_data_num);
        tiling->smallTailDataNum = (small_core_data_num % tile_data_num == 0) ?
            tile_data_num : (small_core_data_num % tile_data_num);
        tiling->bigTailDataNum = (big_core_data_num % tile_data_num == 0) ?
            tile_data_num : (big_core_data_num % tile_data_num);
        tiling->tileDataNum = tile_data_num;
        tiling->tailBlockNum = extra_blocks;

        context->SetBlockDim(num_cores);
        context->GetWorkspaceSizes(1)[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}

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
}

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
}
// Host-side tiling and operator registration.
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t CORE_SPLIT_ELEM_NUM = 2048;
constexpr uint32_t TILE_ELEM_NUM = 8192;
constexpr uint32_t FLOAT_TILE_ELEM_NUM = 4096;
constexpr uint32_t HALF_TILE_ELEM_NUM = 8192;
constexpr uint32_t VECTOR_CORE_CAP = 48;
constexpr uint32_t SMALL_SHAPE_BASE_THRESHOLD = CORE_SPLIT_ELEM_NUM * 24U;

namespace optiling {
    static uint32_t CeilDiv(uint32_t value, uint32_t divisor) {
        return value == 0 || divisor == 0 ? 0U : 1U + (value - 1U) / divisor;
    }

    static uint32_t GetDataTypeSize(ge::DataType dtype) {
        return dtype == ge::DT_FLOAT16 ? 2U : 4U;
    }

    static uint32_t GetTileElemNum(ge::DataType dtype) {
        return dtype == ge::DT_FLOAT16 ? HALF_TILE_ELEM_NUM : FLOAT_TILE_ELEM_NUM;
    }

    static uint32_t GetSmallShapeThreshold(ge::DataType dtype) {
        return dtype == ge::DT_FLOAT16 ? SMALL_SHAPE_BASE_THRESHOLD * 2U : SMALL_SHAPE_BASE_THRESHOLD;
    }

    static uint32_t GetTargetCoreNum(uint32_t length_x, ge::DataType dtype) {
        if (length_x == 0) {
            return 1U;
        }
        if (length_x <= CORE_SPLIT_ELEM_NUM * 3U) {
            return CeilDiv(length_x, CORE_SPLIT_ELEM_NUM);
        }

        uint32_t tile_elem_num = GetTileElemNum(dtype);
        if (length_x <= tile_elem_num * 4U) {
            return 8U;
        }
        if (length_x <= tile_elem_num * 8U) {
            return 12U;
        }
        if (length_x <= tile_elem_num * 16U) {
            return 16U;
        }
        if (length_x <= tile_elem_num * 48U) {
            return 24U;
        }
        return VECTOR_CORE_CAP;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t length_x = static_cast<uint32_t>(tensor_x->GetShapeSize());
        uint32_t type_length = GetDataTypeSize(dtype_x);

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        uint32_t small_shape_threshold = GetSmallShapeThreshold(dtype_x);
        uint32_t IS_SMALL_SHAPE = length_x > 0 && length_x <= small_shape_threshold ? 1U : 0U;
        ASCENDC_TPL_SEL_PARAM(context, DT_X, IS_SMALL_SHAPE);

        FastGeluTilingData *tiling = context->GetTilingData<FastGeluTilingData>();
        tiling->totalLength = length_x;

        uint64_t input_length_bytes = static_cast<uint64_t>(length_x) * type_length;
        uint32_t aligned_length_bytes =
            input_length_bytes == 0
                ? 0U
                : static_cast<uint32_t>(((input_length_bytes + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
        uint32_t total_block_num = aligned_length_bytes / BLOCK_SIZE;
        uint32_t tile_elem_num = GetTileElemNum(dtype_x);
        if (IS_SMALL_SHAPE == 1U && total_block_num > 0) {
            tile_elem_num = total_block_num * BLOCK_SIZE / type_length;
        }

        uint32_t block_dim = 1U;
        if (total_block_num > 0 && IS_SMALL_SHAPE == 0U) {
            uint32_t max_core_num = num_cores_aiv > 0 ? static_cast<uint32_t>(num_cores_aiv) : 1U;
            if (max_core_num > VECTOR_CORE_CAP) {
                max_core_num = VECTOR_CORE_CAP;
            }
            block_dim = GetTargetCoreNum(length_x, dtype_x);
            if (block_dim > max_core_num) {
                block_dim = max_core_num;
            }
            if (block_dim > total_block_num) {
                block_dim = total_block_num;
            }
        }
        context->SetBlockDim(block_dim);

        uint32_t base_block_num = block_dim > 0 ? total_block_num / block_dim : 0U;
        uint32_t tail_block_num = block_dim > 0 ? total_block_num % block_dim : 0U;
        uint32_t tile_block_num = (tile_elem_num * type_length) / BLOCK_SIZE;
        if (tile_block_num == 0) {
            tile_block_num = 1;
        }

        uint32_t small_core_data_num = base_block_num * BLOCK_SIZE / type_length;
        uint32_t big_core_data_num = (base_block_num + 1U) * BLOCK_SIZE / type_length;

        uint32_t small_tile_num = base_block_num / tile_block_num;
        uint32_t final_small_tile_num =
            base_block_num == 0
                ? 0U
                : ((base_block_num % tile_block_num) == 0 ? small_tile_num : small_tile_num + 1U);
        uint32_t small_tail_data_num = small_core_data_num - tile_elem_num * small_tile_num;
        if (small_tail_data_num == 0 && final_small_tile_num > 0) {
            small_tail_data_num = tile_elem_num;
        }

        uint32_t big_block_num = base_block_num + 1U;
        uint32_t big_tile_num = big_block_num / tile_block_num;
        uint32_t final_big_tile_num =
            big_block_num == 0
                ? 0U
                : ((big_block_num % tile_block_num) == 0 ? big_tile_num : big_tile_num + 1U);
        uint32_t big_tail_data_num = big_core_data_num - tile_elem_num * big_tile_num;
        if (big_tail_data_num == 0 && final_big_tile_num > 0) {
            big_tail_data_num = tile_elem_num;
        }

        tiling->smallCoreDataNum = small_core_data_num;
        tiling->bigCoreDataNum = big_core_data_num;
        tiling->finalBigTileNum = final_big_tile_num;
        tiling->finalSmallTileNum = final_small_tile_num;
        tiling->tileDataNum = tile_elem_num;
        tiling->smallTailDataNum = small_tail_data_num;
        tiling->bigTailDataNum = big_tail_data_num;
        tiling->tailBlockNum = tail_block_num;

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

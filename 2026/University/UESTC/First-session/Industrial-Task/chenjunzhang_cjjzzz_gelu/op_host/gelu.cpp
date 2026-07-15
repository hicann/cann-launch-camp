// Host侧Tiling实现
#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"

#include "tiling/platform/platform_ascendc.h"


#include "../op_kernel/gelu_tiling.h"

#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
    constexpr uint32_t GELU_MIN_TILE_LENGTH = 256;
    constexpr uint32_t GELU_MAX_TILE_LENGTH = 2816;
    constexpr uint32_t GELU_DEFAULT_TILE_LENGTH = 1024;
    constexpr uint32_t GELU_BUFFER_NUM = 2;
    constexpr uint32_t GELU_LOCAL_TENSOR_NUM = GELU_BUFFER_NUM * 2 + 1;
    constexpr uint64_t GELU_UB_RESERVED_BYTES = 16 * 1024;
    constexpr uint32_t GELU_BLOCK_BYTES = 32;

    static uint32_t AlignDown(uint32_t value, uint32_t align) {
        return align == 0 ? value : value / align * align;
    }

    static uint32_t AlignUp(uint32_t value, uint32_t align) {
        return align == 0 ? value : (value + align - 1) / align * align;
    }

    static uint32_t DivCeil(uint32_t lhs, uint32_t rhs) {
        return rhs == 0 ? 0 : (lhs + rhs - 1) / rhs;
    }

    static uint32_t CalcBlockDim(uint32_t length, int32_t num_cores_aiv, ge::DataType dtype) {
        if (length == 0 || num_cores_aiv <= 0) {
            return 1;
        }

        uint32_t max_cores = static_cast<uint32_t>(num_cores_aiv);
        if (length <= 512) {
            if (dtype == ge::DT_FLOAT && length > 256) {
                return std::min(std::min(max_cores, length), 4U);
            }
            return 1;
        }
        if (length <= 1024) {
            return std::min(std::min(max_cores, length), 4U);
        }
        if (length <= 2048) {
            return std::min(std::min(max_cores, length), 8U);
        }
        if (length <= 4096) {
            return std::min(std::min(max_cores, length), 12U);
        }
        if (length <= 16384) {
            return std::min(std::min(max_cores, length), 16U);
        }
        return std::min(max_cores, length);
    }

    static uint32_t CalcTileLength(uint64_t ub_size, int32_t dtype_size, uint32_t length, uint32_t block_dim,
                                   ge::DataType dtype) {
        if (dtype_size <= 0 || ub_size <= GELU_UB_RESERVED_BYTES) {
            return GELU_DEFAULT_TILE_LENGTH;
        }

        uint64_t usable_ub = ub_size - GELU_UB_RESERVED_BYTES;
        uint64_t bytes_per_tile = static_cast<uint64_t>(dtype_size) * GELU_LOCAL_TENSOR_NUM;
        uint64_t raw_tile = usable_ub / bytes_per_tile;
        uint32_t align_elems = std::max<uint32_t>(1, GELU_BLOCK_BYTES / static_cast<uint32_t>(dtype_size));
        uint32_t tile_length = AlignDown(static_cast<uint32_t>(raw_tile), align_elems);
        tile_length = std::min(tile_length, GELU_MAX_TILE_LENGTH);
        tile_length = std::max(tile_length, GELU_MIN_TILE_LENGTH);
        bool shrink_tile = length <= 16384;
        if (shrink_tile && block_dim > 0) {
            uint32_t per_core_length = AlignUp(DivCeil(length, block_dim), align_elems);
            per_core_length = std::max(per_core_length, GELU_MIN_TILE_LENGTH);
            tile_length = std::min(tile_length, per_core_length);
        }
        return tile_length;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        const gert::Tensor *tensor_input_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_input_x = tensor_input_x->GetDataType();
        int32_t dtype_size_input_x = ge::GetSizeByDataType(dtype_input_x);
        uint32_t length_input_x = static_cast<uint32_t>(tensor_input_x->GetShapeSize());

        uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

        uint32_t used_cores = CalcBlockDim(length_input_x, num_cores_aiv, dtype_input_x);

        GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
        tiling->length = length_input_x;
        tiling->tileLength = CalcTileLength(ub_size, dtype_size_input_x, length_input_x, used_cores, dtype_input_x);

        context->SetBlockDim(used_cores);

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *input_shape = context->GetInputShape(0);
        gert::Shape *output_shape = context->GetOutputShape(0);
        *output_shape = *input_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        ge::DataType input_dtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, input_dtype);
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
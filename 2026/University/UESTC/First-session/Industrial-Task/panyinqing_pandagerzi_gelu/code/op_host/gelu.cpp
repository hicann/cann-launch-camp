// Host侧Tiling实现
#include "register/op_def_registry.h"

#include "tiling/platform/platform_ascendc.h"

#include <algorithm>

#include "../op_kernel/gelu_tiling.h"

#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t core_num = static_cast<uint32_t>(platform.GetCoreNumAiv());
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_input_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_input_x = tensor_input_x->GetDataType();
        uint32_t dtype_size_input_x = static_cast<uint32_t>(ge::GetSizeByDataType(dtype_input_x));
        uint32_t length_input_x = static_cast<uint32_t>(tensor_input_x->GetShapeSize());

        uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

        constexpr uint32_t BLOCK_SIZE = 32;
        constexpr uint32_t BUFFER_COUNT = 3;  // input, output, temp
        constexpr uint32_t UB_RESERVE_FACTOR = 2;
        uint32_t align_num = std::max(BLOCK_SIZE / dtype_size_input_x, static_cast<uint32_t>(1));
        uint32_t tile_length = static_cast<uint32_t>(ub_size / dtype_size_input_x / BUFFER_COUNT / UB_RESERVE_FACTOR);
        tile_length = std::max(tile_length, align_num);
        tile_length = (tile_length / align_num) * align_num;

        uint32_t min_elements_per_core = dtype_input_x == ge::DT_FLOAT16 ? 512 : 256;
        uint32_t block_dim = (length_input_x + min_elements_per_core - 1) / min_elements_per_core;
        block_dim = std::min(core_num, block_dim);
        block_dim = std::max(block_dim, static_cast<uint32_t>(1));

        GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
        tiling->length = length_input_x;
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

// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext* context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        if (num_cores_aiv == 0) {
            num_cores_aiv = 8;
        }

        const gert::Tensor* tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        int dtype_size_x = ge::GetSizeByDataType(dtype_x);
        uint32_t length_x = tensor_x->GetShapeSize();

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        uint32_t BLOCK_DIM = num_cores_aiv;
        uint32_t align_size = 32 / dtype_size_x;
        
        uint64_t ub_size = 32 * 1024;
        uint64_t safety_margin = 2 * 1024;
        uint32_t num_buffers = 7;
        
        uint32_t max_tile_length = static_cast<uint32_t>((ub_size - safety_margin) / (num_buffers * dtype_size_x));
        max_tile_length = (max_tile_length / align_size) * align_size;

        uint32_t core_length = length_x / BLOCK_DIM;
        uint32_t core_remain = length_x % BLOCK_DIM;

        FastGeluTilingData* tiling = context->GetTilingData<FastGeluTilingData>();

        uint32_t this_core_length = core_length + (core_remain > 0 ? 1 : 0);
        uint32_t aligned_core_length = ((this_core_length + align_size - 1) / align_size) * align_size;

        if (aligned_core_length <= max_tile_length) {
            tiling->totalLength = aligned_core_length;
            tiling->tileNum = 1;
            tiling->tileLength = aligned_core_length;
            tiling->lastTileLength = aligned_core_length;
        }
        else {
            tiling->totalLength = aligned_core_length;
            tiling->tileLength = max_tile_length;

            uint32_t num_tiles = aligned_core_length / max_tile_length;
            uint32_t tail_len = aligned_core_length % max_tile_length;

            if (tail_len == 0) {
                tiling->tileNum = num_tiles;
                tiling->lastTileLength = max_tile_length;
            }
            else {
                tiling->tileNum = num_tiles + 1;
                tiling->lastTileLength = tail_len;
            }
        }

        context->SetBlockDim(BLOCK_DIM);

        size_t* currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext* context) {
        const gert::Shape* x_shape = context->GetInputShape(0);
        gert::Shape* y_shape = context->GetOutputShape(0);
        *y_shape = *x_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext* context) {
        const auto inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class FastGelu : public OpDef {
    public:
        explicit FastGelu(const char* name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND });
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND });
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(FastGelu);
}  // namespace ops

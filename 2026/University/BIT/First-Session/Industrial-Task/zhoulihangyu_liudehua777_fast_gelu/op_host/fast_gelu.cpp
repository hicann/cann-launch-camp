#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

namespace optiling {
    const uint32_t BLOCK_DIM = 32;

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores = platform.GetCoreNumAiv();
        if (num_cores == 0) num_cores = 1;

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t type_size = ge::GetSizeByDataType(dtype_x);
        uint32_t length_x = tensor_x->GetShapeSize();

        uint32_t align_num = BLOCK_DIM / type_size;
        uint32_t total_length_aligned = ((length_x + align_num - 1) / align_num) * align_num;

        uint32_t block_len = total_length_aligned / num_cores;
        block_len = ((block_len + align_num - 1) / align_num) * align_num;

        uint32_t used_cores = num_cores;
        if (block_len == 0) {
            block_len = align_num;
            used_cores = 1;
        } else if (total_length_aligned < block_len * num_cores) {
            used_cores = (total_length_aligned + block_len - 1) / block_len;
        }

        uint32_t tile_len = align_num * 256; 
        if (tile_len > block_len) {
            tile_len = block_len;
        }

        uint32_t tile_num = block_len / tile_len;
        uint32_t last_tile_len = block_len % tile_len;
        if (last_tile_len != 0) {
            tile_num += 1;
        } else {
            last_tile_len = tile_len;
        }

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        FastGeluTilingData *tiling = context->GetTilingData<FastGeluTilingData>();
        tiling->totalLength = length_x;
        tiling->alignNum = align_num;
        tiling->blockLength = block_len;
        tiling->tileNum = tile_num;
        tiling->tileLength = tile_len;
        tiling->lastTileLength = last_tile_len;

        context->SetBlockDim(used_cores);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *x_shape = context->GetInputShape(0);
        gert::Shape *y_shape = context->GetOutputShape(0);
        *y_shape = *x_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const ge::DataType x_dtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, x_dtype);
        return ge::GRAPH_SUCCESS;
    }
}  

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
}
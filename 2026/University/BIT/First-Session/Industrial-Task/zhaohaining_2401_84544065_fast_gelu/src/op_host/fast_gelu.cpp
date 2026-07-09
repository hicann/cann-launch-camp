// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 示例: 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t length_x = tensor_x->GetShapeSize();
        uint32_t dtype_size_x = ge::GetSizeByDataType(dtype_x);

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        uint32_t block_dim = 1;
        if (length_x > 0) {
            uint32_t block_size = 1536;
            if (length_x <= 4096) {
                block_size = 1408;
            } else if (length_x >= 262144) {
                block_size = 1024;
            }

            block_dim = (length_x + block_size - 1) / block_size;
            uint32_t max_block_dim = static_cast<uint32_t>(num_cores_aiv);
            if (block_dim > max_block_dim) {
                block_dim = max_block_dim;
            }
            if (block_dim > length_x) {
                block_dim = length_x;
            }
        }

        constexpr uint32_t BUFFER_NUM = 2;
        constexpr uint32_t TMP_BUFFER_NUM = 2;
        uint32_t tile_length = static_cast<uint32_t>(ub_size / dtype_size_x / (BUFFER_NUM + TMP_BUFFER_NUM));
        tile_length = tile_length / 32 * 32;
        if (tile_length == 0) {
            tile_length = 32;
        }
        if (tile_length > 8192) {
            tile_length = 8192;
        }

        FastGeluTilingData *tiling = context->GetTilingData<FastGeluTilingData>();
        tiling->length = length_x;
        tiling->blockLength = block_dim == 0 ? 0 : (length_x + block_dim - 1) / block_dim;
        tiling->tileLength = tile_length;

        context->SetBlockDim(block_dim);
        // 配置workspace大小
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

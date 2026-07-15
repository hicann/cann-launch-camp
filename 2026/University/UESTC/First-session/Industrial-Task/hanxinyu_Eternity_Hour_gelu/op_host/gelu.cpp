#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {

    constexpr uint32_t TILE_LENGTH = 7168;
    constexpr uint32_t MIN_PER_CORE = 2 * TILE_LENGTH;

    static ge::graphStatus TilingFunc(gert::TilingContext* context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();

        const gert::Tensor* tensor_input_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_input_x = tensor_input_x->GetDataType();

        uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

        // 获取总元素数（int64_t 避免截断）
        int64_t length = tensor_input_x->GetShapeSize();

        // 处理空张量：直接返回，kernel 会跳过
        if (length == 0) {
            GeluTilingData* tiling = context->GetTilingData<GeluTilingData>();
            tiling->length = 0;
            tiling->blockLength = 0;
            context->SetBlockDim(1);
            size_t* ws = context->GetWorkspaceSizes(1);
            ws[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        constexpr uint32_t CACHE_LINE = 64;
        uint32_t dtype_size = (dtype_input_x == ge::DT_FLOAT16) ? 2u : 4u;
        uint32_t align_len = CACHE_LINE / dtype_size;

        // 分配核数
        uint32_t block_dim = static_cast<uint32_t>(num_cores_aiv);
        while (block_dim > 1 && (static_cast<uint64_t>(length) / block_dim) < MIN_PER_CORE) {
            --block_dim;
        }

        // 均匀拆分并对齐
        uint64_t raw_block_len = (length + block_dim - 1) / block_dim;
        uint64_t block_len = ((raw_block_len + align_len - 1) / align_len) * align_len;

        // 剔除空闲核
        uint64_t effective = (length + block_len - 1) / block_len;
        if (block_dim > effective && effective > 0) {
            block_dim = static_cast<uint32_t>(effective);
        }
        if (block_dim < 1) {
            block_dim = 1;
        }

        GeluTilingData* tiling = context->GetTilingData<GeluTilingData>();
        tiling->length = static_cast<uint64_t>(length);
        tiling->blockLength = block_len;
        context->SetBlockDim(block_dim);

        size_t* ws = context->GetWorkspaceSizes(1);
        ws[0] = 0;

        return ge::GRAPH_SUCCESS;
    }

}  // namespace optiling

namespace ge {

    static graphStatus InferShape(gert::InferShapeContext* context) {
        *context->GetOutputShape(0) = *context->GetInputShape(0);
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext* context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }

}  // namespace ge

namespace ops {

    class Gelu : public OpDef {
    public:
        explicit Gelu(const char* name) : OpDef(name) {
            this->Input("input_x")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND });
            this->Output("output")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND });
            this->SetInferShape(ge::InferShape)
                .SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };

    OP_ADD(Gelu);

}  // namespace ops
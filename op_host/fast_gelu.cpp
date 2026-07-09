// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        // ========== 强制单核 ==========
        int32_t num_cores_aiv = 1;
        // int32_t num_cores_aiv = platform.GetCoreNumAiv();
        // =============================
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        int dtype_size_x = ge::GetSizeByDataType(dtype_x);
        uint32_t length_x = tensor_x->GetShapeSize();
        uint32_t size_x = tensor_x->GetSize();

        // 使用原始数据类型（不强制 float32）
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        FastGeluTilingData *tiling = context->GetTilingData<FastGeluTilingData>();

        tiling->length = length_x;
        tiling->block_num = num_cores_aiv;
        tiling->block_len = length_x / num_cores_aiv;
        tiling->remainder = length_x % num_cores_aiv;
        tiling->data_type = DT_X;

        tiling->tile_len = 32;
        tiling->tile_num = (tiling->block_len + tiling->tile_len - 1) / tiling->tile_len;
        if (tiling->tile_num == 0) tiling->tile_num = 1;

        tiling->tail_len = tiling->block_len - (tiling->tile_num - 1) * tiling->tile_len;
        if (tiling->tail_len == 0) tiling->tail_len = tiling->tile_len;

        context->SetBlockDim(num_cores_aiv);

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

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
                .Follow("x");
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(FastGelu);
}  // namespace ops
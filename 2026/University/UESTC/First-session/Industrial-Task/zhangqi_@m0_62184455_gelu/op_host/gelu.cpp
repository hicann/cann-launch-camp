#include "register/op_def_registry.h"

#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();

        const gert::Tensor *tensor_input_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_input_x = tensor_input_x->GetDataType();
        uint32_t length_input_x = tensor_input_x->GetShapeSize();

        uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

        GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();

        uint32_t coreNum = static_cast<uint32_t>(num_cores_aiv);
        if (coreNum == 0) {
            coreNum = 1;
        }

        // 固定 tile 大小, 按 totalLength 自然决定核数
        uint32_t tileLength = MAX_TILE_LENGTH;
        uint32_t tilesNeeded = (length_input_x + tileLength - 1) / tileLength;
        uint32_t blockNum = (tilesNeeded < coreNum) ? tilesNeeded : coreNum;
        if (blockNum < 1) blockNum = 1;

        // 每核分配量, 对齐到 32 元素
        uint32_t basePerCore = (length_input_x + blockNum - 1) / blockNum;
        constexpr uint32_t ALIGN = 32;
        uint32_t perCoreLength = ((basePerCore + ALIGN - 1) / ALIGN) * ALIGN;

        tiling->totalLength = length_input_x;
        tiling->perCoreLength = perCoreLength;
        tiling->tileLength = tileLength;

        context->SetBlockDim(blockNum);

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
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
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("output")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Gelu);
}

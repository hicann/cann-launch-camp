#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {

    static ge::graphStatus TilingFunc(gert::TilingContext* context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

        int32_t maxCores = platform.GetCoreNumAiv();

        const gert::Tensor* tensorInput = context->GetRequiredInputTensor(0);
        ge::DataType dtypeInput = tensorInput->GetDataType();

        uint32_t DT_INPUT_X = static_cast<uint32_t>(dtypeInput);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

        uint32_t totalLen = tensorInput->GetShapeSize();

        constexpr uint32_t TILE_SIZE = 4096;
        constexpr uint32_t ALIGN = 32;

        uint32_t idealCores = (totalLen + TILE_SIZE - 1) / TILE_SIZE;
        uint32_t dim = (idealCores < 1) ? 1 : idealCores;
        if (dim > static_cast<uint32_t>(maxCores)) {
            dim = static_cast<uint32_t>(maxCores);
        }

        uint32_t rawLen = (totalLen + dim - 1) / dim;
        uint32_t alignedLen = ((rawLen + ALIGN - 1) / ALIGN) * ALIGN;

        GeluTilingData* tiling = context->GetTilingData<GeluTilingData>();
        tiling->length = totalLen;
        tiling->blockLength = alignedLen;

        context->SetBlockDim(dim);

        size_t* workspace = context->GetWorkspaceSizes(1);
        workspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }

}  // namespace optiling

namespace ge {

    static graphStatus InferShape(gert::InferShapeContext* context) {
        const gert::Shape* inputShape = context->GetInputShape(0);
        gert::Shape* outputShape = context->GetOutputShape(0);
        *outputShape = *inputShape;
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext* context) {
        const auto inputDtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDtype);
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

}

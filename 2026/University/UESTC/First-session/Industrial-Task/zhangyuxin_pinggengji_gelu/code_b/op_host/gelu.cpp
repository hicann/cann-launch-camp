#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

#include <cstddef>
#include <cstdint>

namespace optiling {

constexpr uint32_t MIN_LENGTH_PER_CORE_FP16 = 2048;
constexpr uint32_t MIN_LENGTH_PER_CORE_FP32 = 1024;
constexpr uint32_t CORE_ALIGN_NUM = 16;

static uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

static uint32_t Min(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

static uint32_t AlignUp(uint32_t len, uint32_t align)
{
    return CeilDiv(len, align) * align;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::Tensor *tensorInputX = context->GetRequiredInputTensor(0);
    ge::DataType dtypeInputX = tensorInputX->GetDataType();
    uint32_t DT_INPUT_X = static_cast<uint32_t>(dtypeInputX);
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

    uint32_t totalLength = tensorInputX->GetShapeSize();
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    uint32_t coreNum = platform.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = 1;
    }

    uint32_t minLengthPerCore = (dtypeInputX == ge::DT_FLOAT) ? MIN_LENGTH_PER_CORE_FP32 : MIN_LENGTH_PER_CORE_FP16;

    uint32_t activeBlockNum = 1;
    uint32_t coreLength = totalLength;
    if (totalLength > minLengthPerCore) {
        uint32_t maxBlockByWork = CeilDiv(totalLength, minLengthPerCore);
        activeBlockNum = Min(coreNum, maxBlockByWork);
        while (activeBlockNum > 1) {
            coreLength = AlignUp(CeilDiv(totalLength, activeBlockNum), CORE_ALIGN_NUM);
            if (static_cast<uint64_t>(activeBlockNum - 1) * coreLength < totalLength) {
                break;
            }
            --activeBlockNum;
        }
        if (activeBlockNum == 1) {
            coreLength = totalLength;
        }
    }

    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
    tiling->totalLength = totalLength;
    tiling->activeBlockNum = activeBlockNum;
    tiling->coreLength = coreLength;

    context->SetBlockDim(activeBlockNum);
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Gelu : public OpDef {
public:
    explicit Gelu(const char *name) : OpDef(name)
    {
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

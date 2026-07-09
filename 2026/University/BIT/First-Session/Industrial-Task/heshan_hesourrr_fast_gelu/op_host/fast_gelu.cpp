#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

namespace {
constexpr uint32_t MIN_ELEMENTS_PER_CORE = 64;

static uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return (value + align - 1) / align * align;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::Tensor *xTensor = context->GetRequiredInputTensor(0);
    const ge::DataType xType = xTensor->GetDataType();
    const uint64_t length = static_cast<uint64_t>(xTensor->GetShapeSize());

    uint32_t DT_X = static_cast<uint32_t>(xType);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (coreNum == 0) {
        coreNum = 1;
    }

    const uint32_t typeSize = (xType == ge::DT_FLOAT16) ? 2U : 4U;
    const uint32_t alignNum = 32U / typeSize;

    uint32_t blockDim = 1;
    uint64_t blockLength = 0;
    if (length > 0) {
        uint64_t usefulCore = (length + MIN_ELEMENTS_PER_CORE - 1) / MIN_ELEMENTS_PER_CORE;
        blockDim = static_cast<uint32_t>(std::min<uint64_t>(coreNum, usefulCore));
        if (blockDim == 0) {
            blockDim = 1;
        }
        blockLength = AlignUp((length + blockDim - 1) / blockDim, alignNum);
    }

    context->SetBlockDim(blockDim);

    FastGeluTilingData *tiling = context->GetTilingData<FastGeluTilingData>();
    tiling->length = length;
    tiling->blockDim = blockDim;
    tiling->blockLength = blockLength;

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class FastGelu : public OpDef {
public:
    explicit FastGelu(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(FastGelu);
}

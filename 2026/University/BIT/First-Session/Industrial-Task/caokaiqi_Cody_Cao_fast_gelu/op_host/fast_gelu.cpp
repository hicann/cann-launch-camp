#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

namespace {
constexpr uint32_t kBlockBytes = 32;
constexpr uint32_t kMaxTileLength = 4096;
constexpr uint32_t kMinElementsPerCore = 1024;
constexpr uint32_t kSmallFastPathLength = 2048;

static uint32_t CeilDiv(uint32_t x, uint32_t y) {
    return (x + y - 1) / y;
}

static uint32_t AlignUp(uint32_t x, uint32_t align) {
    return CeilDiv(x, align) * align;
}

static uint32_t AlignDown(uint32_t x, uint32_t align) {
    return (x / align) * align;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t coreNum = platform.GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    const ge::DataType dtypeX = tensorX->GetDataType();
    const uint32_t dtypeSize = static_cast<uint32_t>(ge::GetSizeByDataType(dtypeX));
    const uint32_t alignElements = kBlockBytes / dtypeSize;
    const uint32_t length = static_cast<uint32_t>(tensorX->GetShapeSize());

    uint32_t usedCoreNum = 1;
    if (length > 0) {
        const uint32_t maxUsefulCores = CeilDiv(length, alignElements);
        usedCoreNum = static_cast<uint32_t>(coreNum) < maxUsefulCores
                          ? static_cast<uint32_t>(coreNum)
                          : maxUsefulCores;
        const uint32_t balancedCores = CeilDiv(length, kMinElementsPerCore);
        if (balancedCores < usedCoreNum) {
            usedCoreNum = balancedCores;
        }
        if (usedCoreNum == 0) {
            usedCoreNum = 1;
        }
    }

    const uint32_t blockLength =
        (length == 0) ? 0 : AlignUp(CeilDiv(length, usedCoreNum), alignElements);

    uint32_t tileLength = kMaxTileLength;
    if (ubSize > 0) {
        // The kernel uses 2 input buffers, 2 output buffers and 2 temp buffers.
        const uint32_t ubLimited =
            static_cast<uint32_t>(ubSize / (dtypeSize * 6));
        tileLength = ubLimited < kMaxTileLength ? ubLimited : kMaxTileLength;
        tileLength = AlignDown(tileLength, alignElements);
        if (tileLength < alignElements) {
            tileLength = alignElements;
        }
    }
    if (length <= kSmallFastPathLength && blockLength > 0 && blockLength < tileLength) {
        tileLength = blockLength;
    }

    const uint32_t DT_X = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    FastGeluTilingData *tiling = context->GetTilingData<FastGeluTilingData>();
    tiling->length = length;
    tiling->blockLength = blockLength;
    tiling->tileLength = tileLength;

    context->SetBlockDim(usedCoreNum);
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
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

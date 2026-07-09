#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

namespace {
constexpr uint32_t kMinBlockDim = 1;
constexpr uint32_t kElementsPerCoreHint = 256;
constexpr uint32_t kTileAlign = 64;
constexpr uint32_t kMaxTileSize = 16384;
constexpr uint32_t kBufferNum = 2;
constexpr uint32_t kFloatWorkBufferCount = 2;
constexpr uint32_t kUbUseNumerator = 31;
constexpr uint32_t kUbUseDenominator = 32;

uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

uint32_t AlignDown(uint32_t value, uint32_t align)
{
    return align == 0 ? value : value / align * align;
}

uint32_t MinU32(uint32_t lhs, uint32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

uint32_t MaxU32(uint32_t lhs, uint32_t rhs)
{
    return lhs > rhs ? lhs : rhs;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    ge::DataType dtypeX = context->GetInputDesc(0)->GetDataType();
    int32_t dtypeSize = ge::GetSizeByDataType(dtypeX);
    if (dtypeSize <= 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t length = static_cast<uint32_t>(context->GetInputShape(0)->GetOriginShape().GetShapeSize());

    uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (coreNum == 0) {
        coreNum = kMinBlockDim;
    }

    uint32_t blockDim = kMinBlockDim;
    if (length > 0) {
        uint32_t neededCores = CeilDiv(length, kElementsPerCoreHint);
        blockDim = MinU32(coreNum, MaxU32(kMinBlockDim, neededCores));
    }

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) {
        ubSize = 192 * 1024;
    }

    uint64_t queueBytesPerElement = static_cast<uint64_t>(dtypeSize) * kBufferNum * 2;
    uint64_t tmpBytesPerElement =
        dtypeX == ge::DT_FLOAT16 ? static_cast<uint64_t>(sizeof(float)) * kFloatWorkBufferCount : 0;
    uint64_t bytesPerElement = queueBytesPerElement + tmpBytesPerElement;
    uint32_t tileSize = static_cast<uint32_t>((ubSize * kUbUseNumerator / kUbUseDenominator) / bytesPerElement);
    tileSize = AlignDown(tileSize, kTileAlign);
    tileSize = MaxU32(kTileAlign, MinU32(tileSize, kMaxTileSize));
    if (length > 0) {
        tileSize = MinU32(tileSize, MaxU32(kTileAlign, length));
    }

    uint32_t DT_X = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    FastGeluTilingData *tiling = context->GetTilingData<FastGeluTilingData>();
    tiling->length = length;
    tiling->tileSize = tileSize;
    tiling->blockDim = blockDim;

    context->SetBlockDim(blockDim);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

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
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class FastGelu : public OpDef {
public:
    explicit FastGelu(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
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
OP_ADD(FastGelu);
}  // namespace ops

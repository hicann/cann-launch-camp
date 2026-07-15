#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace {
constexpr int64_t kUbReserve = 16 * 1024;
constexpr int64_t kCopyUnitBytes = 32;
constexpr int64_t kVecAlignElems = 64;
constexpr int64_t kFp16TileLimit = 8192;
constexpr int64_t kFp32TileLimit = 4096;
constexpr int64_t kSmallWorkPerCore = 512;
constexpr int64_t kLargeWorkPerCore = 768;
constexpr int64_t kMiddleWorkPerCore = 1024;
constexpr int64_t kLargeShapeStart = 8192;
constexpr int64_t kMiddleShapeEnd = 262144;
constexpr int64_t kPoly5ShapeStart = 262144;
constexpr size_t kWorkspaceCount = 1;

struct Plan {
    int64_t elements;
    int64_t coreSpan;
    int64_t tileSpan;
    int64_t coreCount;
    uint32_t usePoly5;
    uint32_t useExpForm;
};

static int64_t DivUp(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

static int64_t AlignUp(int64_t x, int64_t unit)
{
    return DivUp(x, unit) * unit;
}

static int64_t AlignDown(int64_t x, int64_t unit)
{
    return (x / unit) * unit;
}

static int64_t DataTypeBytes(ge::DataType dtype)
{
    return ge::GetSizeByDataType(dtype);
}

static int64_t VecAlignFor(ge::DataType dtype)
{
    int64_t byDma = kCopyUnitBytes / DataTypeBytes(dtype);
    return std::max<int64_t>(byDma, kVecAlignElems);
}

static int64_t PerCoreFloor(int64_t n)
{
    if (n > kLargeShapeStart && n <= kMiddleShapeEnd) {
        return kMiddleWorkPerCore;
    }
    return n > kLargeShapeStart ? kLargeWorkPerCore : kSmallWorkPerCore;
}

static int64_t FlattenLength(gert::TilingContext *ctx, const gert::Tensor *input)
{
    int64_t n = static_cast<int64_t>(input->GetShapeSize());
    auto shape = ctx->GetInputShape(0);
    if (shape != nullptr) {
        auto storage = shape->GetStorageShape();
        n = storage.GetDimNum() == 0 ? 1 : static_cast<int64_t>(storage.GetShapeSize());
    }
    return n;
}

static Plan MakePlan(int64_t n, ge::DataType dtype, int64_t hwCores, uint64_t ubBytes)
{
    Plan p;
    p.elements = n;
    p.coreSpan = 0;
    p.tileSpan = 0;
    p.coreCount = 1;
    p.usePoly5 = 0U;
    p.useExpForm = 0U;

    int64_t align = VecAlignFor(dtype);
    int64_t tileCap = (dtype == ge::DT_FLOAT) ? kFp32TileLimit : kFp16TileLimit;
    int64_t wantedCores = 1;
    if (n > 0) {
        wantedCores = std::min(hwCores, std::max<int64_t>(1, DivUp(n, PerCoreFloor(n))));
    }

    int64_t rawCoreSpan = (n == 0) ? 0 : DivUp(n, wantedCores);
    int64_t coreAlign = (rawCoreSpan > tileCap) ? tileCap : align;
    p.coreSpan = (n == 0) ? 0 : AlignUp(rawCoreSpan, coreAlign);
    p.coreCount = (n == 0) ? 1 : DivUp(n, p.coreSpan);
    p.coreCount = std::max<int64_t>(1, std::min(hwCores, p.coreCount));

    int64_t ubPerElem = DataTypeBytes(dtype) * 5;
    int64_t ubLimitedTile = AlignDown((static_cast<int64_t>(ubBytes) - kUbReserve) / ubPerElem, align);
    p.tileSpan = std::max<int64_t>(align, std::min<int64_t>(tileCap, ubLimitedTile));
    if (p.coreSpan > 0) {
        p.tileSpan = std::min(p.tileSpan, AlignUp(p.coreSpan, align));
        p.tileSpan = std::max<int64_t>(align, AlignDown(p.tileSpan, align));
    }

    p.usePoly5 = (n > kPoly5ShapeStart && p.coreSpan <= p.tileSpan) ? 1U : 0U;
    int64_t loops = (p.coreSpan + p.tileSpan - 1) / p.tileSpan;
    p.useExpForm = (dtype == ge::DT_FLOAT && p.usePoly5 == 0U && loops >= 4) ? 1U : 0U;
    return p;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t aivCount = platform.GetCoreNumAiv();
    if (aivCount <= 0) {
        aivCount = 1;
    }

    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    if (ubBytes <= kUbReserve) {
        return ge::GRAPH_FAILED;
    }

    const gert::Tensor *input = context->GetRequiredInputTensor(0);
    if (input == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType dtype = input->GetDataType();
    if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    int64_t total = FlattenLength(context, input);
    if (total < 0) {
        return ge::GRAPH_FAILED;
    }

    Plan plan = MakePlan(total, dtype, aivCount, ubBytes);
    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->total_elems = plan.elements;
    tiling->core_elems = plan.coreSpan;
    tiling->tile_elems = plan.tileSpan;

    context->SetBlockDim(static_cast<uint32_t>(plan.coreCount));
    size_t *workspace = context->GetWorkspaceSizes(kWorkspaceCount);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 0;

    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtype), plan.usePoly5, plan.useExpForm);
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Gelu : public OpDef {
public:
    explicit Gelu(const char *name) : OpDef(name)
    {
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(Gelu);
}

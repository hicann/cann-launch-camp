// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

#include "../op_kernel/clip_by_value_tiling.h"
#include "../op_kernel/tiling_key_clip_by_value.h"

namespace optiling {
namespace {

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    if (alignment == 0) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

uint64_t AlignDown(uint64_t value, uint64_t alignment)
{
    if (alignment == 0) {
        return value;
    }
    return value / alignment * alignment;
}

uint32_t ChooseBlockDim(uint64_t length, uint32_t coreLimit)
{
    uint32_t blocks;
    if (length <= 8192) {
        blocks = 1;
    } else if (length <= 16384) {
        blocks = 2;
    } else if (length <= 49152) {
        blocks = 4;
    } else if (length <= 131072) {
        blocks = 8;
    } else if (length <= 393216) {
        blocks = 12;
    } else if (length <= 786432) {
        blocks = 16;
    } else if (length <= 1572864) {
        blocks = 24;
    } else if (length <= 3145728) {
        blocks = 32;
    } else {
        blocks = 40;
    }
    return std::max(1U, std::min(blocks, coreLimit));
}

}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::Tensor *x = context->GetRequiredInputTensor(0);
    const gert::Tensor *clipMin = context->GetRequiredInputTensor(1);
    const gert::Tensor *clipMax = context->GetRequiredInputTensor(2);
    if (x == nullptr || clipMin == nullptr || clipMax == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtype = x->GetDataType();
    const int32_t dtypeSize = ge::GetSizeByDataType(dtype);
    if (dtypeSize <= 0) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t totalLength = x->GetShapeSize();
    const uint64_t minLength = clipMin->GetShapeSize();
    const uint64_t maxLength = clipMax->GetShapeSize();
    const bool minIsScalar = (minLength == 1);
    const bool maxIsScalar = (maxLength == 1);

    // 本题只要求标量或与 x 同形状；提前拒绝其余形状，防止越界访问。
    if ((!minIsScalar && minLength != totalLength) ||
        (!maxIsScalar && maxLength != totalLength)) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t dtypeKey = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, dtypeKey);

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreLimit = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (coreLimit == 0) {
        coreLimit = 1;
    }

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    ClipByValueTilingData *tiling = context->GetTilingData<ClipByValueTilingData>();
    tiling->totalLength = totalLength;
    tiling->minIsScalar = minIsScalar ? 1U : 0U;
    tiling->maxIsScalar = maxIsScalar ? 1U : 0U;

    const uint32_t alignElems = 32U / static_cast<uint32_t>(dtypeSize);
    const uint32_t preferredElems = 512U / static_cast<uint32_t>(dtypeSize);
    tiling->alignElems = std::max(1U, alignElems);

    if (totalLength == 0) {
        tiling->blockDim = 1;
        tiling->lengthPerCore = 0;
        tiling->tileLength = std::max(1U, preferredElems);
        context->SetBlockDim(1);
        context->GetWorkspaceSizes(1)[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    uint32_t blockDim = ChooseBlockDim(totalLength, coreLimit);
    blockDim = static_cast<uint32_t>(
        std::min<uint64_t>(blockDim, totalLength));

    uint64_t lengthPerCore = AlignUp(
        (totalLength + blockDim - 1) / blockDim, preferredElems);
    const uint32_t effectiveBlockDim = static_cast<uint32_t>(
        (totalLength + lengthPerCore - 1) / lengthPerCore);

    // x/y 总是双缓冲；非标量 min/max 各自再占一组双缓冲。
    uint32_t bufferCount = 4U;
    if (!minIsScalar) {
        bufferCount += 2U;
    }
    if (!maxIsScalar) {
        bufferCount += 2U;
    }

    // 给队列控制信息和编译器临时空间留出余量。
    const uint64_t usableUb = ubSize > 8192U ? ubSize - 8192U : ubSize;
    uint64_t tileCapacity = usableUb /
        (static_cast<uint64_t>(bufferCount) * static_cast<uint32_t>(dtypeSize));
    tileCapacity = AlignDown(tileCapacity, preferredElems);
    tileCapacity = std::min<uint64_t>(tileCapacity, 32768U);
    if (tileCapacity == 0) {
        tileCapacity = preferredElems;
    }

    uint64_t tileLength = std::min(lengthPerCore, tileCapacity);
    tileLength = AlignDown(tileLength, preferredElems);
    if (tileLength == 0) {
        tileLength = preferredElems;
    }

    tiling->blockDim = effectiveBlockDim;
    tiling->lengthPerCore = lengthPerCore;
    tiling->tileLength = static_cast<uint32_t>(tileLength);

    context->SetBlockDim(effectiveBlockDim);
    context->GetWorkspaceSizes(1)[0] = 0;
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
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class ClipByValue : public OpDef {
public:
    explicit ClipByValue(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("clip_value_min")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("clip_value_max")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(ClipByValue);

}  // namespace ops

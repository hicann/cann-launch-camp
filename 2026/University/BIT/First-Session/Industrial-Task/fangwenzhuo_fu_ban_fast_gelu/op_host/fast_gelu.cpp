// Host侧Tiling实现
#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

namespace {
constexpr uint64_t MIN_ELEMS_PER_CORE = 128;
constexpr uint64_t UB_RESERVED_BYTES = 8192;
constexpr uint64_t BUFFER_BYTES_PER_ELEM = 16;
constexpr uint64_t GM_ALIGN_BYTES = 32;
constexpr uint64_t UB_ALIGN_BYTES = 256;

uint64_t CeilDiv(uint64_t value, uint64_t divisor) {
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

uint64_t AlignUp(uint64_t value, uint64_t align) {
    return align == 0 ? value : CeilDiv(value, align) * align;
}

uint64_t CalcAlignElemNum(uint32_t dtypeSize) {
    return std::max<uint64_t>(1, UB_ALIGN_BYTES / std::max<uint32_t>(1, dtypeSize));
}

uint64_t CalcGmAlignElemNum(uint32_t dtypeSize) {
    return std::max<uint64_t>(1, GM_ALIGN_BYTES / std::max<uint32_t>(1, dtypeSize));
}

uint32_t CalcTileLength(uint64_t ubSize, uint32_t dtypeSize) {
    const uint64_t usableUbSize = ubSize > UB_RESERVED_BYTES ? ubSize - UB_RESERVED_BYTES : ubSize;
    const uint64_t maxElemNum = std::max<uint64_t>(1, usableUbSize / BUFFER_BYTES_PER_ELEM);
    const uint64_t alignElemNum = CalcAlignElemNum(dtypeSize);
    uint64_t tileLength = (maxElemNum / alignElemNum) * alignElemNum;
    if (tileLength == 0) {
        tileLength = maxElemNum;
    }
    return static_cast<uint32_t>(std::min<uint64_t>(tileLength, UINT32_MAX));
}

uint32_t CalcCoreNum(uint64_t totalLength, uint32_t availableCoreNum) {
    if (availableCoreNum == 0 || totalLength == 0) {
        return 1;
    }

    const uint64_t coreNum = std::min<uint64_t>(availableCoreNum, CeilDiv(totalLength, MIN_ELEMS_PER_CORE));
    return static_cast<uint32_t>(std::max<uint64_t>(1, coreNum));
}

uint64_t CalcBlockLength(uint64_t totalLength, uint32_t coreNum, uint32_t dtypeSize) {
    if (totalLength == 0 || coreNum == 0) {
        return 0;
    }
    return AlignUp(CeilDiv(totalLength, coreNum), CalcGmAlignElemNum(dtypeSize));
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t numCoresAiv = static_cast<uint32_t>(std::max<int32_t>(1, platform.GetCoreNumAiv()));
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    if (tensorX == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtypeX = tensorX->GetDataType();
    const int64_t shapeSize = tensorX->GetShapeSize();
    if (shapeSize < 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t dtypeSize = static_cast<uint32_t>(std::max<int64_t>(1, ge::GetSizeByDataType(dtypeX)));
    const uint64_t totalLength = static_cast<uint64_t>(shapeSize);
    const uint32_t tileLength = CalcTileLength(ubSize, dtypeSize);
    const uint32_t coreNum = CalcCoreNum(totalLength, numCoresAiv);
    const uint64_t blockLength = CalcBlockLength(totalLength, coreNum, dtypeSize);

    const uint32_t DT_X = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    FastGeluTilingData *tiling = context->GetTilingData<FastGeluTilingData>();
    tiling->totalLength = totalLength;
    tiling->blockLength = blockLength;
    tiling->tileLength = tileLength;
    tiling->coreNum = coreNum;

    context->SetBlockDim(coreNum);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr) {
        return GRAPH_FAILED;
    }
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const ge::DataType xDtype = context->GetInputDataType(0);
    if (xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_FLOAT) {
        return GRAPH_FAILED;
    }
    return context->SetOutputDataType(0, xDtype);
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

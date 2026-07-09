// Host-side registration, shape/type inference and tiling for FastGelu.
#include <algorithm>
#include <cstdint>
#include <limits>

#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

namespace {
constexpr uint32_t kBlockBytes = 32;
constexpr uint32_t kBufferNum = 1;
constexpr uint32_t kTmpTensorNum = 1;
constexpr uint32_t kTensorSlotNum = 2 * kBufferNum + kTmpTensorNum;
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t coreNumAiv = platform.GetCoreNumAiv();
    uint32_t blockDim = coreNumAiv > 0 ? static_cast<uint32_t>(coreNumAiv) : 1U;
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    ge::DataType dtypeX = tensorX->GetDataType();
    int64_t shapeSize = tensorX->GetShapeSize();
    uint64_t length = shapeSize > 0 ? static_cast<uint64_t>(shapeSize) : 0U;
    uint32_t dtypeBytes = 0;
    ge::TypeUtils::GetDataTypeLength(dtypeX, dtypeBytes);
    if (dtypeBytes == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t blockElements = std::max<uint32_t>(kBlockBytes / dtypeBytes, 1U);

    uint32_t DT_X = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    auto *tiling = context->GetTilingData<FastGeluTilingData>();
    tiling->length = length;
    tiling->blockElements = blockElements;

    uint64_t tileBlockNum = (ubSize / kBlockBytes) / kTensorSlotNum;
    tileBlockNum = std::max<uint64_t>(tileBlockNum, 1U);
    uint64_t maxTileBlockNum =
        std::numeric_limits<uint32_t>::max() / static_cast<uint64_t>(blockElements);
    tileBlockNum = std::min<uint64_t>(tileBlockNum, maxTileBlockNum);
    tiling->tileDataNum = static_cast<uint32_t>(tileBlockNum * blockElements);

    if (length == 0) {
        blockDim = 1;
    } else {
        uint64_t usefulBlocks = (length + blockElements - 1) / blockElements;
        blockDim = static_cast<uint32_t>(std::min<uint64_t>(blockDim, usefulBlocks));
        blockDim = std::max<uint32_t>(blockDim, 1U);
    }
    context->SetBlockDim(blockDim);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
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
    ge::DataType dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, dtype);
    return ge::GRAPH_SUCCESS;
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

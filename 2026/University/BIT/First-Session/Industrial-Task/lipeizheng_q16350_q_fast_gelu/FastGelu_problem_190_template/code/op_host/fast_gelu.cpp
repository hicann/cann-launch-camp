// FastGelu Host 侧：算子注册、Shape/类型推导及 Tiling。
#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

namespace optiling {
namespace {
constexpr uint32_t DATA_BLOCK_BYTES = 32U;
constexpr uint32_t BUFFER_FACTOR = 4U;      // VECIN 双缓冲 + VECOUT 双缓冲
constexpr uint32_t UB_SAFETY_DIVISOR = 2U; // 仅使用约一半 UB，给框架及对齐留余量
constexpr uint32_t MAX_TILE_ELEMENTS = 16384U;
constexpr uint32_t MIN_ELEMENTS_PER_CORE = 4096U;

inline uint64_t CeilDiv(uint64_t value, uint64_t divisor) {
    return (value + divisor - 1U) / divisor;
}

inline uint64_t AlignDown(uint64_t value, uint64_t alignment) {
    return value / alignment * alignment;
}
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    int32_t availableCores = platform.GetCoreNumAiv();
    if (availableCores <= 0) {
        availableCores = 1;
    }

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    const ge::DataType dtypeX = tensorX->GetDataType();
    const int32_t dtypeSize = ge::GetSizeByDataType(dtypeX);
    if (dtypeSize <= 0) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t totalLength = static_cast<uint64_t>(tensorX->GetShapeSize());
    if (totalLength == 0U) {
        return ge::GRAPH_FAILED;
    }

    // 按数据规模选择实际启动核数，避免小张量启动过多核心。
    uint64_t desiredCores = CeilDiv(totalLength, MIN_ELEMENTS_PER_CORE);
    desiredCores = std::max<uint64_t>(1U, desiredCores);
    const uint32_t usedCores = static_cast<uint32_t>(
        std::min<uint64_t>(desiredCores, static_cast<uint64_t>(availableCores)));

    const uint64_t blockLength = CeilDiv(totalLength, usedCores);

    // 一个 Tile 同时需要两块输入队列内存和两块输出队列内存。
    // 只使用约一半 UB，降低不同 CANN 版本下资源不足的风险。
    const uint32_t alignElements = DATA_BLOCK_BYTES / static_cast<uint32_t>(dtypeSize);
    uint64_t tileElementsByUb = ubSize /
        (BUFFER_FACTOR * UB_SAFETY_DIVISOR * static_cast<uint64_t>(dtypeSize));
    tileElementsByUb = AlignDown(tileElementsByUb, alignElements);

    uint32_t tileLength = static_cast<uint32_t>(
        std::min<uint64_t>(tileElementsByUb, MAX_TILE_ELEMENTS));
    if (tileLength < alignElements) {
        tileLength = alignElements;
    }

    const uint32_t dtX = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, dtX);

    FastGeluTilingData *tiling = context->GetTilingData<FastGeluTilingData>();
    tiling->totalLength = totalLength;
    tiling->blockLength = blockLength;
    tiling->tileLength = tileLength;
    tiling->reserved = 0U;

    context->SetBlockDim(usedCores);
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0U;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    *outputShape = *inputShape;
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
        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(FastGelu);
}  // namespace ops

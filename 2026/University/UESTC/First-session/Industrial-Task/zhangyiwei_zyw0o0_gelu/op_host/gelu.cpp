// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t ALIGN_NUM = 8;
constexpr int64_t MIN_UB_FACTOR = 32;
constexpr int64_t BUFFER_NUM = 2;

static int64_t FloorAlign(int64_t value, int64_t align)
{
    return value / align * align;
}

static int64_t CeilAlign(int64_t value, int64_t align)
{
    return (value + align - 1) / align * align;
}

static int64_t CeilDiv(int64_t value, int64_t factor)
{
    return (value + factor - 1) / factor;
}

static int64_t AlignNum(int64_t dtypeSize)
{
    return 512 / dtypeSize;  // 910B 最优 DMA 对齐 512B (32B 仅 70% 带宽)
}

static int64_t ErfTmpBytesPerElem(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT ? 16 : 16;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t coreNum = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (coreNum <= 0 || ubSize == 0) {
        return ge::GRAPH_FAILED;
    }

    const gert::Tensor *inputTensor = context->GetRequiredInputTensor(0);
    if (inputTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType dtype = inputTensor->GetDataType();
    if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    int64_t totalNum = static_cast<int64_t>(inputTensor->GetShapeSize());
    if (totalNum <= 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

    int64_t dtypeSize = static_cast<int64_t>(ge::GetSizeByDataType(dtype));
    int64_t alignNum = AlignNum(dtypeSize);
    int64_t maxUsefulBlocks = CeilDiv(totalNum, alignNum);

    // 冲刺版: 2 个 dtypeSize 的 temp (fp16 原生=2B, fp32=4B)
    int64_t bytesPerElem = dtypeSize * (2 * BUFFER_NUM + 2);
    int64_t maxUbFactor = FloorAlign(static_cast<int64_t>(ubSize) / bytesPerElem, alignNum);
    if (maxUbFactor < alignNum) {
        maxUbFactor = alignNum;
    }

    // [Tiling] blockDim 用满核数 (最多到有效块数); blockFactor 对齐
    // 注: 曾尝试"小 shape 减核增 DMA 块", 本地提速但竞赛 t3/t4 灾难性退步 (shape 不一致), 故回退保守
    int64_t blockDim = coreNum;
    if (blockDim > maxUsefulBlocks) {
        blockDim = maxUsefulBlocks;
    }
    if (blockDim < 1) {
        blockDim = 1;
    }

    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, blockDim), alignNum);
    if (blockFactor < alignNum) {
        blockFactor = alignNum;
    }
    blockDim = CeilDiv(totalNum, blockFactor);
    if (blockDim > coreNum) {
        blockDim = coreNum;
    }
    if (blockDim < 1) {
        blockDim = 1;
    }

    int64_t ubFactor = blockFactor < maxUbFactor ? blockFactor : maxUbFactor;
    ubFactor = FloorAlign(ubFactor, alignNum);
    if (ubFactor < alignNum) {
        ubFactor = alignNum;
    }

    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    tiling->erfTmpBytes = 0;  // 未使用 (手写多项式不需要)
    tiling->reserved = 0;

    context->SetBlockDim(blockDim);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

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
    ge::DataType inputDtype = context->GetInputDataType(0);
    if (inputDtype == ge::DT_UNDEFINED) {
        return ge::GRAPH_FAILED;
    }
    return context->SetOutputDataType(0, inputDtype);
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
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Gelu);
}  // namespace ops

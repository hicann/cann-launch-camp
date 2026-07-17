#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace {
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_BUFFER_PARTS = 6;

uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return ((value + align - 1) / align) * align;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const ge::DataType dtypeInputX = context->GetInputDesc(0)->GetDataType();
    uint32_t inputNum = static_cast<uint32_t>(inputShape->GetStorageShape().GetShapeSize());
    if (inputNum == 0) {
        inputNum = static_cast<uint32_t>(inputShape->GetOriginShape().GetShapeSize());
    }
    const uint32_t typeLength = static_cast<uint32_t>(ge::GetSizeByDataType(dtypeInputX));
    const uint32_t inputLength = inputNum * typeLength;
    const uint32_t inputLengthAligned = AlignUp(inputLength, BLOCK_SIZE);
    const uint32_t totalBlockNum = inputLengthAligned / BLOCK_SIZE;

    coreNum = std::min(coreNum, totalBlockNum);
    coreNum = std::max(coreNum, static_cast<uint32_t>(1));
    const uint32_t everyCoreInputBlockNum = totalBlockNum / coreNum;
    const uint32_t tailBlockNum = totalBlockNum % coreNum;
    context->SetBlockDim(coreNum);

    uint32_t tileBlockNum = static_cast<uint32_t>((ubSize / BLOCK_SIZE / BUFFER_NUM) / UB_BUFFER_PARTS);
    tileBlockNum = std::max(tileBlockNum, static_cast<uint32_t>(1));
    const uint32_t tileLength = (tileBlockNum * BLOCK_SIZE) / typeLength;

    const uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    const uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    const uint32_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint32_t smallTailDataNum = smallCoreDataNum - tileLength * smallTileNum;
    smallTailDataNum = smallTailDataNum == 0 ? tileLength : smallTailDataNum;

    const uint32_t bigCoreInputBlockNum = everyCoreInputBlockNum + 1;
    const uint32_t bigCoreDataNum = bigCoreInputBlockNum * BLOCK_SIZE / typeLength;
    const uint32_t bigTileNum = bigCoreInputBlockNum / tileBlockNum;
    const uint32_t finalBigTileNum = (bigCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint32_t bigTailDataNum = bigCoreDataNum - tileLength * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileLength : bigTailDataNum;

    const uint32_t DT_INPUT_X = static_cast<uint32_t>(dtypeInputX);
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
    tiling->totalLength = inputNum;
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->tileLength = tileLength;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->tailBlockNum = tailBlockNum;

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
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

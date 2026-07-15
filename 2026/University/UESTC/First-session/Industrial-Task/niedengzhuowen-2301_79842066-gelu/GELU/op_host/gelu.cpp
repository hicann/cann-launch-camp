// Host-side tiling implementation for GELU.
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "graph/utils/type_utils.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const gert::Tensor *tensorInput = context->GetRequiredInputTensor(0);
    ge::DataType dtypeInput = tensorInput->GetDataType();
    int dtypeSize = ge::GetSizeByDataType(dtypeInput);
    uint32_t lengthInput = tensorInput->GetShapeSize();

    if (lengthInput == 0) {
        lengthInput = 1;
    }
    uint32_t inputLength = lengthInput * static_cast<uint32_t>(dtypeSize);

    uint32_t DT_INPUT_X = static_cast<uint32_t>(dtypeInput);
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

    const uint32_t BLOCK_SIZE = 32;
    uint32_t inputLengthAlign32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    coreNum = coreNum < inputLengthAlign32 / BLOCK_SIZE ? coreNum : inputLengthAlign32 / BLOCK_SIZE;
    coreNum = coreNum > static_cast<uint32_t>(1) ? coreNum : static_cast<uint32_t>(1);

    uint32_t everyCoreInputBlockNum = inputLengthAlign32 / BLOCK_SIZE / coreNum;
    uint32_t tailBlockNum = (inputLengthAlign32 / BLOCK_SIZE) % coreNum;
    context->SetBlockDim(coreNum);

    std::vector<int64_t> shapeVec = {lengthInput};
    ge::Shape srcShape(shapeVec);
    uint32_t minTmpSize = AscendC::GetGeluMinTmpSize(srcShape, sizeof(float));
    ubSize -= minTmpSize;

    uint32_t ubDataNumber = dtypeInput == ge::DT_FLOAT ? 7 : 8;
    uint32_t tileBlockNum = (ubSize / BLOCK_SIZE) / ubDataNumber;
    uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / static_cast<uint32_t>(dtypeSize);

    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / static_cast<uint32_t>(dtypeSize);
    uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    uint32_t bigCoreInputBlockNum = everyCoreInputBlockNum + 1;
    uint32_t bigCoreDataNum = bigCoreInputBlockNum * BLOCK_SIZE / static_cast<uint32_t>(dtypeSize);
    uint32_t bigTileNum = bigCoreInputBlockNum / tileBlockNum;
    uint32_t finalBigTileNum = (bigCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint32_t bigTailDataNum = bigCoreDataNum - (tileDataNum * bigTileNum);
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->finalBigTileNum = finalBigTileNum;
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

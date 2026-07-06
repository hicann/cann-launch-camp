#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cstdint>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNum();

    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);

    const uint32_t BLOCK_SIZE = 32;
    uint32_t inputLength = inputNum * typeLength;
    uint32_t inputLengthAlign32 = ((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    uint32_t totalBlockNum = inputLengthAlign32 / BLOCK_SIZE;

    coreNum = std::min(coreNum, totalBlockNum);
    coreNum = std::max(coreNum, static_cast<uint32_t>(1));

    uint32_t everyCoreInputBlockNum = totalBlockNum / coreNum;
    uint32_t tailBlockNum = totalBlockNum % coreNum;
    context->SetBlockDim(coreNum);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const uint32_t BUFFER_NUM = 2;
    const uint32_t QUEUE_COUNT = 2;        // x, y
    const uint32_t TMP_FLOAT_COUNT = 2;    // 中间 float 缓冲区

    uint32_t bytesPerElementInUb =
        QUEUE_COUNT * BUFFER_NUM * typeLength + TMP_FLOAT_COUNT * sizeof(float);

    uint32_t maxElementsByUb = static_cast<uint32_t>(ubSize / bytesPerElementInUb);
    uint32_t elementsPerBlock = BLOCK_SIZE / typeLength;
    uint32_t tileDataNum = (maxElementsByUb / elementsPerBlock) * elementsPerBlock;
    tileDataNum = std::max(tileDataNum, elementsPerBlock);

    uint32_t tileBlockNum = tileDataNum * typeLength / BLOCK_SIZE;
    tileBlockNum = std::max(tileBlockNum, static_cast<uint32_t>(1));

    // 小核参数
    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum =
        (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    finalSmallTileNum = std::max(finalSmallTileNum, static_cast<uint32_t>(1));
    uint32_t smallTailDataNum = smallCoreDataNum - tileDataNum * smallTileNum;
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    // 大核参数（多 1 个块）
    everyCoreInputBlockNum += 1;
    uint32_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalBigTileNum =
        (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    finalBigTileNum = std::max(finalBigTileNum, static_cast<uint32_t>(1));
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    LogSigmoidCustomTilingData *tiling = context->GetTilingData<LogSigmoidCustomTilingData>();
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->tailBlockNum = tailBlockNum;

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class LogSigmoidCustom : public OpDef {
public:
    explicit LogSigmoidCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(LogSigmoidCustom);
}

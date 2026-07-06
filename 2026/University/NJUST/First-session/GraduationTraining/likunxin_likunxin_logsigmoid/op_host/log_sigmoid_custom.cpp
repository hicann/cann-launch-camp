#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    uint32_t coreNum = ascendcPlatform.GetCoreNum();

    uint32_t inputNum =
        context->GetInputShape(0)
            ->GetStorageShape()
            .GetShapeSize();

    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(
        context->GetInputDesc(0)->GetDataType(),
        typeLength);

    const uint32_t BLOCK_SIZE = 32;
    const uint32_t BUFFER_NUM = 2;

    uint32_t inputLength = inputNum * typeLength;

    uint32_t inputLengthAlign32 =
        ((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

    uint32_t totalBlockNum = inputLengthAlign32 / BLOCK_SIZE;

    coreNum = std::min(coreNum, totalBlockNum);
    coreNum = std::max(coreNum, static_cast<uint32_t>(1));

    context->SetBlockDim(coreNum);

    uint32_t everyCoreInputBlockNum = totalBlockNum / coreNum;
    uint32_t tailBlockNum = totalBlockNum % coreNum;

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);

    auto dtype = context->GetInputDesc(0)->GetDataType();

    /*
     * UB 占用模型：
     *
     * float / half:
     *   input queue  : BUFFER_NUM * typeLength
     *   output queue : BUFFER_NUM * typeLength
     *   tmp0/tmp1    : 2 * typeLength
     *
     * bf16:
     *   input queue  : BUFFER_NUM * 2B
     *   output queue : BUFFER_NUM * 2B
     *   xFloat/tmp0/tmp1 : 3 * 4B
     *
     * 这比固定 tileDataNum 更像工程写法：tile 大小由 UB 容量和实际临时区需求决定。
     */
    uint32_t bytesPerElement = 0;

    if (dtype == ge::DT_BF16) {
        bytesPerElement =
            BUFFER_NUM * typeLength +
            BUFFER_NUM * typeLength +
            3 * sizeof(float);
    } else {
        bytesPerElement =
            BUFFER_NUM * typeLength +
            BUFFER_NUM * typeLength +
            2 * typeLength;
    }

    uint32_t maxTileDataNum =
        static_cast<uint32_t>(ubSize / bytesPerElement);

    uint32_t alignElementNum = BLOCK_SIZE / typeLength;

    uint32_t tileDataNum =
        (maxTileDataNum / alignElementNum) * alignElementNum;

    tileDataNum = std::max(tileDataNum, alignElementNum);

    uint32_t tileBlockNum =
        tileDataNum * typeLength / BLOCK_SIZE;

    tileBlockNum = std::max(tileBlockNum, static_cast<uint32_t>(1));

    // 小核参数
    uint32_t smallCoreDataNum =
        everyCoreInputBlockNum * BLOCK_SIZE / typeLength;

    uint32_t smallTileNum =
        everyCoreInputBlockNum / tileBlockNum;

    uint32_t finalSmallTileNum =
        (everyCoreInputBlockNum % tileBlockNum) == 0
            ? smallTileNum
            : smallTileNum + 1;

    uint32_t smallTailDataNum =
        smallCoreDataNum - tileDataNum * smallTileNum;

    smallTailDataNum =
        smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    // 大核参数：前 tailBlockNum 个核多处理一个 32B block
    uint32_t bigCoreBlockNum = everyCoreInputBlockNum + 1;

    uint32_t bigCoreDataNum =
        bigCoreBlockNum * BLOCK_SIZE / typeLength;

    uint32_t bigTileNum =
        bigCoreBlockNum / tileBlockNum;

    uint32_t finalBigTileNum =
        (bigCoreBlockNum % tileBlockNum) == 0
            ? bigTileNum
            : bigTileNum + 1;

    uint32_t bigTailDataNum =
        bigCoreDataNum - tileDataNum * bigTileNum;

    bigTailDataNum =
        bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    LogSigmoidCustomTilingData* tiling =
        context->GetTilingData<LogSigmoidCustomTilingData>();

    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->tailBlockNum = tailBlockNum;

    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = 0;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
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
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(LogSigmoidCustom);
}

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

    // 09.01 的 x 既有 2 维也有 3 维，这里不能沿用固定二维 shape 假设。
    // ND 张量在 GM 上按连续一维数据处理，元素总数必须直接来自完整 StorageShape。
    uint64_t inputNum = static_cast<uint64_t>(context->GetInputShape(0)->GetStorageShape().GetShapeSize());

    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    const uint32_t BLOCK_SIZE = 32;

    uint64_t inputLength = inputNum * typeLength;
    uint64_t inputLengthAlign32 = ((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    uint32_t inputBlockNum = static_cast<uint32_t>(inputLengthAlign32 / BLOCK_SIZE);

    coreNum = std::min(coreNum, inputBlockNum);
    coreNum = std::max(coreNum, static_cast<uint32_t>(1));

    uint32_t everyCoreInputBlockNum = inputBlockNum / coreNum;
    uint32_t tailBlockNum = inputBlockNum % coreNum;
    context->SetBlockDim(coreNum);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint32_t BUFFER_NUM = 2;

    // LogSigmoid 对 half/BF16 统一转 float 做 Exp/Ln，再 Cast 回原类型；float 可原地按 float 计算。
    uint32_t ubDataNumber = (context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT) ? 2 : 4;
    uint32_t tileBlockNum = static_cast<uint32_t>((ubSize / BLOCK_SIZE / BUFFER_NUM) / ubDataNumber);
    tileBlockNum = std::max(tileBlockNum, static_cast<uint32_t>(1));
    uint32_t tileDataNum = tileBlockNum * BLOCK_SIZE / typeLength;

    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint32_t smallTailDataNum = smallCoreDataNum - tileDataNum * smallTileNum;
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    // 前 tailBlockNum 个核多分到一个 32B block，称为 big core。
    uint32_t bigCoreInputBlockNum = everyCoreInputBlockNum + 1;
    uint32_t bigCoreDataNum = bigCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t bigTileNum = bigCoreInputBlockNum / tileBlockNum;
    uint32_t finalBigTileNum = (bigCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    LogSigmoidCustomTilingData *tiling = context->GetTilingData<LogSigmoidCustomTilingData>();
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->tailBlockNum = tailBlockNum;
    return ge::GRAPH_SUCCESS;
}
}

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
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(LogSigmoidCustom);
}

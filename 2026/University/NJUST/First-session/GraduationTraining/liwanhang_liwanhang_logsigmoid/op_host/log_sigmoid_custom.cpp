#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 获取硬件平台信息与可用AI Core数
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNum();

    // 计算输入总元素数与单元素字节数
    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint32_t inputLength = inputNum * typeLength;

    // 按32B对齐（Ascend C数据搬运最小粒度）
    const uint32_t BLOCK_SIZE = 32;
    uint32_t totalBlocks = (inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 动态调整实际使用的Core数，最少1个
    coreNum = std::min(coreNum, totalBlocks);
    coreNum = std::max(coreNum, 1u);
    context->SetBlockDim(coreNum);

    // 计算每Core分配的32B块数，余数块由前tailBlockNum个Core多承担1块
    uint32_t everyBlockNum = totalBlocks / coreNum;
    uint32_t tailBlockNum = totalBlocks % coreNum;

    // 获取单Core的UB总大小，保守计算Tile大小（仅用50% UB，避免溢出）
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint32_t BUFFER_NUM = 2;

    // 单Tile总字节开销：输入队列(2份) + 输出队列(2份) + 2个float临时计算缓冲
    uint64_t perTileTotalBytes = 2 * typeLength + 2 * typeLength + 2 * sizeof(float);
    perTileTotalBytes *= BUFFER_NUM; // 双缓冲翻倍
    // 额外预留30%对齐与管道开销
    uint64_t availableUb = ubSize * 5 / 10;
    uint32_t tileDataNum = static_cast<uint32_t>(availableUb / perTileTotalBytes);
    // Tile大小对齐到32B对应的元素数
    uint32_t elemsPerBlock = BLOCK_SIZE / typeLength;
    tileDataNum = (tileDataNum / elemsPerBlock) * elemsPerBlock;
    tileDataNum = std::max(tileDataNum, elemsPerBlock);

    // 普通Core（small）：处理 everyBlockNum 块
    uint32_t smallCoreDataNum = everyBlockNum * BLOCK_SIZE / typeLength;
    uint32_t smallTileNum = smallCoreDataNum / tileDataNum;
    uint32_t finalSmallTileNum = (smallCoreDataNum % tileDataNum == 0) ? smallTileNum : smallTileNum + 1;
    uint32_t smallTailDataNum = smallCoreDataNum - smallTileNum * tileDataNum;
    smallTailDataNum = (smallTailDataNum == 0) ? tileDataNum : smallTailDataNum;

    // 大Core（big）：处理 everyBlockNum + 1 块
    uint32_t bigCoreDataNum = (everyBlockNum + 1) * BLOCK_SIZE / typeLength;
    uint32_t bigTileNum = bigCoreDataNum / tileDataNum;
    uint32_t finalBigTileNum = (bigCoreDataNum % tileDataNum == 0) ? bigTileNum : bigTileNum + 1;
    uint32_t bigTailDataNum = bigCoreDataNum - bigTileNum * tileDataNum;
    bigTailDataNum = (bigTailDataNum == 0) ? tileDataNum : bigTailDataNum;

    // 写入Tiling参数
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

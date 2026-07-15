// Host侧Tiling实现 - 大shape优化版
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"
#include "graph/utils/type_utils.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto maxCoreNum = ascendcPlatform.GetCoreNum();

    uint64_t inputNum = static_cast<uint64_t>(context->GetInputShape(0)->GetStorageShape().GetShapeSize());
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint64_t inputLength = static_cast<uint64_t>(inputNum) * typeLength;
    const uint32_t BLOCK_SIZE = 32;
    const uint64_t L2_CACHE_SIZE = 192 * 1024 * 1024;  // 192MB

    uint32_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    uint32_t totalBlocks = inputLengthAlgin32 / BLOCK_SIZE;

    // ===== 大shape场景：L2Cache切分 =====
    uint64_t totalData = static_cast<uint64_t>(inputLength) * 2;
    uint32_t useL2Cache = (totalData > L2_CACHE_SIZE) ? 1 : 0;

    // 核数计算：大shape用满核，小shape正常
    uint32_t coreNum;
    if (useL2Cache) {
        // 大shape：使用全部核（可超过物理核数）
        coreNum = std::min(maxCoreNum * 2, totalBlocks);
        if (coreNum < 1) coreNum = 1;
    } else {
        coreNum = std::min(maxCoreNum, totalBlocks);
        coreNum = std::max(coreNum, static_cast<uint32_t>(1));
    }
    context->SetBlockDim(coreNum);

    // 负载均衡切分
    uint32_t baseBlocks = totalBlocks / coreNum;
    uint32_t remainBlocks = totalBlocks % coreNum;

    uint32_t smallCoreBlocks = baseBlocks;
    uint32_t bigCoreBlocks = baseBlocks + (remainBlocks > 0 ? 1 : 0);
    uint32_t tailBlockNum = remainBlocks;

    uint32_t smallCoreDataNum = smallCoreBlocks * BLOCK_SIZE / typeLength;
    uint32_t bigCoreDataNum = bigCoreBlocks * BLOCK_SIZE / typeLength;

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // 双缓冲
    const uint32_t BUFFER_NUM_FOR_CALC = 2;
    // 输入2 + 输出2 + 临时1 = 5
    uint32_t ubDataNumber = 5;
    uint32_t tileBlockNum = (ubSize / BLOCK_SIZE / BUFFER_NUM_FOR_CALC) / ubDataNumber;
    uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / typeLength;
    if (tileDataNum == 0) tileDataNum = 1;

    // ===== 大shape：强制更大的 tile =====
    if (useL2Cache) {
        // 大shape用更大的 tile，减少循环次数
        uint32_t bigTile = (typeLength == 2) ? 2048 : 1024;
        if (tileDataNum < bigTile) {
            tileDataNum = bigTile;
        }
    } else {
        uint32_t minTile = (typeLength == 2) ? 1024 : 512;
        if (tileDataNum < minTile) {
            tileDataNum = minTile;
        }
    }
    tileDataNum = ((tileDataNum + 7) / 8) * 8;

    // 不超过每核最大处理数据量
    uint32_t maxPerCore = (smallCoreDataNum > bigCoreDataNum) ? smallCoreDataNum : bigCoreDataNum;
    if (tileDataNum > maxPerCore) {
        tileDataNum = maxPerCore;
        tileDataNum = ((tileDataNum + 7) / 8) * 8;
        if (tileDataNum == 0) tileDataNum = 1;
    }

    // ===== 临时空间：大shape用更大的临时空间 =====
    uint32_t tmpSize;
    if (useL2Cache) {
        tmpSize = tileDataNum * typeLength * 3;  // 大shape给更多临时空间
    } else {
        tmpSize = tileDataNum * typeLength * (typeLength == 2 ? 1 : 2);
    }
    if (tmpSize < 1024) tmpSize = 1024;
    tmpSize = ((tmpSize + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

    // 重新计算 tile 循环参数
    uint32_t tileBlockNumFinal = (tileDataNum * typeLength + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (tileBlockNumFinal == 0) tileBlockNumFinal = 1;

    uint32_t smallTileNum = smallCoreBlocks / tileBlockNumFinal;
    uint32_t finalSmallTileNum = (smallCoreBlocks % tileBlockNumFinal == 0) ? smallTileNum : smallTileNum + 1;
    uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = (smallTailDataNum == 0) ? tileDataNum : smallTailDataNum;

    uint32_t bigTileNum = bigCoreBlocks / tileBlockNumFinal;
    uint32_t finalBigTileNum = (bigCoreBlocks % tileBlockNumFinal == 0) ? bigTileNum : bigTileNum + 1;
    uint32_t bigTailDataNum = bigCoreDataNum - (tileDataNum * bigTileNum);
    bigTailDataNum = (bigTailDataNum == 0) ? tileDataNum : bigTailDataNum;

    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->tailBlockNum = tailBlockNum;
    tiling->tmpSize = tmpSize;
    tiling->useL2Cache = useL2Cache;

    auto dtype_x = context->GetInputDesc(0)->GetDataType();
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtype_x));

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
class Gelu : public ops::OpDef {
public:
    explicit Gelu(const char* name) : ops::OpDef(name)
    {
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Gelu);
}
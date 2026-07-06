#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 获取AscendC硬件平台信息
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // 获取可用AI Core总数
    auto coreNum = ascendcPlatform.GetCoreNum();
    // 获取输入张量总元素数
    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    // 获取输入数据类型的字节宽度
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);

    // 总输入字节数
    uint32_t inputLength = inputNum * typeLength;
    const uint32_t BLOCK_SIZE = 32;
    // 向上对齐到32B搬运粒度
    uint32_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);

    // 根据数据量自适应实际使用的Core数量
    coreNum = std::min(coreNum, inputLengthAlgin32 / BLOCK_SIZE);
    coreNum = std::max(coreNum, static_cast<uint32_t>(1));

    // 每个Core默认分配的32B块数
    uint32_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
    // 无法均分的剩余块，由前tailBlockNum个Core各多承担1块
    uint32_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;

    // 设置Kernel启动的Core数
    context->SetBlockDim(coreNum);

    // 获取单Core的UB内存大小
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const uint32_t BUFFER_NUM = 2;
    // UB内需要的buffer数量：输入队列、输出队列、2个float临时计算buffer
    uint32_t ubDataNumber = 4;
    // 单个Tile可占用的32B块数
    uint32_t tileBlockNum = (ubSize / BLOCK_SIZE / BUFFER_NUM) / ubDataNumber;
    // 单个Tile可处理的元素个数
    uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / typeLength;

    // small core：普通Core的总数据量、Tile数、尾块数据量
    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum =
        (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    // big core：前tailBlockNum个Core多承担1个32B块
    everyCoreInputBlockNum += 1;
    uint32_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalBigTileNum =
        (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    // 写入Tiling数据
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

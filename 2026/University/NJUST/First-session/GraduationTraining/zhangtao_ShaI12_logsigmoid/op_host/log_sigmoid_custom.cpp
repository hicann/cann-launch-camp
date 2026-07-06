#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 获取 AscendC 平台信息
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    // 获取当前硬件可用的 AI Core 数量
    auto coreNum = ascendcPlatform.GetCoreNum();

    // 获取输入 x 的元素总数
    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();

    // 获取输入数据类型所占字节数
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);

    // 输入总字节数
    uint32_t inputLength = inputNum * typeLength;

    // Ascend C 中数据搬运通常按 32B 对齐
    const uint32_t BLOCK_SIZE = 32;

    // 将输入总字节数向上对齐到 32B
    uint32_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);

    // 根据数据量决定实际使用多少个 Core
    coreNum = std::min(coreNum, inputLengthAlgin32 / BLOCK_SIZE);
    coreNum = std::max(coreNum, static_cast<uint32_t>(1));

    // 每个 Core 默认分到的 32B block 数
    uint32_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;

    // 不能平均分完的 block 数
    uint32_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;

    // 设置 Kernel 启动时使用的 Core 数
    context->SetBlockDim(coreNum);

    // 获取每个 Core 的 UB 大小
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // 双缓冲数量
    const uint32_t BUFFER_NUM = 2;

    // 需要额外的临时 buffer: 
    // 1. tmpFloatX: 用于类型转换的临时 buffer
    // 2. tmpFloatZ: 用于计算的临时 buffer
    uint32_t ubDataNumber = 4;

    // 每个 Tile 可以占用多少个 32B block
    uint32_t tileBlockNum = (ubSize / BLOCK_SIZE / BUFFER_NUM) / ubDataNumber;

    // 每个 Tile 可以处理多少个元素
    uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / typeLength;

    // small core：普通 Core 处理的数据量
    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;

    // small core 能完整处理多少个 Tile
    uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;

    // small core 最终 Tile 数
    uint32_t finalSmallTileNum =
        (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;

    // small core 最后一个 Tile 的数据量
    uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    // big core 比 small core 多处理一个 32B block
    everyCoreInputBlockNum += 1;

    // big core 处理的数据量
    uint32_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;

    // big core 能完整处理多少个 Tile
    uint32_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;

    // big core 最终 Tile 数
    uint32_t finalBigTileNum =
        (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;

    // big core 最后一个 Tile 的数据量
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    // 获取 TilingData 指针
    LogSigmoidCustomTilingData *tiling = context->GetTilingData<LogSigmoidCustomTilingData>();

    // 将 Host 侧计算好的 Tiling 参数写入 TilingData
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->tailBlockNum = tailBlockNum;

    auto input_desc = context->GetInputDesc(0);
    auto data_type = input_desc->GetDataType();
    if (data_type == ge::DT_FLOAT) {
        tiling->dataType = 0;
    } else if (data_type == ge::DT_FLOAT16) {
        tiling->dataType = 1;
    } else if (data_type == ge::DT_BF16) {
        tiling->dataType = 2;
    } else {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    // LogSigmoid 的输出 shape 和输入 shape 一样
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    // LogSigmoid 的输出数据类型和输入数据类型保持一致
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

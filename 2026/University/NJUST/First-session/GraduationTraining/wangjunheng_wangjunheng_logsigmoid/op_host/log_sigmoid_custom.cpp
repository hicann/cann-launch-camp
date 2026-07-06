#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 获取平台信息，用于查询 AI Core 数量和 UB 大小
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    // 获取 AI Core 数量
    auto coreNum = ascendcPlatform.GetCoreNum();

    // 获取输入元素总数
    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();

    // 获取输入类型字节数
    // float16 / bf16 = 2B
    // float32 = 4B
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);

    // 输入总字节数
    uint32_t inputLength = inputNum * typeLength;

    // Ascend C 数据搬运按 32B 对齐
    const uint32_t BLOCK_SIZE = 32;

    // 输入长度向上对齐到 32B
    uint32_t inputLengthAlign32 =
        ((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

    // 实际使用的 Core 数不能超过 32B block 数
    coreNum = std::min(coreNum, inputLengthAlign32 / BLOCK_SIZE);
    coreNum = std::max(coreNum, static_cast<uint32_t>(1));

    // 设置 Kernel 启动 Core 数
    context->SetBlockDim(coreNum);

    // 总 32B block 数
    uint32_t totalBlockNum = inputLengthAlign32 / BLOCK_SIZE;

    // 每个 Core 默认处理的 32B block 数
    uint32_t everyCoreInputBlockNum = totalBlockNum / coreNum;

    // 剩余 block 数
    // 前 tailBlockNum 个 Core 多处理 1 个 32B block
    uint32_t tailBlockNum = totalBlockNum % coreNum;

    // 获取 UB 大小
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const uint32_t BUFFER_NUM = 2;

    // LogSigmoid 对 bf16 需要额外 float 临时空间
    // 这里统一按输入、输出、float输入、float输出共 4 份估算 UB
    uint32_t ubDataNumber = 4;

    // 每个 tile 可使用的 32B block 数
    uint32_t tileBlockNum = (ubSize / BLOCK_SIZE / BUFFER_NUM) / ubDataNumber;

    // 每个 tile 的元素个数
    uint32_t tileDataNum = tileBlockNum * BLOCK_SIZE / typeLength;

    // small core 处理的数据量
    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;

    // small core 完整 tile 数
    uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;

    // small core 最终 tile 数
    uint32_t finalSmallTileNum =
        (everyCoreInputBlockNum % tileBlockNum == 0) ? smallTileNum : smallTileNum + 1;

    // small core 最后一个 tile 数据量
    uint32_t smallTailDataNum = smallCoreDataNum - tileDataNum * smallTileNum;
    smallTailDataNum = (smallTailDataNum == 0) ? tileDataNum : smallTailDataNum;

    // big core 比 small core 多一个 32B block
    uint32_t bigEveryCoreInputBlockNum = everyCoreInputBlockNum + 1;

    // big core 处理的数据量
    uint32_t bigCoreDataNum = bigEveryCoreInputBlockNum * BLOCK_SIZE / typeLength;

    // big core 完整 tile 数
    uint32_t bigTileNum = bigEveryCoreInputBlockNum / tileBlockNum;

    // big core 最终 tile 数
    uint32_t finalBigTileNum =
        (bigEveryCoreInputBlockNum % tileBlockNum == 0) ? bigTileNum : bigTileNum + 1;

    // big core 最后一个 tile 数据量
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = (bigTailDataNum == 0) ? tileDataNum : bigTailDataNum;

    // 写入 TilingData
    LogSigmoidCustomTilingData *tiling =
        context->GetTilingData<LogSigmoidCustomTilingData>();

    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->tailBlockNum = tailBlockNum;

    // 当前算子不需要 workspace
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    // 输出 shape 与输入 shape 一致
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    // 输出 dtype 与输入 dtype 一致
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

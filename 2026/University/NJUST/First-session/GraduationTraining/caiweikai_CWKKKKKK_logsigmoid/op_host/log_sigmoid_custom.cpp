#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 获取硬件平台信息，用于计算核数和UB大小
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNum();                     // 可用AI Core总数

    // 获取输入数据总元素数
    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();

    // 获取数据类型长度（字节）
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);

    uint32_t inputLength = inputNum * typeLength;                    // 总数据量（字节）

    const uint32_t BLOCK_SIZE = 32;                                  // 按32字节对齐，便于带宽优化

    // 将输入长度向上对齐到32字节的整数倍
    uint32_t inputLengthAlign32 =
        ((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

    // 计算实际使用的核数：不超过可用核数，且确保每个核至少处理一个BLOCK_SIZE
    coreNum = std::min(coreNum, inputLengthAlign32 / BLOCK_SIZE);
    coreNum = std::max(coreNum, static_cast<uint32_t>(1));

    context->SetBlockDim(coreNum);                                   // 设置核数

    uint32_t totalBlockNum = inputLengthAlign32 / BLOCK_SIZE;       // 总块数（每块32字节）
    uint32_t everyCoreInputBlockNum = totalBlockNum / coreNum;      // 每个核平均块数
    uint32_t tailBlockNum = totalBlockNum % coreNum;                // 余数，分给前tailBlockNum个核

    // 获取UB大小（片上存储）
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const uint32_t BUFFER_NUM = 2;                                   // 双缓冲
    uint32_t ubDataNumber = 4;                                      // 每个float/half/bf16占4字节对齐

    // 每个tile的块数：UB能容纳的块数，除以双缓冲，再除以数据类型占用（按4字节为单位）
    uint32_t tileBlockNum = (ubSize / BLOCK_SIZE / BUFFER_NUM) / ubDataNumber;
    uint32_t tileDataNum = tileBlockNum * BLOCK_SIZE / typeLength;  // 每个tile的元素数

    // 计算“小核”（无尾块）和“大核”（多一个尾块）的参数
    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;   // 小核元素数
    uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;

    // 小核的tile总数（向上取整）
    uint32_t finalSmallTileNum =
        (everyCoreInputBlockNum % tileBlockNum == 0) ? smallTileNum : smallTileNum + 1;

    // 小核最后一个tile的实际元素数
    uint32_t smallTailDataNum = smallCoreDataNum - tileDataNum * smallTileNum;
    smallTailDataNum = (smallTailDataNum == 0) ? tileDataNum : smallTailDataNum;

    // 大核参数（多一个BLOCK_SIZE）
    uint32_t bigEveryCoreInputBlockNum = everyCoreInputBlockNum + 1;
    uint32_t bigCoreDataNum = bigEveryCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t bigTileNum = bigEveryCoreInputBlockNum / tileBlockNum;

    uint32_t finalBigTileNum =
        (bigEveryCoreInputBlockNum % tileBlockNum == 0) ? bigTileNum : bigTileNum + 1;

    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = (bigTailDataNum == 0) ? tileDataNum : bigTailDataNum;

    // 填充Tiling结构体，供Kernel使用
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

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;          // 无需额外workspace

    return ge::GRAPH_SUCCESS;
}
}

// 以下为shape推导、数据类型推导和算子注册，无需修改
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

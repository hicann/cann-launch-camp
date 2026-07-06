#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 获取 AI Core 数量和单个 AI Core 的 UB 大小。
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNum();

    // 获取输入张量 x 的元素总数。
    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();

    // 获取输入数据类型的单元素字节数
    // float16/bfloat16 2 字节，float32 4 字节。
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);

    // 先把输入总字节数向上补齐到 32B 的整数倍，再换算成 32B 数据块个数
    const uint32_t BLOCK_SIZE = 32;
    uint32_t inputLength = inputNum * typeLength;
    uint32_t inputLengthAlign32 = ((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    uint32_t totalBlockNum = inputLengthAlign32 / BLOCK_SIZE;

    // 实际启用的核数不能超过 32B 数据块个数
    coreNum = std::min(coreNum, totalBlockNum);
    coreNum = std::max(coreNum, static_cast<uint32_t>(1));

    // 核间切分：先平均分块，不能整除的余数交给前 tailBlockNum 个核
    uint32_t everyCoreInputBlockNum = totalBlockNum / coreNum;
    uint32_t tailBlockNum = totalBlockNum % coreNum;
    context->SetBlockDim(coreNum);

    // 获取单个 AI Core 的 UB 大小
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // 估算一次 tile 需要占用多少 UB 空间。
    // BUFFER_NUM=2 表示输入/输出队列采用双缓冲。
    // QUEUE_COUNT=2 对应一个输入 x 和一个输出 y。
    // TMP_FLOAT_COUNT=2 是为了后续支持 bfloat16：Kernel 里会准备 float 临时 Tensor。
    const uint32_t BUFFER_NUM = 2;
    const uint32_t QUEUE_COUNT = 2;      // input x + output y
    const uint32_t TMP_FLOAT_COUNT = 2;  // bf16 路径中使用的 float 临时空间

    uint32_t bytesPerElementInUb =
        QUEUE_COUNT * BUFFER_NUM * typeLength + TMP_FLOAT_COUNT * sizeof(float);

    // 根据 UB 大小算出单次最多处理多少元素
    // tileDataNum 必须换算成 32B 对齐后的元素个数
    uint32_t maxElementsByUb = static_cast<uint32_t>(ubSize / bytesPerElementInUb);
    uint32_t elementsPerBlock = BLOCK_SIZE / typeLength;
    uint32_t tileDataNum = (maxElementsByUb / elementsPerBlock) * elementsPerBlock;
    tileDataNum = std::max(tileDataNum, elementsPerBlock);

    // 单次 tile 对应多少个 32B 数据块
    uint32_t tileBlockNum = tileDataNum * typeLength / BLOCK_SIZE;
    tileBlockNum = std::max(tileBlockNum, static_cast<uint32_t>(1));

    // 计算小核的处理规模
    // smallCoreDataNum：小核总共处理多少个元素
    // finalSmallTileNum：小核需要循环处理多少次 tile
    // smallTailDataNum：小核最后一次 tile 实际处理多少元素
    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum =
        (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    finalSmallTileNum = std::max(finalSmallTileNum, static_cast<uint32_t>(1));

    uint32_t smallTailDataNum = smallCoreDataNum - tileDataNum * smallTileNum;
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    // 计算大核的处理规模
    // 大核比小核多处理 1 个 32B 数据块
    everyCoreInputBlockNum += 1;
    uint32_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalBigTileNum =
        (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    finalBigTileNum = std::max(finalBigTileNum, static_cast<uint32_t>(1));

    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    // 把 Host 侧算好的切分参数写入 TilingData
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
    // Sigmoid 是逐元素算子，输出 y 的 shape 与输入 x 完全相同。
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    // 输出 dtype 跟输入 dtype 保持一致：
    // float16 -> float16，float -> float，bfloat16 -> bfloat16。
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class LogSigmoidCustom : public OpDef {
public:
    explicit LogSigmoidCustom(const char* name) : OpDef(name)
    {
        // 注册输入 x
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // 注册输出 y
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // 绑定 shape/dtype 推导函数
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        // 绑定 AI Core Tiling 函数
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

// 注册 SigmoidCustom 算子
OP_ADD(LogSigmoidCustom);
}

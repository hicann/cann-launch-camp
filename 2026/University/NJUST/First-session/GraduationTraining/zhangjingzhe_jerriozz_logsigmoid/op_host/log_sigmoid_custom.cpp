#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platform.GetCoreNum();
    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();

    uint32_t typeLen = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLen);

    const uint32_t BLOCK = 32;
    uint32_t inputBytes = inputNum * typeLen;
    uint32_t alignedBytes = ((inputBytes + BLOCK - 1) / BLOCK) * BLOCK;

    // 确定实际可以使用的核心数
    coreNum = std::max(1u, std::min(coreNum, alignedBytes / BLOCK));

    uint32_t perCoreBlock = (alignedBytes / BLOCK) / coreNum;
    uint32_t tailBlockNum = (alignedBytes / BLOCK) % coreNum;

    context->SetBlockDim(coreNum);

    uint64_t ubSize;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const uint32_t BUFFER_NUM = 2;

    // 精准计算每个数据点在 UB 中占用的字节数：
    // inQueue (双缓冲, typeLen) + outQueue (双缓冲, typeLen) + calcBuf (单缓冲 float, 4字节)
    uint32_t bytesPerElement = (BUFFER_NUM * typeLen) + (BUFFER_NUM * typeLen) + 4;
    
    // 计算最大可容纳的 element 数量
    uint32_t maxTileElements = ubSize / bytesPerElement;

    // 为了满足矢量计算的 32 Byte 对齐要求
    uint32_t alignElements = BLOCK / typeLen;
    uint32_t tileDataNum = (maxTileElements / alignElements) * alignElements;

    uint32_t smallCoreDataNum = perCoreBlock * BLOCK / typeLen;
    uint32_t bigCoreDataNum = (perCoreBlock + 1) * BLOCK / typeLen;

    uint32_t smallTileNum = smallCoreDataNum / tileDataNum;
    uint32_t smallTailDataNum = smallCoreDataNum % tileDataNum;
    uint32_t finalSmallTileNum = smallTileNum + (smallTailDataNum > 0 ? 1 : 0);
    if (smallTailDataNum == 0) smallTailDataNum = tileDataNum;

    uint32_t bigTileNum = bigCoreDataNum / tileDataNum;
    uint32_t bigTailDataNum = bigCoreDataNum % tileDataNum;
    uint32_t finalBigTileNum = bigTileNum + (bigTailDataNum > 0 ? 1 : 0);
    if (bigTailDataNum == 0) bigTailDataNum = tileDataNum;

    auto* tiling = context->GetTilingData<LogSigmoidCustomTilingData>();
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

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(LogSigmoidCustom);
}

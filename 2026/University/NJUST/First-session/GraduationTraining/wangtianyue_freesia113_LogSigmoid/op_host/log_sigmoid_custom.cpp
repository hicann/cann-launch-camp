#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    LogSigmoidCustomTilingData *tiling = context->GetTilingData<LogSigmoidCustomTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);

    // 获取数据总大小
    int32_t data_sz = 1;
    for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
        data_sz *= x1_shape->GetStorageShape().GetDim(i);

    // 获取数据类型长度
    uint32_t typeLength = 0;
    auto dataType = context->GetInputDesc(0)->GetDataType();
    if (dataType == ge::DT_FLOAT) {
        typeLength = 4;
    } else if (dataType == ge::DT_FLOAT16) {
        typeLength = 2;
    } else if (dataType == ge::DT_BF16) {
        typeLength = 2;
    }

    uint32_t inputLength = data_sz * typeLength;
    const uint32_t BLOCK_SIZE = 32;

    // 计算核数量
    uint32_t coreNum = 8;
    uint32_t inputLengthAlign32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    coreNum = std::min(coreNum, inputLengthAlign32 / BLOCK_SIZE);
    coreNum = std::max(coreNum, static_cast<uint32_t>(1));

    uint32_t everyCoreInputBlockNum = inputLengthAlign32 / BLOCK_SIZE / coreNum;
    uint32_t tailBlockNum = (inputLengthAlign32 / BLOCK_SIZE) % coreNum;
    context->SetBlockDim(coreNum);

    // 使用固定的UB大小和tile大小
    const uint32_t BUFFER_NUM = 2;
    uint32_t ubDataNumber = (dataType == ge::DT_BF16) ? 4 : 2;
    uint64_t ubSize = 64 * 1024;  // 假设UB大小为64KB
    uint32_t tileBlockNum = (ubSize / BLOCK_SIZE / BUFFER_NUM) / ubDataNumber;
    uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / typeLength;

    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    // 前 tailBlockNum 个核多分到一个 32B block，称为 big core。
    everyCoreInputBlockNum += 1;
    uint32_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->tailBlockNum = tailBlockNum;

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
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

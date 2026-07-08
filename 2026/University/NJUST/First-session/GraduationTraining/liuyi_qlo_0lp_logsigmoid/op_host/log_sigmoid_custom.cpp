#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNum();

    LogSigmoidCustomTilingData *tiling = context->GetTilingData<LogSigmoidCustomTilingData>();
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    uint32_t inputNum = x_shape->GetStorageShape().GetShapeSize();

    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint32_t inputLength = inputNum * typeLength;

    const uint32_t BLOCK_SIZE = 32;
    // 1. 32B 对齐
    uint32_t inputLengthAlign32 = ((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    // 2. 核数选择：至少每核 1 个 32B 块
    coreNum = std::min(coreNum, inputLengthAlign32 / BLOCK_SIZE);
    coreNum = std::max(coreNum, static_cast<uint32_t>(1));
    uint32_t everyCoreInputBlockNum = inputLengthAlign32 / BLOCK_SIZE / coreNum;
    uint32_t tailBlockNum = (inputLengthAlign32 / BLOCK_SIZE) % coreNum;
    context->SetBlockDim(coreNum);

    // 3. 核内 UB 切分：in队列 + out队列 (各 typeLength) + 3个float32 TBuf
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint32_t TMP_FLOAT_NUM = 3;
    uint32_t bytesPerElement = 2 * typeLength + TMP_FLOAT_NUM * sizeof(float);
    uint32_t tileDataNum = ubSize / bytesPerElement;
    uint32_t elementsPerBlock = BLOCK_SIZE / typeLength;
    tileDataNum = (tileDataNum / elementsPerBlock) * elementsPerBlock; // 32B 对齐
    if (tileDataNum == 0) { tileDataNum = elementsPerBlock; }
    uint32_t tileBlockNum = tileDataNum * typeLength / BLOCK_SIZE;      // 单批 32B 块数

    // 4. 小核参数
    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum == 0) ? smallTileNum : smallTileNum + 1;
    uint32_t smallTailDataNum = smallCoreDataNum - tileDataNum * smallTileNum;
    smallTailDataNum = (smallTailDataNum == 0) ? tileDataNum : smallTailDataNum;

    // 5. 大核参数（多处理 1 个 32B 块）
    uint32_t bigCoreInputBlockNum = everyCoreInputBlockNum + 1;
    uint32_t bigCoreDataNum = bigCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t bigTileNum = bigCoreInputBlockNum / tileBlockNum;
    uint32_t finalBigTileNum = (bigCoreInputBlockNum % tileBlockNum == 0) ? bigTileNum : bigTileNum + 1;
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = (bigTailDataNum == 0) ? tileDataNum : bigTailDataNum;

    // 6. 封装 TilingData
    tiling->smallCoreDataNum   = smallCoreDataNum;
    tiling->bigCoreDataNum     = bigCoreDataNum;
    tiling->finalBigTileNum    = finalBigTileNum;
    tiling->finalSmallTileNum  = finalSmallTileNum;
    tiling->tileDataNum        = tileDataNum;
    tiling->smallTailDataNum   = smallTailDataNum;
    tiling->bigTailDataNum     = bigTailDataNum;
    tiling->tailBlockNum       = tailBlockNum;

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

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
} // namespace ge

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
} // namespace ops


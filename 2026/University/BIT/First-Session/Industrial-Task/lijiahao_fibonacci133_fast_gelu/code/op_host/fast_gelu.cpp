#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t maxCoreNum = platform.GetCoreNumAiv();

    // 获取输入信息（使用原版 API）
    const gert::Tensor* tensor_x = context->GetRequiredInputTensor(0);
    uint32_t totalLength = tensor_x->GetShapeSize();
    ge::DataType dtype = tensor_x->GetDataType();
    int dtypeSize = ge::GetSizeByDataType(dtype);
    uint32_t totalBytes = totalLength * dtypeSize;

    // 模板参数选择
    uint32_t DT_X = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    const uint32_t BLOCK_SIZE = 32;
    const uint32_t BUFFER_NUM = 2;
    const uint32_t UB_DATA_NUM = 4;          // x, tmp1, tmp2, y 四块双缓冲

    // 计算最大 tile 大小（基于 UB 容量）
    uint64_t ubSize;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t tileBlockNum = (ubSize / BLOCK_SIZE / BUFFER_NUM) / UB_DATA_NUM;
    uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / dtypeSize;
    if (tileDataNum == 0) tileDataNum = 1;

    // ---------- 自适应核数选择 ----------
    uint32_t coreNum;
    if (totalBytes <= 64 * 1024) {           // 小数据量用单核，消除并行开销
        coreNum = 1;
    } else {
        uint32_t alignedBytes = ((totalBytes + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
        coreNum = std::min(maxCoreNum, alignedBytes / BLOCK_SIZE / 2); // 每核至少2个32B块
        if (coreNum < 1) coreNum = 1;
    }
    context->SetBlockDim(coreNum);

    // 按 32B 对齐计算每核分块
    uint32_t inputAligned = ((totalLength * dtypeSize + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    uint32_t everyCoreBlocks = inputAligned / BLOCK_SIZE / coreNum;
    uint32_t tailBlockNum = (inputAligned / BLOCK_SIZE) % coreNum;

    // 小核数据量（基础块）
    uint32_t smallCoreDataNum = everyCoreBlocks * BLOCK_SIZE / dtypeSize;
    uint32_t smallTileNum = everyCoreBlocks / tileBlockNum;
    uint32_t finalSmallTileNum = (everyCoreBlocks % tileBlockNum == 0) ? smallTileNum : smallTileNum + 1;
    uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = (smallTailDataNum == 0) ? tileDataNum : smallTailDataNum;

    // 大核数据量（基础块 + 1 个 32B 块用于对齐冗余）
    everyCoreBlocks += 1;
    uint32_t bigCoreDataNum = everyCoreBlocks * BLOCK_SIZE / dtypeSize;
    uint32_t bigTileNum = everyCoreBlocks / tileBlockNum;
    uint32_t finalBigTileNum = (everyCoreBlocks % tileBlockNum == 0) ? bigTileNum : bigTileNum + 1;
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = (bigTailDataNum == 0) ? tileDataNum : bigTailDataNum;

    // 单核或没有大核时统一大小核参数，避免地址越界
    if (coreNum == 1 || tailBlockNum == 0) {
        bigCoreDataNum = smallCoreDataNum;
        finalBigTileNum = finalSmallTileNum;
        bigTailDataNum = smallTailDataNum;
    }

    // 填充 tiling 数据
    FastGeluTilingData* tiling = context->GetTilingData<FastGeluTilingData>();
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->tailBlockNum = tailBlockNum;

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class FastGelu : public OpDef {
public:
    explicit FastGelu(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(FastGelu);
}  // namespace ops
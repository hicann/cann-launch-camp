// Host 侧 Tiling — 泛化设计：核间均分（大核/小核）+ 核内按UB切分（批次+尾块）
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    // 1. 获取平台信息：可用核数
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNum();

    // 2. 获取输入数据类型和大小
    const gert::Tensor *tensorInputX = context->GetRequiredInputTensor(0);
    ge::DataType dtypeInputX = tensorInputX->GetDataType();
    uint32_t inputNum = static_cast<uint32_t>(tensorInputX->GetShapeSize());
    if (inputNum == 0) inputNum = 1;

    uint32_t DT_INPUT_X = static_cast<uint32_t>(dtypeInputX);
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

    uint32_t typeSize = (dtypeInputX == ge::DT_FLOAT16) ? 2 : 4;
    uint32_t inputBytes = inputNum * typeSize;

    // 3. 32B 内存对齐（硬件约束）
    constexpr uint32_t BLOCK_SIZE = 32;
    uint32_t alignedBytes = ((inputBytes + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

    // 4. 核间数据拆分：大核/小核
    uint32_t totalBlocks = alignedBytes / BLOCK_SIZE;
    coreNum = std::min(coreNum, totalBlocks);
    // 每核至少处理 4 个 32B 块，避免小数据量多核开销
    constexpr uint32_t MIN_BLOCKS_PER_CORE = 4;
    coreNum = std::min(coreNum, totalBlocks / MIN_BLOCKS_PER_CORE);
    coreNum = std::max(coreNum, static_cast<uint32_t>(1));
    uint32_t everyCoreInputBlockNum = totalBlocks / coreNum;
    uint32_t tailBlockNum = totalBlocks % coreNum;
    context->SetBlockDim(coreNum);

    // 5. 核内数据切分：160KB UB 均分给 5 个 buffer（inQueueX×2 + outQueueY×2 + tmpBuffer×1）
    constexpr uint32_t UB_BYTES = 160 * 1024;
    constexpr uint32_t ubDataNumber = 5;
    uint32_t tileBlockNum = (UB_BYTES / BLOCK_SIZE) / ubDataNumber;
    uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / typeSize;

    // 6. 小核参数（基准核）
    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeSize;
    uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0
                                     ? smallTileNum
                                     : smallTileNum + 1;
    uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    // 7. 大核参数（比基准核多1个32B块）
    everyCoreInputBlockNum += 1;
    uint32_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeSize;
    uint32_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0
                                   ? bigTileNum
                                   : bigTileNum + 1;
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    // 8. 写入结构体
    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
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

static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const auto inputDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDtype);
    return ge::GRAPH_SUCCESS;
}

}

namespace ops {

class Gelu : public OpDef {
public:
    explicit Gelu(const char *name) : OpDef(name) {
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Gelu);

}

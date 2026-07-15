#include "../op_kernel/gelu_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t hwCoreNum = ascendcPlatform.GetCoreNum();

    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();

    uint32_t typeLength = 0;
    ge::DataType inputDtype = context->GetInputDesc(0)->GetDataType();
    ge::TypeUtils::GetDataTypeLength(inputDtype, typeLength);

    const uint32_t BLOCK_SIZE = 32;

    uint32_t inputLength = inputNum * typeLength;
    uint32_t inputLengthAlign32 =
        ((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

    uint32_t totalBlock = inputLengthAlign32 / BLOCK_SIZE;
    totalBlock = std::max(totalBlock, 1U);

    uint32_t coreNum = hwCoreNum;

    if (totalBlock <= 32) {
        coreNum = 1;
    } else if (totalBlock <= 64) {
        coreNum = std::min(hwCoreNum, 2U);
    } else if (totalBlock <= 128) {
        coreNum = std::min(hwCoreNum, 4U);
    } else if (totalBlock <= 256) {
        coreNum = std::min(hwCoreNum, 8U);
    } else {
        coreNum = std::min(hwCoreNum, 20U);
    }

    coreNum = std::min(coreNum, totalBlock);
    coreNum = std::max(coreNum, 1U);

    uint32_t everyCoreInputBlockNum = totalBlock / coreNum;
    uint32_t tailBlockNum = totalBlock % coreNum;

    context->SetBlockDim(coreNum);

    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t bigCoreBlockNum = everyCoreInputBlockNum + 1;
    uint32_t bigCoreDataNum = bigCoreBlockNum * BLOCK_SIZE / typeLength;

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    
    uint32_t tileBlockNum =
        static_cast<uint32_t>((ubSize * 8 / 10) / BLOCK_SIZE / 2);

    tileBlockNum = std::max(tileBlockNum, 1U);

    uint32_t maxCoreBlockNum =
        everyCoreInputBlockNum + ((tailBlockNum > 0) ? 1U : 0U);

    tileBlockNum = std::min(tileBlockNum, maxCoreBlockNum);
    tileBlockNum = std::max(tileBlockNum, 1U);

    uint32_t tileDataNum = tileBlockNum * BLOCK_SIZE / typeLength;

    uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum =
        (everyCoreInputBlockNum % tileBlockNum == 0) ? smallTileNum : smallTileNum + 1;
    finalSmallTileNum = std::max(finalSmallTileNum, 1U);

    uint32_t smallTailDataNum =
        smallCoreDataNum - tileDataNum * smallTileNum;
    smallTailDataNum = (smallTailDataNum == 0) ? tileDataNum : smallTailDataNum;

    uint32_t bigTileNum = bigCoreBlockNum / tileBlockNum;
    uint32_t finalBigTileNum =
        (bigCoreBlockNum % tileBlockNum == 0) ? bigTileNum : bigTileNum + 1;
    finalBigTileNum = std::max(finalBigTileNum, 1U);

    uint32_t bigTailDataNum =
        bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = (bigTailDataNum == 0) ? tileDataNum : bigTailDataNum;

    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();

    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->tailBlockNum = tailBlockNum;

    tiling->input_dtype_flag = (inputDtype == ge::DT_FLOAT16) ? 0U : 1U;

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
    auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Gelu : public OpDef {
public:
    explicit Gelu(const char* name) : OpDef(name)
    {
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

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Gelu);
}
#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* tilingCtx)
{
    auto hwPlatform = platform_ascendc::PlatformAscendC(tilingCtx->GetPlatformInfo());
    uint32_t totalCoreCount = hwPlatform.GetCoreNum();

    uint32_t elemCount = tilingCtx->GetInputShape(0)->GetStorageShape().GetShapeSize();

    uint32_t bytePerElem = 0;
    ge::TypeUtils::GetDataTypeLength(tilingCtx->GetInputDesc(0)->GetDataType(), bytePerElem);
    uint32_t totalBytes = elemCount * bytePerElem;

    constexpr uint32_t ALIGN_UNIT = 32;
    uint32_t alignedBytes = ((totalBytes + ALIGN_UNIT - 1) / ALIGN_UNIT) * ALIGN_UNIT;

    uint32_t activeCores = std::min(totalCoreCount, alignedBytes / ALIGN_UNIT);
    activeCores = std::max(activeCores, static_cast<uint32_t>(1));

    uint32_t baseBlocksPerCore = alignedBytes / ALIGN_UNIT / activeCores;
    uint32_t extraBlocks = (alignedBytes / ALIGN_UNIT) % activeCores;
    tilingCtx->SetBlockDim(activeCores);

    uint64_t ubCapacity;
    hwPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCapacity);
    constexpr uint32_t PIPELINE_DEPTH = 2;

    constexpr uint32_t BUF_SLOTS = 4;
    uint32_t tileBlocks = (ubCapacity / ALIGN_UNIT / PIPELINE_DEPTH) / BUF_SLOTS;
    uint32_t elemsPerTile = (tileBlocks * ALIGN_UNIT) / bytePerElem;

    uint32_t smallElems = baseBlocksPerCore * ALIGN_UNIT / bytePerElem;
    uint32_t smallFullTiles = baseBlocksPerCore / tileBlocks;
    uint32_t smallTotalTiles =
        (baseBlocksPerCore % tileBlocks) == 0 ? smallFullTiles : smallFullTiles + 1;
    uint32_t smallLastTileElems = smallElems - (elemsPerTile * smallFullTiles);
    smallLastTileElems = (smallLastTileElems == 0) ? elemsPerTile : smallLastTileElems;

    uint32_t bigBlocksPerCore = baseBlocksPerCore + 1;
    uint32_t bigElems = bigBlocksPerCore * ALIGN_UNIT / bytePerElem;
    uint32_t bigFullTiles = bigBlocksPerCore / tileBlocks;
    uint32_t bigTotalTiles =
        (bigBlocksPerCore % tileBlocks) == 0 ? bigFullTiles : bigFullTiles + 1;
    uint32_t bigLastTileElems = bigElems - elemsPerTile * bigFullTiles;
    bigLastTileElems = (bigLastTileElems == 0) ? elemsPerTile : bigLastTileElems;

    LogSigmoidCustomTilingData *pTiling = tilingCtx->GetTilingData<LogSigmoidCustomTilingData>();
    pTiling->smallCoreElemCnt      = smallElems;
    pTiling->bigCoreElemCnt        = bigElems;
    pTiling->elemPerTile           = elemsPerTile;
    pTiling->smallLastTileElemCnt  = smallLastTileElems;
    pTiling->bigLastTileElemCnt    = bigLastTileElems;
    pTiling->smallTotalTileCnt     = smallTotalTiles;
    pTiling->bigTotalTileCnt       = bigTotalTiles;
    pTiling->extraBlockCnt         = extraBlocks;

    return ge::GRAPH_SUCCESS;
}
} 


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* shapeCtx)
{
    const gert::Shape* srcShape = shapeCtx->GetInputShape(0);
    gert::Shape* dstShape = shapeCtx->GetOutputShape(0);
    *dstShape = *srcShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* dtypeCtx)
{
    auto inDtype = dtypeCtx->GetInputDataType(0);
    dtypeCtx->SetOutputDataType(0, inDtype);
    return ge::GRAPH_SUCCESS;
}
} 


namespace ops {
class LogSigmoidCustom : public OpDef {
public:
    explicit LogSigmoidCustom(const char* opName) : OpDef(opName)
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

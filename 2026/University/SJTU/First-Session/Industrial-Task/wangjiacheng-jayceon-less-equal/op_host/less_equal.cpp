// Host-side shape inference and tiling for LessEqual.
#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/less_equal_tiling.h"
#include "../op_kernel/tiling_key_less_equal.h"

namespace {

constexpr uint32_t TILE_LENGTH = 8192;
constexpr uint64_t MIN_ELEMENTS_PER_CORE = 32768;

template <typename ShapeT>
bool MakeBroadcastShape(const ShapeT &x1Shape, const ShapeT &x2Shape,
                        uint64_t (&outDims)[LESS_EQUAL_MAX_DIMS], uint32_t &rank)
{
    const int64_t rank1 = x1Shape.GetDimNum();
    const int64_t rank2 = x2Shape.GetDimNum();
    const int64_t outRank = std::max(rank1, rank2);
    if (outRank < 0 || outRank > LESS_EQUAL_MAX_DIMS) {
        return false;
    }

    rank = static_cast<uint32_t>(outRank);
    for (uint32_t i = 0; i < rank; ++i) {
        const int64_t i1 = static_cast<int64_t>(i) - (outRank - rank1);
        const int64_t i2 = static_cast<int64_t>(i) - (outRank - rank2);
        const int64_t d1 = i1 < 0 ? 1 : x1Shape.GetDim(i1);
        const int64_t d2 = i2 < 0 ? 1 : x2Shape.GetDim(i2);
        if (d1 < 0 || d2 < 0 || (d1 != d2 && d1 != 1 && d2 != 1)) {
            return false;
        }
        outDims[i] = static_cast<uint64_t>(d1 == 1 ? d2 : d1);
    }
    return true;
}

template <typename ShapeT>
void MakeBroadcastStrides(const ShapeT &inputShape, const uint64_t (&outDims)[LESS_EQUAL_MAX_DIMS],
                          uint32_t rank, uint64_t (&strides)[LESS_EQUAL_MAX_DIMS])
{
    const int64_t inputRank = inputShape.GetDimNum();
    const int64_t leading = static_cast<int64_t>(rank) - inputRank;
    uint64_t rawStride = 1;
    for (int64_t i = static_cast<int64_t>(rank) - 1; i >= 0; --i) {
        const int64_t inputIndex = i - leading;
        const uint64_t dim = inputIndex < 0 ? 1 : static_cast<uint64_t>(inputShape.GetDim(inputIndex));
        strides[i] = (dim == 1 && outDims[i] != 1) ? 0 : rawStride;
        rawStride *= dim;
    }
}

}  // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const auto x1StorageShape = context->GetInputShape(0)->GetStorageShape();
    const auto x2StorageShape = context->GetInputShape(1)->GetStorageShape();
    auto *tiling = context->GetTilingData<LessEqualTilingData>();

    uint64_t outputDims[LESS_EQUAL_MAX_DIMS] = {};
    uint32_t rank = 0;
    if (!MakeBroadcastShape(x1StorageShape, x2StorageShape, outputDims, rank)) {
        return ge::GRAPH_FAILED;
    }

    uint64_t totalLength = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        tiling->outputDims[i] = outputDims[i];
        totalLength *= outputDims[i];
    }
    MakeBroadcastStrides(x1StorageShape, outputDims, rank, tiling->x1Strides);
    MakeBroadcastStrides(x2StorageShape, outputDims, rank, tiling->x2Strides);
    for (uint32_t i = rank; i < LESS_EQUAL_MAX_DIMS; ++i) {
        tiling->outputDims[i] = 1;
        tiling->x1Strides[i] = 0;
        tiling->x2Strides[i] = 0;
    }

    const ge::DataType dtype = context->GetInputDesc(0)->GetDataType();
    const uint64_t dtypeSize = static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    const uint64_t x1Length = static_cast<uint64_t>(x1StorageShape.GetShapeSize());
    const uint64_t x2Length = static_cast<uint64_t>(x2StorageShape.GetShapeSize());
    const uint64_t inner = rank == 0 ? 1 : outputDims[rank - 1];
    const bool rowBroadcastFast = rank != 0 && inner <= TILE_LENGTH &&
        (inner * dtypeSize) % 32 == 0;
    if (x1Length == totalLength && x2Length == totalLength) {
        tiling->mode = LESS_EQUAL_MODE_FLAT;
    } else if (x1Length == 1 && x2Length == totalLength) {
        tiling->mode = LESS_EQUAL_MODE_X1_SCALAR;
    } else if (x2Length == 1 && x1Length == totalLength) {
        tiling->mode = LESS_EQUAL_MODE_X2_SCALAR;
    } else if (x1Length == inner && x2Length == totalLength) {
        tiling->mode = rowBroadcastFast ?
            LESS_EQUAL_MODE_X1_ROW : LESS_EQUAL_MODE_X1_REPEAT;
    } else if (x2Length == inner && x1Length == totalLength) {
        tiling->mode = rowBroadcastFast ?
            LESS_EQUAL_MODE_X2_ROW : LESS_EQUAL_MODE_X2_REPEAT;
    } else {
        tiling->mode = LESS_EQUAL_MODE_GENERAL;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
    coreNum = std::max(coreNum, 1U);
    const uint64_t usefulCores = totalLength == 0 ? 1 :
        (totalLength + MIN_ELEMENTS_PER_CORE - 1) / MIN_ELEMENTS_PER_CORE;
    const uint32_t blockNum =
        static_cast<uint32_t>(std::min<uint64_t>(coreNum, std::max<uint64_t>(usefulCores, 1)));

    tiling->totalLength = totalLength;
    tiling->rank = rank;
    tiling->blockNum = blockNum;
    tiling->tileLength = TILE_LENGTH;

    const uint32_t DT_X1 = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_X1);

    context->SetBlockDim(blockNum);
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x1Shape = context->GetInputShape(0);
    const gert::Shape *x2Shape = context->GetInputShape(1);
    gert::Shape *yShape = context->GetOutputShape(0);

    uint64_t outputDims[LESS_EQUAL_MAX_DIMS] = {};
    uint32_t rank = 0;
    if (!MakeBroadcastShape(*x1Shape, *x2Shape, outputDims, rank)) {
        return GRAPH_FAILED;
    }
    yShape->SetDimNum(rank);
    for (uint32_t i = 0; i < rank; ++i) {
        yShape->SetDim(i, static_cast<int64_t>(outputDims[i]));
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context->GetInputDataType(0) != context->GetInputDataType(1)) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_BOOL);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class LessEqual : public OpDef {
public:
    explicit LessEqual(const char *name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(LessEqual);

}  // namespace ops

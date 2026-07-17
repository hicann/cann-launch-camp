// Host-side registration, shape inference and tiling for LessEqual.
#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/less_equal_tiling.h"
#include "../op_kernel/tiling_key_less_equal.h"

namespace {
constexpr uint64_t BOOL_BLOCK_ELEMENTS = 32;
constexpr uint32_t FALLBACK_TILE_ELEMENTS = 4096;
constexpr uint32_t MAX_TILE_ELEMENTS = 8192;
constexpr uint32_t MIN_TILE_ELEMENTS = 128;
constexpr uint32_t TILE_ALIGNMENT_ELEMENTS = 128;
constexpr uint32_t DOUBLE_BUFFER_COUNT = 2;
constexpr uint64_t UB_SAFETY_RESERVE_BYTES = 16U * 1024U;
constexpr uint64_t SELECT_INTERNAL_RESERVE_BYTES = 8U * 1024U;

constexpr uint64_t COST_PER_128_INT8 = 1296;
constexpr uint64_t COST_PER_128_FP16 = 1680;  // fp16 VECOUT=3
constexpr uint64_t COST_PER_128_FP32 = 2576;
constexpr uint64_t COST_PER_128_INT32 = 3088;

constexpr uint32_t MAX_TILE_INT8 = 22528;
constexpr uint32_t MAX_TILE_FP16 = 16384;
constexpr uint32_t MAX_TILE_FP32 = 11264;
constexpr uint32_t MAX_TILE_INT32 = 10240;

inline uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    if (divisor == 0) return 0;
    return value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

inline uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}

bool SafeMul(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (lhs == 0 || rhs == 0) { result = 0; return true; }
    if (lhs > std::numeric_limits<uint64_t>::max() / rhs) return false;
    result = lhs * rhs;
    return true;
}

uint32_t GetDataTypeSize(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16: return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32:   return 4;
        case ge::DT_INT8:    return 1;
        default:             return 0;
    }
}

bool UsesVectorSelectPath(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT ||
           dtype == ge::DT_INT8 || dtype == ge::DT_INT32;
}

uint64_t EstimateUbBytes(uint32_t tileElements, uint32_t typeBytes, ge::DataType dtype)
{
    const uint64_t inputBuf = AlignUp(static_cast<uint64_t>(tileElements) * typeBytes, 32);
    const uint64_t outputBuf = AlignUp(tileElements, 32);
    // fp16 uses VECOUT depth 3; other types use 2.
    const uint64_t outBufCount = (dtype == ge::DT_FLOAT16) ? 3U : 2U;
    uint64_t total = DOUBLE_BUFFER_COUNT * (2U * inputBuf) + outBufCount * outputBuf;
    // tempHalf1Buf_ is unconditionally allocated in kernel Init for all types.
    total += AlignUp(static_cast<uint64_t>(tileElements) * sizeof(uint16_t), 32);
    if (UsesVectorSelectPath(dtype))
        total += AlignUp(CeilDiv(tileElements, 8U), 32);
    if (dtype == ge::DT_INT8) {
        // int8 needs a second half workspace for int8→half conversion.
        total += AlignUp(static_cast<uint64_t>(tileElements) * sizeof(uint16_t), 32);
    } else if (dtype == ge::DT_INT32) {
        // int32 workspace for Min.
        total += AlignUp(static_cast<uint64_t>(tileElements) * sizeof(int32_t), 32);
    }
    // DT_FLOAT16 and DT_FLOAT use only the always-allocated tempHalf1Buf_.
    return total;
}

uint32_t SelectTileLength(uint64_t ubSize, const ge::DataType dtype, uint64_t totalLength)
{
    const uint32_t typeBytes = GetDataTypeSize(dtype);
    if (typeBytes == 0 || ubSize == 0) return FALLBACK_TILE_ELEMENTS;

    uint64_t reserve = UB_SAFETY_RESERVE_BYTES;
    if (UsesVectorSelectPath(dtype)) reserve += SELECT_INTERNAL_RESERVE_BYTES;
    const uint64_t usableUb = ubSize > reserve ? ubSize - reserve : ubSize;

    uint32_t maxTile = MAX_TILE_ELEMENTS;
    uint64_t costPer128 = COST_PER_128_FP16;
    switch (typeBytes) {
        case 1: maxTile = MAX_TILE_INT8;   costPer128 = COST_PER_128_INT8;   break;
        case 2: maxTile = MAX_TILE_FP16;   costPer128 = COST_PER_128_FP16;   break;
        case 4:
            if (dtype == ge::DT_FLOAT) { maxTile = MAX_TILE_FP32; costPer128 = COST_PER_128_FP32; }
            else                       { maxTile = MAX_TILE_INT32; costPer128 = COST_PER_128_INT32; }
            break;
    }

    uint64_t tile64 = (usableUb * TILE_ALIGNMENT_ELEMENTS) / costPer128;
    tile64 = (tile64 / TILE_ALIGNMENT_ELEMENTS) * TILE_ALIGNMENT_ELEMENTS;
    if (tile64 > maxTile) tile64 = maxTile;
    if (tile64 > totalLength) tile64 = totalLength;
    if (tile64 < MIN_TILE_ELEMENTS) tile64 = MIN_TILE_ELEMENTS;

    while (tile64 > MIN_TILE_ELEMENTS && EstimateUbBytes(tile64, typeBytes, dtype) > usableUb)
        tile64 -= TILE_ALIGNMENT_ELEMENTS;

    return static_cast<uint32_t>(tile64);
}

void AnalyzeInnerSpanHost(LessEqualTilingData &tiling)
{
    uint32_t x1Mode = 0, x2Mode = 0;
    uint64_t innerLength = 1;
    bool firstNonBcast = true;

    for (uint32_t reverse = 0; reverse < tiling.rank; ++reverse) {
        const uint32_t dim = tiling.rank - 1U - reverse;
        const uint64_t dimSize = tiling.outputShape[dim];
        if (dimSize == 1U) continue;

        const uint64_t s1 = tiling.x1Stride[dim];
        const uint64_t s2 = tiling.x2Stride[dim];
        const uint32_t c1 = (s1 == 0U) ? 0U : ((s1 == innerLength) ? 1U : 2U);
        const uint32_t c2 = (s2 == 0U) ? 0U : ((s2 == innerLength) ? 1U : 2U);

        if (c1 == 2U || c2 == 2U) break;

        if (firstNonBcast) { x1Mode = c1; x2Mode = c2; firstNonBcast = false; }
        else if (x1Mode != c1 || x2Mode != c2) break;

        innerLength *= dimSize;
    }

    tiling.innerLength = innerLength;
    tiling.x1InnerContiguous = x1Mode;
    tiling.x2InnerContiguous = x2Mode;
}

bool MakeBroadcastShape(const gert::Shape &x1Shape, const gert::Shape &x2Shape, gert::Shape &outputShape)
{
    const size_t x1Rank = x1Shape.GetDimNum();
    const size_t x2Rank = x2Shape.GetDimNum();
    const size_t rank = std::max(x1Rank, x2Rank);
    if (rank > LESS_EQUAL_MAX_DIMS) return false;

    outputShape.SetDimNum(rank);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t x1Dim = (i < rank - x1Rank) ? 1 : x1Shape.GetDim(i - (rank - x1Rank));
        const int64_t x2Dim = (i < rank - x2Rank) ? 1 : x2Shape.GetDim(i - (rank - x2Rank));
        if (x1Dim < 0 || x2Dim < 0) return false;

        int64_t outputDim = 0;
        if (x1Dim == x2Dim)            outputDim = x1Dim;
        else if (x1Dim == 1)           outputDim = x2Dim;
        else if (x2Dim == 1)           outputDim = x1Dim;
        else                           return false;
        outputShape.SetDim(i, outputDim);
    }
    return true;
}

bool FillShapeAndStrides(const gert::Shape &x1Shape, const gert::Shape &x2Shape, LessEqualTilingData &tiling)
{
    gert::Shape outputShape;
    if (!MakeBroadcastShape(x1Shape, x2Shape, outputShape)) return false;

    const size_t x1Rank = x1Shape.GetDimNum();
    const size_t x2Rank = x2Shape.GetDimNum();
    const size_t rank = outputShape.GetDimNum();
    tiling.rank = static_cast<uint32_t>(rank);
    tiling.innerLength = 1;
    tiling.x1InnerContiguous = 0;
    tiling.x2InnerContiguous = 0;
    tiling.reserved = 0;

    uint64_t x1Dims[LESS_EQUAL_MAX_DIMS] = {0};
    uint64_t x2Dims[LESS_EQUAL_MAX_DIMS] = {0};
    for (size_t i = 0; i < LESS_EQUAL_MAX_DIMS; ++i) {
        tiling.outputShape[i] = 1;
        tiling.x1Stride[i] = 0;
        tiling.x2Stride[i] = 0;
    }

    for (size_t i = 0; i < rank; ++i) {
        const int64_t x1Dim = (i < rank - x1Rank) ? 1 : x1Shape.GetDim(i - (rank - x1Rank));
        const int64_t x2Dim = (i < rank - x2Rank) ? 1 : x2Shape.GetDim(i - (rank - x2Rank));
        const int64_t outputDim = outputShape.GetDim(i);
        if (x1Dim < 0 || x2Dim < 0 || outputDim < 0) return false;
        x1Dims[i] = static_cast<uint64_t>(x1Dim);
        x2Dims[i] = static_cast<uint64_t>(x2Dim);
        tiling.outputShape[i] = static_cast<uint64_t>(outputDim);
    }

    uint64_t totalLength = 1;
    for (size_t i = 0; i < rank; ++i) {
        if (!SafeMul(totalLength, tiling.outputShape[i], totalLength)) return false;
    }
    tiling.totalLength = totalLength;

    uint64_t x1RunningStride = 1, x2RunningStride = 1;
    for (size_t reverse = 0; reverse < rank; ++reverse) {
        const size_t i = rank - 1 - reverse;
        tiling.x1Stride[i] = (x1Dims[i] == 1) ? 0 : x1RunningStride;
        tiling.x2Stride[i] = (x2Dims[i] == 1) ? 0 : x2RunningStride;
        if (!SafeMul(x1RunningStride, x1Dims[i], x1RunningStride) ||
            !SafeMul(x2RunningStride, x2Dims[i], x2RunningStride)) return false;
    }
    tiling.x1Length = x1RunningStride;
    tiling.x2Length = x2RunningStride;

    bool sameShape = (x1Rank == x2Rank);
    if (sameShape) {
        for (size_t i = 0; i < x1Rank; ++i) {
            if (x1Shape.GetDim(i) != x2Shape.GetDim(i)) { sameShape = false; break; }
        }
    }

    if (sameShape)                            tiling.broadcastMode = LESS_EQUAL_SAME_SHAPE;
    else if (tiling.x1Length == 1 && tiling.totalLength != 0) tiling.broadcastMode = LESS_EQUAL_X1_SCALAR;
    else if (tiling.x2Length == 1 && tiling.totalLength != 0) tiling.broadcastMode = LESS_EQUAL_X2_SCALAR;
    else                                                        tiling.broadcastMode = LESS_EQUAL_GENERAL_BROADCAST;

    AnalyzeInnerSpanHost(tiling);
    return true;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;

    const gert::Tensor *tensorX1 = context->GetRequiredInputTensor(0);
    const gert::Tensor *tensorX2 = context->GetRequiredInputTensor(1);
    const gert::StorageShape *storageShapeX1 = context->GetRequiredInputShape(0);
    const gert::StorageShape *storageShapeX2 = context->GetRequiredInputShape(1);
    if (tensorX1 == nullptr || tensorX2 == nullptr ||
        storageShapeX1 == nullptr || storageShapeX2 == nullptr)
        return ge::GRAPH_FAILED;

    const ge::DataType dtypeX1 = tensorX1->GetDataType();
    const ge::DataType dtypeX2 = tensorX2->GetDataType();
    if (dtypeX1 != dtypeX2) return ge::GRAPH_FAILED;

    const uint32_t DT_X1 = static_cast<uint32_t>(dtypeX1);
    ASCENDC_TPL_SEL_PARAM(context, DT_X1);

    LessEqualTilingData *tiling = context->GetTilingData<LessEqualTilingData>();
    if (tiling == nullptr) return ge::GRAPH_FAILED;

    const gert::Shape &x1Shape = storageShapeX1->GetOriginShape();
    const gert::Shape &x2Shape = storageShapeX2->GetOriginShape();
    if (!FillShapeAndStrides(x1Shape, x2Shape, *tiling)) return ge::GRAPH_FAILED;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t coreNumAiv = platform.GetCoreNumAiv();
    const uint32_t availableCores = coreNumAiv > 0 ? static_cast<uint32_t>(coreNumAiv) : 1U;

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    tiling->tileLength = SelectTileLength(ubSize, dtypeX1, tiling->totalLength);

    uint32_t blockDim = 1;
    tiling->blockLength = 0;
    if (tiling->totalLength > 0) {
        // ---- Refined dynamic core scheduling ----
        //
        // Two constraints that work together:
        //
        //   Tile floor:   ceil(total / tile)
        //     Minimum cores needed so most process whole tiles.  Below this
        //     floor every core gets a partial tile → wasted vector capacity.
        //
        //   Underfill cap: total * DEN / (tile * NUM)
        //     Prevents over-splitting when total is only slightly above a tile
        //     boundary.  E.g. total = tile + 1 would give tileFloor = 2, but
        //     each core gets <1 partial tile.  We require each core to receive
        //     at least (NUM/DEN) of a full tile's worth of elements, otherwise
        //     we drop a core.
        //
        // Final usedCoreNum = min(tileFloor, underfillCap, availableCores).
        const uint64_t tileFloor = CeilDiv(tiling->totalLength, tiling->tileLength);
        uint64_t usedCoreNum = tileFloor;

        if (usedCoreNum > 1) {
            // Underfill threshold: each core gets at least 3/4 of a full tile.
            constexpr uint64_t TF_NUM = 3;
            constexpr uint64_t TF_DEN = 4;
            const uint64_t underfillCap =
                tiling->totalLength * TF_DEN / (tiling->tileLength * TF_NUM);
            if (underfillCap < usedCoreNum) {
                usedCoreNum = underfillCap;
            }
            if (usedCoreNum < 1) usedCoreNum = 1;
        }

        if (usedCoreNum > availableCores) {
            usedCoreNum = availableCores;
        }

        tiling->blockLength = AlignUp(
            CeilDiv(tiling->totalLength, static_cast<uint32_t>(usedCoreNum)),
            BOOL_BLOCK_ELEMENTS);
        blockDim = static_cast<uint32_t>(usedCoreNum);
    }
    if (context->SetBlockDim(blockDim) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;

    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) return ge::GRAPH_FAILED;
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr ||
        context->GetInputShape(1) == nullptr || context->GetOutputShape(0) == nullptr)
        return GRAPH_FAILED;

    gert::Shape outputShape;
    if (!MakeBroadcastShape(*context->GetInputShape(0), *context->GetInputShape(1), outputShape))
        return GRAPH_FAILED;
    *context->GetOutputShape(0) = outputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr || context->GetInputDataType(0) != context->GetInputDataType(1))
        return GRAPH_FAILED;
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
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(LessEqual);
}  // namespace ops

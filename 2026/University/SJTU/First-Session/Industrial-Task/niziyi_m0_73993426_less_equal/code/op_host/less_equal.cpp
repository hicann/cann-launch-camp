#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/less_equal_tiling.h"
#include "../op_kernel/tiling_key_less_equal.h"

namespace {
constexpr uint32_t X1_INDEX = 0;
constexpr uint32_t X2_INDEX = 1;
constexpr uint32_t Y_INDEX = 0;
constexpr uint32_t VECTOR_ALIGN = 128;

inline uint32_t CeilDivU32(uint32_t value, uint32_t divisor)
{
    return divisor == 0U ? 0U : (value + divisor - 1U) / divisor;
}

inline uint32_t AlignUpU32(uint32_t value, uint32_t align)
{
    return CeilDivU32(value, align) * align;
}

inline uint32_t MinU32(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

inline uint32_t MaxU32(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}

// V28: distinguish exact physical-shape equality from generic LINEAR
// broadcasting. Typed aggressive scheduling is only used after the proven
// V11b tiny-LINEAR policy has had priority.
bool ShapesExactlyEqual(const gert::Shape &shape1, const gert::Shape &shape2)
{
    const uint32_t rank1 = static_cast<uint32_t>(shape1.GetDimNum());
    const uint32_t rank2 = static_cast<uint32_t>(shape2.GetDimNum());
    if (rank1 != rank2) {
        return false;
    }
    for (uint32_t i = 0U; i < rank1; ++i) {
        if (shape1.GetDim(i) != shape2.GetDim(i)) {
            return false;
        }
    }
    return true;
}

struct BroadcastInfo {
    uint32_t rank = 1;
    uint32_t total = 1;
    uint32_t outShape[LESS_EQUAL_MAX_RANK] = {};
    uint32_t x1Stride[LESS_EQUAL_MAX_RANK] = {};
    uint32_t x2Stride[LESS_EQUAL_MAX_RANK] = {};
};

bool BuildBroadcastInfo(const gert::Shape &shape1,
                        const gert::Shape &shape2,
                        BroadcastInfo &info)
{
    const uint32_t rank1 = static_cast<uint32_t>(shape1.GetDimNum());
    const uint32_t rank2 = static_cast<uint32_t>(shape2.GetDimNum());
    const uint32_t rank = MaxU32(rank1, rank2);
    if (rank > LESS_EQUAL_MAX_RANK) {
        return false;
    }

    if (rank == 0U) {
        info.rank = 1U;
        info.total = 1U;
        info.outShape[0] = 1U;
        info.x1Stride[0] = 0U;
        info.x2Stride[0] = 0U;
        return true;
    }

    uint32_t dim1[LESS_EQUAL_MAX_RANK] = {};
    uint32_t dim2[LESS_EQUAL_MAX_RANK] = {};
    for (uint32_t i = 0U; i < rank; ++i) {
        dim1[i] = 1U;
        dim2[i] = 1U;
    }
    for (uint32_t i = 0U; i < rank1; ++i) {
        const int64_t dim = shape1.GetDim(i);
        if (dim < 0 || static_cast<uint64_t>(dim) > UINT32_MAX) {
            return false;
        }
        dim1[rank - rank1 + i] = static_cast<uint32_t>(dim);
    }
    for (uint32_t i = 0U; i < rank2; ++i) {
        const int64_t dim = shape2.GetDim(i);
        if (dim < 0 || static_cast<uint64_t>(dim) > UINT32_MAX) {
            return false;
        }
        dim2[rank - rank2 + i] = static_cast<uint32_t>(dim);
    }

    uint32_t mergedShape[LESS_EQUAL_MAX_RANK] = {};
    uint8_t mergedPattern[LESS_EQUAL_MAX_RANK] = {};
    uint32_t mergedRank = 0U;

    for (uint32_t i = 0U; i < rank; ++i) {
        const uint32_t d1 = dim1[i];
        const uint32_t d2 = dim2[i];
        uint32_t outDim = 0U;
        if (d1 == d2) {
            outDim = d1;
        } else if (d1 == 1U) {
            outDim = d2;
        } else if (d2 == 1U) {
            outDim = d1;
        } else {
            return false;
        }

        const bool x1Broadcast = d1 == 1U && outDim != 1U;
        const bool x2Broadcast = d2 == 1U && outDim != 1U;
        const uint8_t pattern = static_cast<uint8_t>(
            (x1Broadcast ? 1U : 0U) | (x2Broadcast ? 2U : 0U));

        if (mergedRank > 0U && mergedPattern[mergedRank - 1U] == pattern) {
            const uint64_t product = static_cast<uint64_t>(mergedShape[mergedRank - 1U]) * outDim;
            if (product > UINT32_MAX) {
                return false;
            }
            mergedShape[mergedRank - 1U] = static_cast<uint32_t>(product);
        } else {
            if (mergedRank >= LESS_EQUAL_MAX_RANK) {
                return false;
            }
            mergedShape[mergedRank] = outDim;
            mergedPattern[mergedRank] = pattern;
            ++mergedRank;
        }
    }

    info.rank = mergedRank == 0U ? 1U : mergedRank;
    uint64_t total = 1U;
    for (uint32_t i = 0U; i < info.rank; ++i) {
        info.outShape[i] = mergedShape[i];
        total *= mergedShape[i];
        if (total > UINT32_MAX) {
            return false;
        }
    }
    info.total = static_cast<uint32_t>(total);

    uint64_t running1 = 1U;
    uint64_t running2 = 1U;
    for (int32_t i = static_cast<int32_t>(info.rank) - 1; i >= 0; --i) {
        const bool x1Broadcast = (mergedPattern[i] & 1U) != 0U;
        const bool x2Broadcast = (mergedPattern[i] & 2U) != 0U;
        if (running1 > UINT32_MAX || running2 > UINT32_MAX) {
            return false;
        }
        info.x1Stride[i] = x1Broadcast ? 0U : static_cast<uint32_t>(running1);
        info.x2Stride[i] = x2Broadcast ? 0U : static_cast<uint32_t>(running2);
        if (!x1Broadcast) {
            running1 *= info.outShape[i];
        }
        if (!x2Broadcast) {
            running2 *= info.outShape[i];
        }
    }
    return true;
}

uint32_t SelectTileLength(ge::DataType dtype, uint64_t ubSize)
{
    uint32_t cap = 4096U;
    uint32_t bytesPerElement = 22U;
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_INT8) {
        cap = 8192U;
        bytesPerElement = 18U;
    }

    const uint64_t reserve = 24U * 1024U;
    uint64_t available = ubSize > reserve ? ubSize - reserve : 64U * 1024U;
    uint64_t byUb = available / bytesPerElement;
    if (byUb > cap) {
        byUb = cap;
    }
    uint32_t tile = static_cast<uint32_t>(byUb);
    tile = tile / VECTOR_ALIGN * VECTOR_ALIGN;
    if (tile < VECTOR_ALIGN) {
        tile = VECTOR_ALIGN;
    }
    return tile;
}

uint32_t SelectCoreNum(uint32_t totalLength, uint32_t outerLength,
                       uint32_t mode, uint32_t tileLength,
                       uint32_t platformCoreNum)
{
    if (totalLength == 0U) {
        return 1U;
    }
    uint32_t desired = 1U;
    if (totalLength > tileLength) {
        desired = CeilDivU32(totalLength, 32768U);
        if (desired == 0U) {
            desired = 1U;
        }
    }
    desired = MinU32(desired, platformCoreNum);
    if (mode != LESS_EQUAL_MODE_LINEAR) {
        desired = MinU32(desired, outerLength == 0U ? 1U : outerLength);
    }
    return desired == 0U ? 1U : desired;
}

uint32_t SelectStaticHalfTargetPerCore(uint32_t totalLength)
{
    if (totalLength <= 256U) {
        return 32U;
    }
    if (totalLength <= 1024U) {
        return 128U;
    }
    if (totalLength <= 4096U) {
        return 256U;
    }
    return 512U;
}

}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t platformCoreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (platformCoreNum == 0U) {
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSize = 0U;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const gert::Tensor *x1Tensor = context->GetRequiredInputTensor(X1_INDEX);
    const gert::Tensor *x2Tensor = context->GetRequiredInputTensor(X2_INDEX);
    const gert::StorageShape *x1StorageShape = context->GetInputShape(X1_INDEX);
    const gert::StorageShape *x2StorageShape = context->GetInputShape(X2_INDEX);
    if (x1Tensor == nullptr || x2Tensor == nullptr ||
        x1StorageShape == nullptr || x2StorageShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtype1 = x1Tensor->GetDataType();
    const ge::DataType dtype2 = x2Tensor->GetDataType();
    if (dtype1 != dtype2) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &x1Shape = x1StorageShape->GetStorageShape();
    const gert::Shape &x2Shape = x2StorageShape->GetStorageShape();
    const bool exactSameShape = ShapesExactlyEqual(x1Shape, x2Shape);

    BroadcastInfo broadcast;
    if (!BuildBroadcastInfo(x1Shape, x2Shape, broadcast)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t mode = LESS_EQUAL_MODE_ROW;
    const uint32_t innerLength = broadcast.outShape[broadcast.rank - 1U];
    const uint32_t outerLength = innerLength == 0U ? 0U : broadcast.total / innerLength;
    if (broadcast.rank == 1U) {
        mode = LESS_EQUAL_MODE_LINEAR;
    } else if (broadcast.rank == 2U &&
               broadcast.x1Stride[0] == 0U && broadcast.x1Stride[1] == 1U &&
               broadcast.x2Stride[0] == innerLength && broadcast.x2Stride[1] == 1U) {
        mode = LESS_EQUAL_MODE_REUSE_X1;
    } else if (broadcast.rank == 2U &&
               broadcast.x2Stride[0] == 0U && broadcast.x2Stride[1] == 1U &&
               broadcast.x1Stride[0] == innerLength && broadcast.x1Stride[1] == 1U) {
        mode = LESS_EQUAL_MODE_REUSE_X2;
    }

    uint32_t tileLength = SelectTileLength(dtype1, ubSize);
    uint32_t usedCoreNum = SelectCoreNum(broadcast.total, outerLength, mode,
                                         tileLength, platformCoreNum);
    uint32_t workPerCore = 0U;
    if (broadcast.total == 0U) {
        usedCoreNum = 1U;
        workPerCore = 0U;
    } else if (mode == LESS_EQUAL_MODE_LINEAR) {
        workPerCore = AlignUpU32(CeilDivU32(broadcast.total, usedCoreNum), VECTOR_ALIGN);
        usedCoreNum = CeilDivU32(broadcast.total, workPerCore);
    } else {
        workPerCore = CeilDivU32(outerLength, usedCoreNum);
        usedCoreNum = CeilDivU32(outerLength, workPerCore);
    }
    if (usedCoreNum == 0U) {
        usedCoreNum = 1U;
    }

    // V11B: use moderate parallelism only for tiny one-dimensional
    // LINEAR outputs. Unlike the V10b-P2 diagnostic probe, launch only cores
    // that own valid output elements; all other shapes retain the V3 policy.
    if (mode == LESS_EQUAL_MODE_LINEAR &&
        broadcast.total > 0U && broadcast.total <= 128U) {
        constexpr uint32_t TINY_LINEAR_CORE_GRAIN = 32U;
        usedCoreNum = CeilDivU32(broadcast.total, TINY_LINEAR_CORE_GRAIN);
        usedCoreNum = MinU32(usedCoreNum, platformCoreNum);
        if (usedCoreNum == 0U) {
            usedCoreNum = 1U;
        }
        workPerCore = CeilDivU32(broadcast.total, usedCoreNum);
    }

    // V28 typed same-shape policy. Important ordering rule: the V11b tiny
    // policy above always wins for total<=128, preserving TP2. Only larger
    // exact-shape light-type workloads use aggressive AIV parallelism.
    if (mode == LESS_EQUAL_MODE_LINEAR && exactSameShape &&
        broadcast.total > 128U && (dtype1 == ge::DT_FLOAT16 || dtype1 == ge::DT_INT8)) {
        constexpr uint32_t SAME_SHAPE_CORE_ALIGN = 32U;
        workPerCore = AlignUpU32(
            CeilDivU32(broadcast.total, platformCoreNum),
            SAME_SHAPE_CORE_ALIGN);
        if (workPerCore == 0U) {
            workPerCore = SAME_SHAPE_CORE_ALIGN;
        }
        usedCoreNum = CeilDivU32(broadcast.total, workPerCore);
        usedCoreNum = MinU32(usedCoreNum, platformCoreNum);
        if (usedCoreNum == 0U) {
            usedCoreNum = 1U;
        }
    }

    const bool useStaticHalf =
        mode == LESS_EQUAL_MODE_LINEAR && exactSameShape &&
        dtype1 == ge::DT_FLOAT16 && broadcast.total > 128U &&
        broadcast.total <= 16384U;
    if (useStaticHalf) {
        const uint32_t targetPerCore =
            SelectStaticHalfTargetPerCore(broadcast.total);
        usedCoreNum = MinU32(CeilDivU32(broadcast.total, targetPerCore),
                             platformCoreNum);
        if (usedCoreNum == 0U) {
            usedCoreNum = 1U;
        }
        workPerCore = AlignUpU32(
            CeilDivU32(broadcast.total, usedCoreNum), 32U);
        usedCoreNum = CeilDivU32(broadcast.total, workPerCore);
    }

    const uint32_t DT_X1 = static_cast<uint32_t>(dtype1);

    // V31 priority:
    // 1) use a static-UB binary for the proven FP16 same-shape hot range;
    // 2) preserve V30's multi-tile DB+precomputed-ones binary;
    // 3) compile proven exact-shape FP16/INT8 single-tile workloads into a
    //    minimal binary that carries no broadcast shape/stride state;
    // 4) keep every other case on the unchanged V30 normal binary.
    uint32_t SCH_MODE = LESS_EQUAL_SCH_NORMAL;
    if (useStaticHalf) {
        SCH_MODE = LESS_EQUAL_SCH_STATIC_HALF;
    } else if (mode == LESS_EQUAL_MODE_LINEAR && workPerCore > tileLength) {
        SCH_MODE = LESS_EQUAL_SCH_LINEAR_DB_ONES;
    } else if (mode == LESS_EQUAL_MODE_LINEAR && exactSameShape &&
               dtype1 == ge::DT_INT8 && broadcast.total > 128U &&
               broadcast.total <= 4096U) {
        SCH_MODE = LESS_EQUAL_SCH_FAST_LIGHT;
        constexpr uint32_t fastAlign = 256U;
        const uint32_t maxCoreSegment = MinU32(workPerCore, broadcast.total);
        const uint32_t requiredTile = AlignUpU32(maxCoreSegment, fastAlign);
        tileLength = MinU32(tileLength, requiredTile);
        if (tileLength < fastAlign) {
            tileLength = fastAlign;
        }
    }
    ASCENDC_TPL_SEL_PARAM(context, DT_X1, SCH_MODE);

    LessEqualTilingData *tiling = context->GetTilingData<LessEqualTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->totalLength = broadcast.total;
    tiling->innerLength = innerLength;
    tiling->outerLength = outerLength;
    tiling->workPerCore = workPerCore;
    tiling->tileLength = tileLength;
    tiling->rank = broadcast.rank;
    tiling->mode = mode;
    tiling->usedCoreNum = usedCoreNum;
    for (uint32_t i = 0U; i < LESS_EQUAL_MAX_RANK; ++i) {
        tiling->outShape[i] = broadcast.outShape[i];
        tiling->x1Stride[i] = broadcast.x1Stride[i];
        tiling->x2Stride[i] = broadcast.x2Stride[i];
    }

    context->SetBlockDim(usedCoreNum);
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0U;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *shape1 = context->GetInputShape(X1_INDEX);
    const gert::Shape *shape2 = context->GetInputShape(X2_INDEX);
    gert::Shape *output = context->GetOutputShape(Y_INDEX);
    if (shape1 == nullptr || shape2 == nullptr || output == nullptr) {
        return GRAPH_FAILED;
    }

    const uint32_t rank1 = static_cast<uint32_t>(shape1->GetDimNum());
    const uint32_t rank2 = static_cast<uint32_t>(shape2->GetDimNum());
    const uint32_t outRank = MaxU32(rank1, rank2);
    output->SetDimNum(outRank);
    for (uint32_t i = 0U; i < outRank; ++i) {
        const int32_t index1 = static_cast<int32_t>(i) -
            static_cast<int32_t>(outRank - rank1);
        const int32_t index2 = static_cast<int32_t>(i) -
            static_cast<int32_t>(outRank - rank2);
        const int64_t d1 = index1 >= 0 ? shape1->GetDim(index1) : 1;
        const int64_t d2 = index2 >= 0 ? shape2->GetDim(index2) : 1;
        int64_t outDim = 0;
        if (d1 == d2) {
            outDim = d1;
        } else if (d1 == 1) {
            outDim = d2;
        } else if (d2 == 1) {
            outDim = d1;
        } else {
            return GRAPH_FAILED;
        }
        output->SetDim(i, outDim);
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(Y_INDEX, ge::DT_BOOL);
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
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(LessEqual);
}  // namespace ops

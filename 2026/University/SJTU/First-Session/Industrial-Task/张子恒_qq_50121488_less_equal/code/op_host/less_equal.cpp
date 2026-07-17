// LessEqual Host side: operator registration, shape/dtype inference and tiling.
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstdint>

#include "../op_kernel/less_equal_tiling.h"
#include "../op_kernel/tiling_key_less_equal.h"

namespace optiling {
namespace {
constexpr uint64_t UINT32_LIMIT = static_cast<uint64_t>(UINT32_MAX);

inline uint32_t MaxU32(uint32_t a, uint32_t b) { return a > b ? a : b; }
inline uint32_t MinU32(uint32_t a, uint32_t b) { return a < b ? a : b; }
inline uint32_t CeilDivU32(uint32_t value, uint32_t divisor) {
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}
inline uint32_t AlignDownU32(uint32_t value, uint32_t align) {
    return align == 0 ? value : value / align * align;
}
inline uint32_t GetStorageRank(const gert::StorageShape *shape) {
    return static_cast<uint32_t>(shape->GetStorageShape().GetDimNum());
}
inline uint64_t GetStorageDimU64(const gert::StorageShape *shape, uint32_t index) {
    const int64_t dim = shape->GetStorageShape().GetDim(index);
    return dim > 0 ? static_cast<uint64_t>(dim) : 0U;
}
inline uint64_t MergeBroadcastDim(uint64_t current, uint64_t candidate) {
    if (current == 1) return candidate;
    if (candidate == 1 || candidate == current) return current;
    return current > candidate ? current : candidate;
}

// Align an input shape to the output shape from the right and compute broadcast strides.
// All products are calculated in uint64_t, then checked before being narrowed into tiling.
bool FillBroadcastStride(const gert::StorageShape *shape,
                         uint32_t outputRank,
                         uint32_t *broadcastStride) {
    const uint32_t storageRank = GetStorageRank(shape);
    const uint32_t inputRank = MinU32(storageRank, outputRank);
    const uint32_t inputStart = storageRank - inputRank;
    uint64_t rawDims[LESS_EQUAL_MAX_RANK] = {0};
    uint64_t rawStride[LESS_EQUAL_MAX_RANK] = {0};

    for (uint32_t i = 0; i < inputRank; ++i) {
        rawDims[i] = GetStorageDimU64(shape, inputStart + i);
        if (rawDims[i] > UINT32_LIMIT) {
            return false;
        }
    }

    uint64_t running = 1;
    for (int32_t i = static_cast<int32_t>(inputRank) - 1; i >= 0; --i) {
        if (running > UINT32_LIMIT) {
            return false;
        }
        rawStride[i] = running;
        if (rawDims[i] != 0 && running > UINT32_LIMIT / rawDims[i]) {
            return false;
        }
        running = rawDims[i] == 0 ? 0 : running * rawDims[i];
    }

    const uint32_t rankOffset = outputRank - inputRank;
    for (uint32_t i = 0; i < outputRank; ++i) {
        if (i < rankOffset) {
            broadcastStride[i] = 0;
        } else {
            const uint64_t inputDim = rawDims[i - rankOffset];
            const uint64_t stride = rawStride[i - rankOffset];
            if (stride > UINT32_LIMIT) {
                return false;
            }
            broadcastStride[i] = inputDim == 1 ? 0 : static_cast<uint32_t>(stride);
        }
    }
    for (uint32_t i = outputRank; i < LESS_EQUAL_MAX_RANK; ++i) {
        broadcastStride[i] = 0;
    }
    return true;
}

// Compute the broadcast output shape. The kernel tiling struct keeps uint32_t fields for
// compatibility, so host tiling must reject any value that cannot be represented safely.
bool FillOutputShape(const gert::StorageShape *x1Shape,
                     const gert::StorageShape *x2Shape,
                     LessEqualTilingData *tiling) {
    const uint32_t x1Rank = GetStorageRank(x1Shape);
    const uint32_t x2Rank = GetStorageRank(x2Shape);
    const uint32_t actualOutputRank = MaxU32(x1Rank, x2Rank);
    const uint32_t outputRank = MinU32(actualOutputRank, LESS_EQUAL_MAX_RANK);

    const uint32_t x1Start = x1Rank > outputRank ? x1Rank - outputRank : 0;
    const uint32_t x2Start = x2Rank > outputRank ? x2Rank - outputRank : 0;
    const uint32_t effectiveX1Rank = MinU32(x1Rank, outputRank);
    const uint32_t effectiveX2Rank = MinU32(x2Rank, outputRank);
    const uint32_t x1Offset = outputRank - effectiveX1Rank;
    const uint32_t x2Offset = outputRank - effectiveX2Rank;

    tiling->rank = outputRank;
    tiling->length = 1;
    for (uint32_t i = 0; i < outputRank; ++i) {
        uint64_t outputDim = 1;
        if (i >= x1Offset) {
            outputDim = MergeBroadcastDim(outputDim,
                GetStorageDimU64(x1Shape, x1Start + i - x1Offset));
        }
        if (i >= x2Offset) {
            outputDim = MergeBroadcastDim(outputDim,
                GetStorageDimU64(x2Shape, x2Start + i - x2Offset));
        }
        if (outputDim > UINT32_LIMIT) {
            return false;
        }
        if (outputDim != 0 && static_cast<uint64_t>(tiling->length) > UINT32_LIMIT / outputDim) {
            return false;
        }
        tiling->outputShape[i] = static_cast<uint32_t>(outputDim);
        tiling->length = outputDim == 0 ? 0 : tiling->length * static_cast<uint32_t>(outputDim);
    }
    for (uint32_t i = outputRank; i < LESS_EQUAL_MAX_RANK; ++i) {
        tiling->outputShape[i] = 1;
    }
    return true;
}

// Largest suffix that is still contiguous after broadcasting; used by the kernel to DMA spans.
bool GetBroadcastContiguousSpan(const LessEqualTilingData *tiling,
                                const uint32_t *broadcastStride,
                                uint32_t *contiguousSpan) {
    uint64_t span = 1;
    for (int32_t dim = static_cast<int32_t>(tiling->rank) - 1; dim >= 0; --dim) {
        const uint32_t dimSize = tiling->outputShape[dim];
        if (dimSize == 0) {
            *contiguousSpan = 0;
            return true;
        }
        if (dimSize == 1) {
            continue;
        }
        if (static_cast<uint64_t>(broadcastStride[dim]) != span) {
            break;
        }
        if (span > UINT32_LIMIT / dimSize) {
            return false;
        }
        span *= dimSize;
    }
    if (span > UINT32_LIMIT) {
        return false;
    }
    *contiguousSpan = static_cast<uint32_t>(span);
    return true;
}
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t coreNum = platform.GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const gert::Tensor *tensorX1 = context->GetRequiredInputTensor(0);
    const gert::Tensor *tensorX2 = context->GetRequiredInputTensor(1);
    const gert::StorageShape *shapeX1 = context->GetInputShape(0);
    const gert::StorageShape *shapeX2 = context->GetInputShape(1);
    const ge::DataType dtype = tensorX1->GetDataType();
    const uint32_t dtypeSize = static_cast<uint32_t>(ge::GetSizeByDataType(dtype));

    OP_TILING_CHECK(dtypeSize == 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "unsupported dtype"),
                    return ge::GRAPH_FAILED);

    const uint32_t DT_X1 = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_X1);

    LessEqualTilingData *tiling = context->GetTilingData<LessEqualTilingData>();
    OP_TILING_CHECK(tiling == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "tiling data is null"),
                    return ge::GRAPH_FAILED);

    const bool outputShapeOk = FillOutputShape(shapeX1, shapeX2, tiling);
    OP_TILING_CHECK(!outputShapeOk,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                        "broadcast output shape exceeds uint32 range"),
                    return ge::GRAPH_FAILED);

    const int64_t x1ShapeSize = tensorX1->GetShapeSize();
    const int64_t x2ShapeSize = tensorX2->GetShapeSize();
    OP_TILING_CHECK(x1ShapeSize < 0 || x2ShapeSize < 0 ||
                        static_cast<uint64_t>(x1ShapeSize) > UINT32_LIMIT ||
                        static_cast<uint64_t>(x2ShapeSize) > UINT32_LIMIT,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                        "input tensor shape size exceeds uint32 range"),
                    return ge::GRAPH_FAILED);
    tiling->x1Numel = static_cast<uint32_t>(x1ShapeSize);
    tiling->x2Numel = static_cast<uint32_t>(x2ShapeSize);

    const bool x1StrideOk = FillBroadcastStride(shapeX1, tiling->rank, tiling->x1Stride);
    const bool x2StrideOk = FillBroadcastStride(shapeX2, tiling->rank, tiling->x2Stride);
    OP_TILING_CHECK(!x1StrideOk || !x2StrideOk,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                        "broadcast stride exceeds uint32 range"),
                    return ge::GRAPH_FAILED);

    const bool x1SpanOk =
        GetBroadcastContiguousSpan(tiling, tiling->x1Stride, &tiling->x1ContiguousSpan);
    const bool x2SpanOk =
        GetBroadcastContiguousSpan(tiling, tiling->x2Stride, &tiling->x2ContiguousSpan);
    OP_TILING_CHECK(!x1SpanOk || !x2SpanOk,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                        "contiguous span exceeds uint32 range"),
                    return ge::GRAPH_FAILED);

    tiling->contiguous =
        (tiling->x1Numel == tiling->length && tiling->x2Numel == tiling->length) ? 1U : 0U;

    constexpr uint32_t vectorAlignElements = 128U;
    constexpr uint64_t selectReserveBytes = 8U * 1024U;
    const uint32_t scratchElementBytes = MaxU32(dtypeSize, 2U);
    const uint64_t bytesPerElement = static_cast<uint64_t>(dtypeSize) * 2U + 1U +
        static_cast<uint64_t>(scratchElementBytes) * 2U;
    const uint64_t usableUbSize = ubSize > selectReserveBytes ? ubSize - selectReserveBytes : 0;

    uint32_t tileLength = vectorAlignElements;
    if (usableUbSize > 0 && bytesPerElement > 0) {
        uint32_t ubElements = static_cast<uint32_t>(usableUbSize / bytesPerElement);
        ubElements = AlignDownU32(ubElements, vectorAlignElements);
        if (ubElements > 32768U) {
            ubElements = AlignDownU32(32768U, vectorAlignElements);
        }
        tileLength = ubElements >= vectorAlignElements ? ubElements : vectorAlignElements;
    }
    tiling->tileLength = tileLength;

    uint32_t blockDim = 1;
    if (tiling->length > 0) {
        blockDim = CeilDivU32(tiling->length, 1024U);
        blockDim = MinU32(blockDim == 0 ? 1 : blockDim, static_cast<uint32_t>(coreNum));
        blockDim = MinU32(blockDim, tiling->length);
        if (blockDim == 0) blockDim = 1;
    }
    OP_TILING_CHECK(blockDim == 0 || tiling->tileLength == 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                        "invalid blockDim or tileLength"),
                    return ge::GRAPH_FAILED);
    tiling->blockDim = blockDim;
    context->SetBlockDim(blockDim);
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
namespace {
inline size_t MaxSizeT(size_t a, size_t b) { return a > b ? a : b; }
inline int64_t MergeBroadcastDimI64(int64_t current, int64_t candidate) {
    if (current == 1) return candidate;
    if (candidate == 1 || candidate == current) return current;
    return current > candidate ? current : candidate;
}
}  // namespace

static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *x1Shape = context->GetInputShape(0);
    const gert::Shape *x2Shape = context->GetInputShape(1);
    gert::Shape *outputShape = context->GetOutputShape(0);
    const size_t x1Rank = x1Shape->GetDimNum();
    const size_t x2Rank = x2Shape->GetDimNum();
    const size_t outputRank = MaxSizeT(x1Rank, x2Rank);
    outputShape->SetDimNum(outputRank);
    const size_t x1Offset = outputRank - x1Rank;
    const size_t x2Offset = outputRank - x2Rank;
    for (size_t i = 0; i < outputRank; ++i) {
        int64_t outputDim = 1;
        if (i >= x1Offset) outputDim = MergeBroadcastDimI64(outputDim, x1Shape->GetDim(i - x1Offset));
        if (i >= x2Offset) outputDim = MergeBroadcastDimI64(outputDim, x2Shape->GetDim(i - x2Offset));
        outputShape->SetDim(i, outputDim);
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, ge::DT_BOOL);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class LessEqual : public OpDef {
public:
    explicit LessEqual(const char *name) : OpDef(name) {
        this->Input("x1").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(LessEqual);
}  // namespace ops
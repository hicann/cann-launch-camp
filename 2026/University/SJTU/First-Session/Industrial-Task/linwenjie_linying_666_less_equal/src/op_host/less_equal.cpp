// Host-side registration, shape inference and tiling for LessEqual.
#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/less_equal_tiling.h"
#include "../op_kernel/tiling_key_less_equal.h"

namespace {
constexpr uint64_t BOOL_BLOCK_ELEMENTS = 32;  // bool occupies one byte; align core boundaries to 32 B.
constexpr uint32_t FALLBACK_TILE_ELEMENTS = 4096;
constexpr uint32_t MAX_TILE_ELEMENTS = 8192;
constexpr uint32_t MAX_INT32_TILE_ELEMENTS = 12288;
constexpr uint32_t MIN_TILE_ELEMENTS = 128;
constexpr uint32_t TILE_ALIGNMENT_ELEMENTS = 128;  // Safe for both fp16/fp32 Compare padding.
constexpr uint32_t DOUBLE_BUFFER_COUNT = 2;
constexpr uint64_t UB_SAFETY_RESERVE_BYTES = 16U * 1024U;
// Select mode 1/2 on Atlas A2 requires 8 KiB of additional UB scratch space.
constexpr uint64_t SELECT_INTERNAL_RESERVE_BYTES = 8U * 1024U;
constexpr uint64_t MIN_ELEMENTS_PER_CORE = 2048;
constexpr uint64_t SMALL_SAME_SHAPE_MAX_ELEMENTS = 2048;
constexpr uint64_t MEDIUM_SCALAR_MAX_ELEMENTS = 8192;
constexpr uint64_t MEDIUM_INT32_SCALAR_MAX_ELEMENTS = 4096;

inline uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    if (divisor == 0) {
        return 0;
    }
    return value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

inline uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}

bool SafeMul(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (lhs == 0 || rhs == 0) {
        result = 0;
        return true;
    }
    if (lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}


uint32_t GetDataTypeSize(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
            return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        case ge::DT_INT8:
            return 1;
        default:
            return 0;
    }
}

uint32_t GetVectorElementsPerRepeat(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 64;  // 256 B / 4 B.
        case ge::DT_FLOAT16:
        case ge::DT_INT8:
            return 128; // fp16 compare, or int8 converted to fp16.
        default:
            return MIN_TILE_ELEMENTS;
    }
}

uint32_t SelectSmallTileLength(uint64_t totalLength, ge::DataType dtype)
{
    const uint32_t alignment = GetVectorElementsPerRepeat(dtype);
    const uint64_t padded = AlignUp(std::max<uint64_t>(1, totalLength), alignment);
    return static_cast<uint32_t>(std::max<uint64_t>(alignment, padded));
}

bool UsesVectorSelectPath(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT ||
           dtype == ge::DT_INT8 || dtype == ge::DT_INT32;
}

uint64_t EstimateUbBytes(uint32_t tileElements, uint32_t typeBytes, ge::DataType dtype)
{
    const uint64_t inputBufferBytes = AlignUp(static_cast<uint64_t>(tileElements) * typeBytes, 32);
    const uint64_t outputBufferBytes = AlignUp(tileElements, 32);
    uint64_t total = DOUBLE_BUFFER_COUNT * (2U * inputBufferBytes + outputBufferBytes);

    if (UsesVectorSelectPath(dtype)) {
        total += AlignUp(CeilDiv(tileElements, 8U), 32);  // Compare bit mask.
    }
    if (dtype == ge::DT_FLOAT) {
        total += AlignUp(static_cast<uint64_t>(tileElements) * sizeof(uint16_t), 32);
    } else if (dtype == ge::DT_INT8) {
        total += 2U * AlignUp(static_cast<uint64_t>(tileElements) * sizeof(uint16_t), 32);
    }
    return total;
}

uint32_t SelectTileLength(uint64_t ubSize, ge::DataType dtype)
{
    const uint32_t typeBytes = GetDataTypeSize(dtype);
    if (typeBytes == 0 || ubSize == 0) {
        return FALLBACK_TILE_ELEMENTS;
    }

    uint64_t reserve = UB_SAFETY_RESERVE_BYTES;
    if (UsesVectorSelectPath(dtype)) {
        reserve += SELECT_INTERNAL_RESERVE_BYTES;
    }
    const uint64_t usableUb = ubSize > reserve ? ubSize - reserve : ubSize;

    uint32_t tile = (dtype == ge::DT_INT32) ? MAX_INT32_TILE_ELEMENTS : MAX_TILE_ELEMENTS;
    while (tile > MIN_TILE_ELEMENTS && EstimateUbBytes(tile, typeBytes, dtype) > usableUb) {
        tile -= TILE_ALIGNMENT_ELEMENTS;
    }
    if (EstimateUbBytes(tile, typeBytes, dtype) > usableUb) {
        return MIN_TILE_ELEMENTS;
    }
    return tile;
}

void AnalyzeInnerSpanHost(LessEqualTilingData &tiling)
{
    int32_t x1Mode = -1;  // -1 undecided, 0 constant, 1 contiguous.
    int32_t x2Mode = -1;
    uint64_t innerLength = 1;

    for (uint32_t reverse = 0; reverse < tiling.rank; ++reverse) {
        const uint32_t dimension = tiling.rank - 1U - reverse;
        const uint64_t outputDimension = tiling.outputShape[dimension];
        if (outputDimension == 1U) {
            continue;
        }

        const int32_t currentX1Mode = (tiling.x1Stride[dimension] == 0U)
            ? 0
            : ((tiling.x1Stride[dimension] == innerLength) ? 1 : 2);
        const int32_t currentX2Mode = (tiling.x2Stride[dimension] == 0U)
            ? 0
            : ((tiling.x2Stride[dimension] == innerLength) ? 1 : 2);
        if (currentX1Mode == 2 || currentX2Mode == 2 ||
            (x1Mode != -1 && x1Mode != currentX1Mode) ||
            (x2Mode != -1 && x2Mode != currentX2Mode)) {
            break;
        }

        if (x1Mode == -1) {
            x1Mode = currentX1Mode;
        }
        if (x2Mode == -1) {
            x2Mode = currentX2Mode;
        }
        innerLength *= outputDimension;
    }

    tiling.innerLength = innerLength;
    tiling.x1InnerContiguous = static_cast<uint32_t>(x1Mode == 1);
    tiling.x2InnerContiguous = static_cast<uint32_t>(x2Mode == 1);
}

bool MakeBroadcastShape(const gert::Shape &x1Shape, const gert::Shape &x2Shape, gert::Shape &outputShape)
{
    const size_t x1Rank = x1Shape.GetDimNum();
    const size_t x2Rank = x2Shape.GetDimNum();
    const size_t rank = std::max(x1Rank, x2Rank);
    if (rank > LESS_EQUAL_MAX_DIMS) {
        return false;
    }

    outputShape.SetDimNum(rank);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t x1Dim = (i < rank - x1Rank) ? 1 : x1Shape.GetDim(i - (rank - x1Rank));
        const int64_t x2Dim = (i < rank - x2Rank) ? 1 : x2Shape.GetDim(i - (rank - x2Rank));
        if (x1Dim < 0 || x2Dim < 0) {
            return false;
        }

        int64_t outputDim = 0;
        if (x1Dim == x2Dim) {
            outputDim = x1Dim;
        } else if (x1Dim == 1) {
            outputDim = x2Dim;
        } else if (x2Dim == 1) {
            outputDim = x1Dim;
        } else {
            return false;
        }
        outputShape.SetDim(i, outputDim);
    }
    return true;
}

bool FillShapeAndStrides(const gert::Shape &x1Shape, const gert::Shape &x2Shape, LessEqualTilingData &tiling)
{
    gert::Shape outputShape;
    if (!MakeBroadcastShape(x1Shape, x2Shape, outputShape)) {
        return false;
    }

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
        if (x1Dim < 0 || x2Dim < 0 || outputDim < 0) {
            return false;
        }
        x1Dims[i] = static_cast<uint64_t>(x1Dim);
        x2Dims[i] = static_cast<uint64_t>(x2Dim);
        tiling.outputShape[i] = static_cast<uint64_t>(outputDim);
    }

    uint64_t totalLength = 1;
    if (rank == 0) {  // A rank-0 tensor is a scalar with one element.
        totalLength = 1;
    } else {
        for (size_t i = 0; i < rank; ++i) {
            if (!SafeMul(totalLength, tiling.outputShape[i], totalLength)) {
                return false;
            }
        }
    }
    tiling.totalLength = totalLength;

    uint64_t x1RunningStride = 1;
    uint64_t x2RunningStride = 1;
    for (size_t reverse = 0; reverse < rank; ++reverse) {
        const size_t i = rank - 1 - reverse;
        tiling.x1Stride[i] = (x1Dims[i] == 1) ? 0 : x1RunningStride;
        tiling.x2Stride[i] = (x2Dims[i] == 1) ? 0 : x2RunningStride;
        if (!SafeMul(x1RunningStride, x1Dims[i], x1RunningStride) ||
            !SafeMul(x2RunningStride, x2Dims[i], x2RunningStride)) {
            return false;
        }
    }
    tiling.x1Length = x1RunningStride;
    tiling.x2Length = x2RunningStride;

    bool sameShape = (x1Rank == x2Rank);
    if (sameShape) {
        for (size_t i = 0; i < x1Rank; ++i) {
            if (x1Shape.GetDim(i) != x2Shape.GetDim(i)) {
                sameShape = false;
                break;
            }
        }
    }

    if (sameShape) {
        tiling.broadcastMode = LESS_EQUAL_SAME_SHAPE;
    } else if (tiling.x1Length == 1 && tiling.totalLength != 0) {
        tiling.broadcastMode = LESS_EQUAL_X1_SCALAR;
    } else if (tiling.x2Length == 1 && tiling.totalLength != 0) {
        tiling.broadcastMode = LESS_EQUAL_X2_SCALAR;
    } else {
        tiling.broadcastMode = LESS_EQUAL_GENERAL_BROADCAST;
    }
    AnalyzeInnerSpanHost(tiling);
    return true;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Tensor *tensorX1 = context->GetRequiredInputTensor(0);
    const gert::Tensor *tensorX2 = context->GetRequiredInputTensor(1);
    const gert::StorageShape *storageShapeX1 = context->GetRequiredInputShape(0);
    const gert::StorageShape *storageShapeX2 = context->GetRequiredInputShape(1);
    if (tensorX1 == nullptr || tensorX2 == nullptr || storageShapeX1 == nullptr || storageShapeX2 == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtypeX1 = tensorX1->GetDataType();
    const ge::DataType dtypeX2 = tensorX2->GetDataType();
    if (dtypeX1 != dtypeX2) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t DT_X1 = static_cast<uint32_t>(dtypeX1);

    LessEqualTilingData *tiling = context->GetTilingData<LessEqualTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &x1Shape = storageShapeX1->GetOriginShape();
    const gert::Shape &x2Shape = storageShapeX2->GetOriginShape();
    if (!FillShapeAndStrides(x1Shape, x2Shape, *tiling)) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t coreNumAiv = platform.GetCoreNumAiv();
    const uint32_t availableCores = coreNumAiv > 0 ? static_cast<uint32_t>(coreNumAiv) : 1U;

    const uint32_t vectorRepeatElements = GetVectorElementsPerRepeat(dtypeX1);
    uint32_t PATH_KIND = LESS_EQUAL_PATH_GENERAL;
    if (tiling->totalLength > 0 && tiling->totalLength <= vectorRepeatElements) {
        if (tiling->broadcastMode == LESS_EQUAL_SAME_SHAPE) {
            PATH_KIND = LESS_EQUAL_PATH_TINY_SAME;
        } else if (tiling->broadcastMode == LESS_EQUAL_X1_SCALAR) {
            PATH_KIND = LESS_EQUAL_PATH_TINY_X1_SCALAR;
        } else if (tiling->broadcastMode == LESS_EQUAL_X2_SCALAR) {
            PATH_KIND = LESS_EQUAL_PATH_TINY_X2_SCALAR;
        }
    }
    if (PATH_KIND == LESS_EQUAL_PATH_GENERAL &&
        tiling->totalLength > vectorRepeatElements) {
        const uint64_t scalarFastLimit = (dtypeX1 == ge::DT_INT32)
            ? MEDIUM_INT32_SCALAR_MAX_ELEMENTS
            : MEDIUM_SCALAR_MAX_ELEMENTS;
        if (tiling->totalLength <= scalarFastLimit) {
            if (tiling->broadcastMode == LESS_EQUAL_X1_SCALAR) {
                PATH_KIND = LESS_EQUAL_PATH_MEDIUM_X1_SCALAR;
            } else if (tiling->broadcastMode == LESS_EQUAL_X2_SCALAR) {
                PATH_KIND = LESS_EQUAL_PATH_MEDIUM_X2_SCALAR;
            }
        }
    }
    if (PATH_KIND == LESS_EQUAL_PATH_GENERAL &&
        tiling->broadcastMode == LESS_EQUAL_SAME_SHAPE &&
        tiling->totalLength > 0 &&
        tiling->totalLength <= SMALL_SAME_SHAPE_MAX_ELEMENTS) {
        PATH_KIND = LESS_EQUAL_PATH_SMALL_SAME;
    }

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (PATH_KIND == LESS_EQUAL_PATH_TINY_SAME ||
        PATH_KIND == LESS_EQUAL_PATH_TINY_X1_SCALAR ||
        PATH_KIND == LESS_EQUAL_PATH_TINY_X2_SCALAR) {
        // Static-tensor TINY kernels always execute exactly one 256-byte vector repeat.
        tiling->tileLength = vectorRepeatElements;
    } else if (PATH_KIND == LESS_EQUAL_PATH_SMALL_SAME ||
               PATH_KIND == LESS_EQUAL_PATH_MEDIUM_X1_SCALAR ||
               PATH_KIND == LESS_EQUAL_PATH_MEDIUM_X2_SCALAR) {
        tiling->tileLength = SelectSmallTileLength(tiling->totalLength, dtypeX1);
    } else {
        tiling->tileLength = SelectTileLength(ubSize, dtypeX1);
    }

    uint32_t blockDim = 1;
    tiling->blockLength = 0;
    if (tiling->totalLength > 0) {
        if (PATH_KIND != LESS_EQUAL_PATH_GENERAL) {
            // TINY, SMALL and MEDIUM_SCALAR paths are dedicated one-core implementations.
            tiling->blockLength = tiling->totalLength;
            blockDim = 1;
        } else {
            // Preserve the v13-A general partition for dense/scalar work. For true broadcast,
            // keep each core on whole inner spans whenever possible so cores do not start/end
            // in the middle of a broadcast row and repeat scalar coordinate work.
            const uint64_t workBasedBlocks = CeilDiv(tiling->totalLength, MIN_ELEMENTS_PER_CORE);
            uint32_t desiredBlocks = static_cast<uint32_t>(
                std::min<uint64_t>(availableCores, std::max<uint64_t>(1, workBasedBlocks)));
            if (tiling->broadcastMode == LESS_EQUAL_GENERAL_BROADCAST &&
                tiling->innerLength >= BOOL_BLOCK_ELEMENTS &&
                tiling->totalLength % tiling->innerLength == 0U) {
                const uint64_t rowCount = tiling->totalLength / tiling->innerLength;
                desiredBlocks = static_cast<uint32_t>(
                    std::min<uint64_t>(desiredBlocks, std::max<uint64_t>(1, rowCount)));
                const uint64_t rowsPerBlock = CeilDiv(rowCount, desiredBlocks);
                tiling->blockLength = rowsPerBlock * tiling->innerLength;
                blockDim = static_cast<uint32_t>(CeilDiv(rowCount, rowsPerBlock));
            } else {
                tiling->blockLength = AlignUp(CeilDiv(tiling->totalLength, desiredBlocks), BOOL_BLOCK_ELEMENTS);
                blockDim = static_cast<uint32_t>(CeilDiv(tiling->totalLength, tiling->blockLength));
            }
        }
    }

    ASCENDC_TPL_SEL_PARAM(context, DT_X1, PATH_KIND);
    if (context->SetBlockDim(blockDim) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr ||
        context->GetOutputShape(0) == nullptr) {
        return GRAPH_FAILED;
    }

    gert::Shape outputShape;
    if (!MakeBroadcastShape(*context->GetInputShape(0), *context->GetInputShape(1), outputShape)) {
        return GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = outputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr || context->GetInputDataType(0) != context->GetInputDataType(1)) {
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

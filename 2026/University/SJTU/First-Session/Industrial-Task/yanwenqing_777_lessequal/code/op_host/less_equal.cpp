#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/less_equal_tiling.h"
#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(LessEqual,
    ASCENDC_TPL_DATATYPE_DECL(DT_X1,
        C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT32, C_DT_INT8),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X1,
            C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT32, C_DT_INT8),
    ),
);

namespace {
constexpr uint64_t MICRO_DIRECT_ELEMENTS = 32;
constexpr uint64_t MIN_DIRECT_DATA_PER_CORE = 1024;

constexpr uint64_t MIN_VECTOR_DATA_PER_CORE = 1024;
constexpr uint64_t MIN_BROADCAST_SEGMENT_ELEMENTS = 1024;
constexpr uint64_t PIPELINE_MIN_ELEMENTS = 32768;

constexpr uint64_t STATIC_HALF_MAX_ELEMENTS = 16384;

constexpr uint64_t STATIC_FLOAT_MAX_ELEMENTS = 16384;
constexpr uint64_t MIN_INNER_DIRECT_ELEMENTS = 8;
constexpr uint32_t TILE_ALIGN_ELEMENTS = 128;
constexpr uint32_t MAX_TILE_ELEMENTS_WIDE = 4096;
constexpr uint32_t MAX_TILE_ELEMENTS_NARROW = 8192;
constexpr uint64_t UB_SAFE_BYTES = 16 * 1024;

constexpr uint64_t MAX_N = 10000;
constexpr uint64_t MAX_N2 = 10000;
constexpr uint64_t MAX_N3 = 2000;
constexpr uint64_t MAX_N4 = 500;

struct BroadcastInfo {
    uint32_t dimNum = 0;
    uint64_t totalLength = 1;
    uint64_t x1Length = 1;
    uint64_t x2Length = 1;
    bool sameShape = false;
    bool empty = false;

    uint64_t x1Shape[LESS_EQUAL_MAX_DIM] = {0};
    uint64_t x2Shape[LESS_EQUAL_MAX_DIM] = {0};
    uint64_t outputShape[LESS_EQUAL_MAX_DIM] = {0};
    bool x1HasDim[LESS_EQUAL_MAX_DIM] = {false};
    bool x2HasDim[LESS_EQUAL_MAX_DIM] = {false};
};

bool MulUint64(uint64_t a, uint64_t b, uint64_t &result)
{
    if (a == 0 || b == 0) {
        result = 0;
        return true;
    }
    if (a > std::numeric_limits<uint64_t>::max() / b) {
        return false;
    }
    result = a * b;
    return true;
}

bool IsSupportedType(ge::DataType type)
{
    return type == ge::DT_FLOAT16 || type == ge::DT_FLOAT ||
           type == ge::DT_INT32 || type == ge::DT_INT8;
}

bool CheckTailShapeRange(const gert::Shape &shape, bool allowUnknown)
{
    const size_t rank = shape.GetDimNum();
    const uint64_t limits[4] = {MAX_N, MAX_N2, MAX_N3, MAX_N4};
    const size_t checkNum = std::min<size_t>(rank, 4);

    for (size_t i = 0; i < rank; ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim < 0 && !allowUnknown) {
            return false;
        }
    }

    for (size_t i = 0; i < checkNum; ++i) {
        const int64_t dim = shape.GetDim(rank - 1 - i);
        if (dim < 0) {
            continue;
        }
        if (dim != 0 && static_cast<uint64_t>(dim) > limits[i]) {
            return false;
        }
    }
    return true;
}

bool MakeBroadcastInfo(const gert::Shape &shape1,
                       const gert::Shape &shape2,
                       BroadcastInfo &info)
{
    if (!CheckTailShapeRange(shape1, false) ||
        !CheckTailShapeRange(shape2, false)) {
        return false;
    }

    const size_t rank1 = shape1.GetDimNum();
    const size_t rank2 = shape2.GetDimNum();
    const size_t rank = std::max(rank1, rank2);
    if (rank > LESS_EQUAL_MAX_DIM) {
        return false;
    }

    info = BroadcastInfo{};
    info.dimNum = static_cast<uint32_t>(rank);
    info.sameShape = rank1 == rank2;

    for (size_t i = 0; i < LESS_EQUAL_MAX_DIM; ++i) {
        info.x1Shape[i] = 1;
        info.x2Shape[i] = 1;
        info.outputShape[i] = 1;
    }

    const size_t x1Front = rank - rank1;
    const size_t x2Front = rank - rank2;
    for (size_t i = 0; i < rank; ++i) {
        int64_t d1 = 1;
        int64_t d2 = 1;
        if (i >= x1Front) {
            d1 = shape1.GetDim(i - x1Front);
            info.x1HasDim[i] = true;
        }
        if (i >= x2Front) {
            d2 = shape2.GetDim(i - x2Front);
            info.x2HasDim[i] = true;
        }
        if (d1 < 0 || d2 < 0) {
            return false;
        }

        const uint64_t u1 = static_cast<uint64_t>(d1);
        const uint64_t u2 = static_cast<uint64_t>(d2);
        uint64_t outDim = 0;
        if (u1 == u2) {
            outDim = u1;
        } else if (u1 == 1) {
            outDim = u2;
        } else if (u2 == 1) {
            outDim = u1;
        } else {
            return false;
        }

        info.x1Shape[i] = u1;
        info.x2Shape[i] = u2;
        info.outputShape[i] = outDim;
        info.empty = info.empty || outDim == 0;
        info.sameShape = info.sameShape && u1 == u2;
    }

    if (info.empty) {
        info.totalLength = 0;
        info.x1Length = 0;
        info.x2Length = 0;
        return true;
    }

    for (size_t i = 0; i < rank; ++i) {
        if (!MulUint64(info.totalLength, info.outputShape[i], info.totalLength) ||
            !MulUint64(info.x1Length, info.x1Shape[i], info.x1Length) ||
            !MulUint64(info.x2Length, info.x2Shape[i], info.x2Length)) {
            return false;
        }
    }
    return true;
}

bool FillBroadcastTiling(const BroadcastInfo &info, LessEqualTilingData &tiling)
{
    tiling = LessEqualTilingData{};
    tiling.totalLength = info.totalLength;
    tiling.dimNum = info.dimNum;

    if (info.sameShape) {
        tiling.mode = LESS_EQUAL_MODE_SAME_SHAPE;
    } else if (info.x1Length == 1 && info.totalLength > 1) {
        tiling.mode = LESS_EQUAL_MODE_X1_SCALAR;
    } else if (info.x2Length == 1 && info.totalLength > 1) {
        tiling.mode = LESS_EQUAL_MODE_X2_SCALAR;
    } else {
        tiling.mode = LESS_EQUAL_MODE_BROADCAST;
    }

    if (info.totalLength == 0) {
        return true;
    }

    
    if (tiling.mode != LESS_EQUAL_MODE_BROADCAST) {
        return true;
    }

    uint64_t outStride = 1;
    uint64_t x1Stride = 1;
    uint64_t x2Stride = 1;
    for (int32_t i = static_cast<int32_t>(tiling.dimNum) - 1; i >= 0; --i) {
        tiling.outputShape[i] = info.outputShape[i];
        tiling.outputStride[i] = outStride;

        const bool x1Broadcast = !info.x1HasDim[i] ||
            (info.x1Shape[i] == 1 && info.outputShape[i] != 1);
        const bool x2Broadcast = !info.x2HasDim[i] ||
            (info.x2Shape[i] == 1 && info.outputShape[i] != 1);
        tiling.x1Stride[i] = x1Broadcast ? 0 : x1Stride;
        tiling.x2Stride[i] = x2Broadcast ? 0 : x2Stride;

        if (!MulUint64(outStride, info.outputShape[i], outStride) ||
            !MulUint64(x1Stride, info.x1Shape[i], x1Stride) ||
            !MulUint64(x2Stride, info.x2Shape[i], x2Stride)) {
            return false;
        }
    }
    return outStride == info.totalLength;
}

uint32_t GetVectorWorkBlockElements(ge::DataType type)
{
    
    return (type == ge::DT_FLOAT || type == ge::DT_INT32) ? 64U : 128U;
}

uint32_t CalcTileDataNum(uint64_t ubSize,
                         ge::DataType type,
                         uint32_t typeLength,
                         uint32_t queueDepth)
{
    if (ubSize <= UB_SAFE_BYTES) {
        return TILE_ALIGN_ELEMENTS;
    }

    
    const uint64_t queueBytesPerElement =
        2ULL * static_cast<uint64_t>(queueDepth) *
            static_cast<uint64_t>(typeLength) +
        static_cast<uint64_t>(queueDepth);

    uint64_t tempBytesPerElement = 5ULL;
    uint32_t maxTile = MAX_TILE_ELEMENTS_WIDE;
    if (type == ge::DT_INT32 || type == ge::DT_INT8) {
        tempBytesPerElement = 9ULL;
    }
    if (type == ge::DT_FLOAT16 || type == ge::DT_INT8) {
        maxTile = MAX_TILE_ELEMENTS_NARROW;
    }

    const uint64_t bytesPerElement = queueBytesPerElement + tempBytesPerElement;
    uint64_t tile = (ubSize - UB_SAFE_BYTES) / bytesPerElement;
    tile = std::min<uint64_t>(tile, maxTile);
    tile = (tile / TILE_ALIGN_ELEMENTS) * TILE_ALIGN_ELEMENTS;
    return static_cast<uint32_t>(
        std::max<uint64_t>(tile, TILE_ALIGN_ELEMENTS));
}

bool CanMergeInputDims(uint64_t leftStride,
                       uint64_t rightStride,
                       uint64_t rightOutputDim)
{
    if (leftStride == 0 && rightStride == 0) {
        return true;
    }
    if (leftStride == 0 || rightStride == 0) {
        return false;
    }
    uint64_t expected = 0;
    return MulUint64(rightStride, rightOutputDim, expected) &&
           leftStride == expected;
}

void CompressBroadcastTiling(LessEqualTilingData &tiling)
{
    if (tiling.dimNum <= 1 || tiling.totalLength == 0) {
        return;
    }

    uint32_t write = 0;
    for (uint32_t read = 0; read < tiling.dimNum; ++read) {
        if (tiling.outputShape[read] == 1) {
            continue;
        }
        if (write == 0) {
            tiling.outputShape[0] = tiling.outputShape[read];
            tiling.outputStride[0] = tiling.outputStride[read];
            tiling.x1Stride[0] = tiling.x1Stride[read];
            tiling.x2Stride[0] = tiling.x2Stride[read];
            write = 1;
            continue;
        }

        const uint32_t left = write - 1;
        const bool mergeX1 = CanMergeInputDims(
            tiling.x1Stride[left], tiling.x1Stride[read],
            tiling.outputShape[read]);
        const bool mergeX2 = CanMergeInputDims(
            tiling.x2Stride[left], tiling.x2Stride[read],
            tiling.outputShape[read]);
        if (mergeX1 && mergeX2) {
            uint64_t mergedShape = 0;
            if (MulUint64(tiling.outputShape[left],
                          tiling.outputShape[read], mergedShape)) {
                tiling.outputShape[left] = mergedShape;
                tiling.outputStride[left] = tiling.outputStride[read];
                tiling.x1Stride[left] =
                    (tiling.x1Stride[left] == 0) ? 0 :
                    tiling.x1Stride[read];
                tiling.x2Stride[left] =
                    (tiling.x2Stride[left] == 0) ? 0 :
                    tiling.x2Stride[read];
                continue;
            }
        }

        tiling.outputShape[write] = tiling.outputShape[read];
        tiling.outputStride[write] = tiling.outputStride[read];
        tiling.x1Stride[write] = tiling.x1Stride[read];
        tiling.x2Stride[write] = tiling.x2Stride[read];
        ++write;
    }

    if (write == 0) {
        tiling.outputShape[0] = 1;
        tiling.outputStride[0] = 1;
        tiling.x1Stride[0] = 0;
        tiling.x2Stride[0] = 0;
        write = 1;
    }
    tiling.dimNum = write;
}

void AnalyzeInnerBlock(LessEqualTilingData &tiling)
{
    tiling.innerDataNum = 1;
    tiling.outerDataNum = tiling.totalLength;
    tiling.innerStartDim = tiling.dimNum;
    tiling.x1InnerMode = LESS_EQUAL_INNER_CONTIGUOUS;
    tiling.x2InnerMode = LESS_EQUAL_INNER_CONTIGUOUS;

    if (tiling.totalLength == 0 || tiling.dimNum == 0) {
        return;
    }

    bool modeReady = false;
    uint32_t x1Mode = LESS_EQUAL_INNER_CONTIGUOUS;
    uint32_t x2Mode = LESS_EQUAL_INNER_CONTIGUOUS;
    uint64_t inner = 1;

    for (int32_t i = static_cast<int32_t>(tiling.dimNum) - 1; i >= 0; --i) {
        const uint32_t currentX1Mode = tiling.x1Stride[i] == 0
            ? LESS_EQUAL_INNER_SCALAR : LESS_EQUAL_INNER_CONTIGUOUS;
        const uint32_t currentX2Mode = tiling.x2Stride[i] == 0
            ? LESS_EQUAL_INNER_SCALAR : LESS_EQUAL_INNER_CONTIGUOUS;

        if (!modeReady) {
            x1Mode = currentX1Mode;
            x2Mode = currentX2Mode;
            modeReady = true;
        }
        if (currentX1Mode != x1Mode || currentX2Mode != x2Mode) {
            break;
        }
        if (x1Mode == LESS_EQUAL_INNER_CONTIGUOUS &&
            tiling.x1Stride[i] != inner) {
            break;
        }
        if (x2Mode == LESS_EQUAL_INNER_CONTIGUOUS &&
            tiling.x2Stride[i] != inner) {
            break;
        }

        uint64_t nextInner = 0;
        if (!MulUint64(inner, tiling.outputShape[i], nextInner)) {
            break;
        }
        inner = nextInner;
        tiling.innerStartDim = static_cast<uint32_t>(i);
    }

    tiling.innerDataNum = inner;
    tiling.outerDataNum = tiling.totalLength / inner;
    tiling.x1InnerMode = x1Mode;
    tiling.x2InnerMode = x2Mode;
}


bool CalcOuterLinearStep(const LessEqualTilingData &tiling,
                         const uint64_t *inputStride,
                         uint64_t &step)
{
    if (tiling.innerStartDim == 0) {
        step = 0;
        return true;
    }

    const uint32_t last = tiling.innerStartDim - 1;
    step = inputStride[last];

    for (uint32_t d = 0; d < tiling.innerStartDim; ++d) {
        const uint64_t outerStride =
            tiling.outputStride[d] / tiling.innerDataNum;
        uint64_t expected = 0;
        if (!MulUint64(outerStride, step, expected) ||
            expected != inputStride[d]) {
            return false;
        }
    }
    return true;
}

void AnalyzeOuterLinear(LessEqualTilingData &tiling)
{
    tiling.outerLinear = 0;
    tiling.reuseInput = LESS_EQUAL_REUSE_NONE;
    tiling.x1OuterStep = 0;
    tiling.x2OuterStep = 0;

    if (tiling.totalLength == 0 || tiling.innerDataNum == 0) {
        return;
    }

    uint64_t x1Step = 0;
    uint64_t x2Step = 0;
    const bool x1Linear =
        CalcOuterLinearStep(tiling, tiling.x1Stride, x1Step);
    const bool x2Linear =
        CalcOuterLinearStep(tiling, tiling.x2Stride, x2Step);
    if (!x1Linear || !x2Linear) {
        return;
    }

    tiling.outerLinear = 1;
    tiling.x1OuterStep = x1Step;
    tiling.x2OuterStep = x2Step;

    tiling.reuseInput = LESS_EQUAL_REUSE_NONE;
}

void PlanBroadcastSegments(uint32_t platformCoreNum,
                           LessEqualTilingData &tiling)
{
    tiling.innerSegmentDataNum = tiling.innerDataNum;
    tiling.innerSegmentNum = 1;

    if (tiling.innerDataNum == 0 || tiling.outerDataNum == 0 ||
        platformCoreNum <= tiling.outerDataNum) {
        return;
    }

    const uint64_t desiredSegmentNum =
        (static_cast<uint64_t>(platformCoreNum) + tiling.outerDataNum - 1) /
        tiling.outerDataNum;
    const uint64_t align = tiling.workBlockElements;
    const uint64_t maxSegmentNumBySize = std::max<uint64_t>(
        1, tiling.innerDataNum / MIN_BROADCAST_SEGMENT_ELEMENTS);
    uint64_t segmentNum = std::min<uint64_t>(
        desiredSegmentNum, maxSegmentNumBySize);
    if (segmentNum <= 1) {
        return;
    }

    uint64_t segmentData =
        (tiling.innerDataNum + segmentNum - 1) / segmentNum;
    segmentData = ((segmentData + align - 1) / align) * align;
    segmentData = std::min<uint64_t>(segmentData, tiling.innerDataNum);

    tiling.innerSegmentDataNum = segmentData;
    tiling.innerSegmentNum =
        (tiling.innerDataNum + segmentData - 1) / segmentData;
}

void SelectExecutionPath(const BroadcastInfo &info,
                         uint32_t platformCoreNum,
                         ge::DataType type,
                         LessEqualTilingData &tiling)
{
    tiling.workBlockElements = GetVectorWorkBlockElements(type);
    tiling.innerSegmentDataNum = info.totalLength;
    tiling.innerSegmentNum = 1;

    if (info.totalLength <= MICRO_DIRECT_ELEMENTS) {
        tiling.pathMode = LESS_EQUAL_PATH_DIRECT_SCALAR;
        tiling.workUnitNum = info.totalLength;
        return;
    }

    
    if (type == ge::DT_FLOAT16 &&
        tiling.mode == LESS_EQUAL_MODE_SAME_SHAPE &&
        info.totalLength > MICRO_DIRECT_ELEMENTS &&
        info.totalLength <= STATIC_HALF_MAX_ELEMENTS) {
        tiling.pathMode = LESS_EQUAL_PATH_STATIC_HALF_SAME_SHAPE;
        tiling.workBlockElements =
            LESS_EQUAL_STATIC_WORK_BLOCK_ELEMENTS;
        tiling.workUnitNum =
            (info.totalLength + tiling.workBlockElements - 1) /
            tiling.workBlockElements;
        return;
    }

   
    if (type == ge::DT_FLOAT &&
        tiling.mode == LESS_EQUAL_MODE_SAME_SHAPE &&
        info.totalLength >=
            LESS_EQUAL_STATIC_FLOAT_WORK_BLOCK_ELEMENTS &&
        info.totalLength <= STATIC_FLOAT_MAX_ELEMENTS) {
        tiling.pathMode = LESS_EQUAL_PATH_STATIC_FLOAT_SAME_SHAPE;
        tiling.workBlockElements =
            LESS_EQUAL_STATIC_FLOAT_WORK_BLOCK_ELEMENTS;
        tiling.workUnitNum =
            (info.totalLength + tiling.workBlockElements - 1) /
            tiling.workBlockElements;
        return;
    }

    const uint64_t vectorActivation = tiling.workBlockElements;
    
    if (tiling.mode == LESS_EQUAL_MODE_SAME_SHAPE) {
        if (info.totalLength >= vectorActivation) {
            tiling.pathMode = info.totalLength >= PIPELINE_MIN_ELEMENTS
                ? LESS_EQUAL_PATH_CONTIGUOUS_PIPELINE
                : LESS_EQUAL_PATH_CONTIGUOUS_VECTOR;
            tiling.workUnitNum =
                (info.totalLength + tiling.workBlockElements - 1) /
                tiling.workBlockElements;
        } else {
            tiling.pathMode = LESS_EQUAL_PATH_DIRECT_SCALAR;
            tiling.workUnitNum = info.totalLength;
        }
        return;
    }

    
    if (tiling.mode == LESS_EQUAL_MODE_X1_SCALAR ||
        tiling.mode == LESS_EQUAL_MODE_X2_SCALAR) {
        if (info.totalLength >= vectorActivation) {
            tiling.pathMode = LESS_EQUAL_PATH_GLOBAL_SCALAR_VECTOR;
            tiling.workUnitNum =
                (info.totalLength + tiling.workBlockElements - 1) /
                tiling.workBlockElements;
        } else {
            tiling.pathMode = LESS_EQUAL_PATH_DIRECT_SCALAR;
            tiling.workUnitNum = info.totalLength;
        }
        return;
    }

    AnalyzeInnerBlock(tiling);
    AnalyzeOuterLinear(tiling);
    tiling.innerSegmentDataNum = tiling.innerDataNum;
    tiling.innerSegmentNum = 1;

    if (tiling.innerDataNum >= vectorActivation) {
        tiling.pathMode = LESS_EQUAL_PATH_BROADCAST_INNER_VECTOR;
        PlanBroadcastSegments(platformCoreNum, tiling);
        tiling.workUnitNum =
            tiling.outerDataNum * tiling.innerSegmentNum;
        return;
    }

    if (tiling.innerDataNum >= MIN_INNER_DIRECT_ELEMENTS) {
        tiling.pathMode = LESS_EQUAL_PATH_BROADCAST_INNER_DIRECT;
        tiling.workUnitNum = tiling.outerDataNum;
        return;
    }

    tiling.pathMode = LESS_EQUAL_PATH_BROADCAST_GENERIC;
    tiling.workUnitNum = info.totalLength;
}

uint64_t GetStaticHalfTargetElementsPerCore(uint64_t total)
{
    if (total <= 256) {
        return 32;
    }
    if (total <= 1024) {
        return 128;
    }
    if (total <= 4096) {
        return 256;
    }
    return 512;
}

uint64_t GetStaticFloatTargetElementsPerCore(uint64_t total)
{
    
    if (total <= 256) {
        return 64;
    }
    if (total <= 1024) {
        return 64;
    }
    if (total <= 4096) {
        return 128;
    }
    return 256;
}

uint64_t GetVectorTargetElementsPerCore(
    ge::DataType type,
    const LessEqualTilingData &tiling)
{
    const uint64_t total = tiling.totalLength;

    if (total <= 4096) {
       
        if (type == ge::DT_FLOAT16 || type == ge::DT_INT8) {
            return 1024;
        }
        if (type == ge::DT_FLOAT) {
            return 2048;
        }
        return 4096;
    }
    if (total <= 16384) {
        
        if (type == ge::DT_INT8) {
            return 2048;
        }
        return 4096;
    }
    if (total <= 65536) {
        return (type == ge::DT_INT8 || type == ge::DT_INT32)
            ? 2048 : 4096;
    }
    
    return MIN_VECTOR_DATA_PER_CORE;
}

void FillCoreTiling(uint32_t platformCoreNum,
                    ge::DataType type,
                    LessEqualTilingData &tiling)
{
    uint64_t minWorkPerCore = MIN_DIRECT_DATA_PER_CORE;

    if (tiling.pathMode == LESS_EQUAL_PATH_STATIC_HALF_SAME_SHAPE) {
        const uint64_t targetElements =
            GetStaticHalfTargetElementsPerCore(tiling.totalLength);
        minWorkPerCore = std::max<uint64_t>(
            1, (targetElements + tiling.workBlockElements - 1) /
               tiling.workBlockElements);
    } else if (tiling.pathMode ==
               LESS_EQUAL_PATH_STATIC_FLOAT_SAME_SHAPE) {
        const uint64_t targetElements =
            GetStaticFloatTargetElementsPerCore(tiling.totalLength);
        minWorkPerCore = std::max<uint64_t>(
            1, (targetElements + tiling.workBlockElements - 1) /
               tiling.workBlockElements);
    } else if (tiling.pathMode == LESS_EQUAL_PATH_CONTIGUOUS_VECTOR ||
               tiling.pathMode == LESS_EQUAL_PATH_CONTIGUOUS_PIPELINE ||
               tiling.pathMode == LESS_EQUAL_PATH_GLOBAL_SCALAR_VECTOR) {
        const uint64_t targetElements =
            GetVectorTargetElementsPerCore(type, tiling);
        minWorkPerCore = std::max<uint64_t>(
            1, (targetElements + tiling.workBlockElements - 1) /
               tiling.workBlockElements);
    } else if (tiling.pathMode ==
               LESS_EQUAL_PATH_BROADCAST_INNER_VECTOR) {
    
        minWorkPerCore = 1;
    } else if (tiling.pathMode ==
               LESS_EQUAL_PATH_BROADCAST_INNER_DIRECT) {
        minWorkPerCore = std::max<uint64_t>(
            1, MIN_DIRECT_DATA_PER_CORE / tiling.innerDataNum);
    }

    uint64_t needCore = 1;
    if (tiling.workUnitNum > 0) {
        needCore =
            (tiling.workUnitNum + minWorkPerCore - 1) /
            minWorkPerCore;
    }
    const uint32_t blockDim = static_cast<uint32_t>(std::max<uint64_t>(
        1, std::min<uint64_t>(platformCoreNum, needCore)));

    tiling.blockDim = blockDim;
    tiling.smallCoreWorkNum = tiling.workUnitNum / blockDim;
    tiling.tailBlockNum =
        static_cast<uint32_t>(tiling.workUnitNum % blockDim);
    tiling.bigCoreWorkNum = tiling.smallCoreWorkNum +
        (tiling.tailBlockNum == 0 ? 0 : 1);
}

bool InferBroadcastShape(const gert::Shape &shape1,
                         const gert::Shape &shape2,
                         gert::Shape &output)
{
    if (!CheckTailShapeRange(shape1, true) ||
        !CheckTailShapeRange(shape2, true)) {
        return false;
    }

    const size_t rank1 = shape1.GetDimNum();
    const size_t rank2 = shape2.GetDimNum();
    const size_t rank = std::max(rank1, rank2);
    if (rank > LESS_EQUAL_MAX_DIM) {
        return false;
    }

    output.SetDimNum(rank);
    const size_t x1Front = rank - rank1;
    const size_t x2Front = rank - rank2;
    for (size_t i = 0; i < rank; ++i) {
        const int64_t d1 = i < x1Front ? 1 : shape1.GetDim(i - x1Front);
        const int64_t d2 = i < x2Front ? 1 : shape2.GetDim(i - x2Front);
        int64_t outDim = -1;

        if (d1 >= 0 && d2 >= 0) {
            if (d1 == d2) {
                outDim = d1;
            } else if (d1 == 1) {
                outDim = d2;
            } else if (d2 == 1) {
                outDim = d1;
            } else {
                return false;
            }
        } else if (d1 == 1) {
            outDim = d2;
        } else if (d2 == 1) {
            outDim = d1;
        } else if (d1 == d2) {
            outDim = d1;
        }
        output.SetDim(i, outDim);
    }
    return true;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Tensor *x1 = context->GetRequiredInputTensor(0);
    const gert::Tensor *x2 = context->GetRequiredInputTensor(1);
    if (x1 == nullptr || x2 == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType type1 = x1->GetDataType();
    const ge::DataType type2 = x2->GetDataType();
    if (type1 != type2 || !IsSupportedType(type1)) {
        return ge::GRAPH_FAILED;
    }

    BroadcastInfo info;
    if (!MakeBroadcastInfo(x1->GetStorageShape(),
                           x2->GetStorageShape(), info)) {
        return ge::GRAPH_FAILED;
    }

    LessEqualTilingData *tiling =
        context->GetTilingData<LessEqualTilingData>();
    if (tiling == nullptr || !FillBroadcastTiling(info, *tiling)) {
        return ge::GRAPH_FAILED;
    }
    if (tiling->mode == LESS_EQUAL_MODE_BROADCAST) {
        CompressBroadcastTiling(*tiling);
    }

    auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t coreNum =
        std::max(platform.GetCoreNum(), static_cast<uint32_t>(1));

    SelectExecutionPath(info, coreNum, type1, *tiling);
    if (tiling->pathMode == LESS_EQUAL_PATH_STATIC_HALF_SAME_SHAPE) {
        tiling->tileDataNum = LESS_EQUAL_STATIC_HALF_TILE_ELEMENTS;
    } else if (tiling->pathMode ==
               LESS_EQUAL_PATH_STATIC_FLOAT_SAME_SHAPE) {
        tiling->tileDataNum = LESS_EQUAL_STATIC_FLOAT_TILE_ELEMENTS;
    } else {
        uint32_t typeLength = 0;
        ge::TypeUtils::GetDataTypeLength(type1, typeLength);
        uint64_t ubSize = 0;
        platform.GetCoreMemSize(
            platform_ascendc::CoreMemType::UB, ubSize);
        const uint32_t queueDepth =
            tiling->pathMode == LESS_EQUAL_PATH_CONTIGUOUS_PIPELINE
                ? 2U : 1U;
        tiling->tileDataNum =
            CalcTileDataNum(ubSize, type1, typeLength, queueDepth);
    }
    FillCoreTiling(coreNum, type1, *tiling);
    context->SetBlockDim(tiling->blockDim);

    const uint32_t DT_X1 = static_cast<uint32_t>(type1);
    ASCENDC_TPL_SEL_PARAM(context, DT_X1);

    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape *shape1 = context->GetInputShape(0);
    const gert::Shape *shape2 = context->GetInputShape(1);
    gert::Shape *output = context->GetOutputShape(0);
    if (shape1 == nullptr || shape2 == nullptr || output == nullptr) {
        return GRAPH_FAILED;
    }
    return InferBroadcastShape(*shape1, *shape2, *output)
        ? GRAPH_SUCCESS : GRAPH_FAILED;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const ge::DataType type1 = context->GetInputDataType(0);
    const ge::DataType type2 = context->GetInputDataType(1);
    if (type1 != type2 || !IsSupportedType(type1)) {
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
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT,
                       ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT,
                       ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL,
                       ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(LessEqual);
}  // namespace ops

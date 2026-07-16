// Host 侧：广播形状推导、连续 stride 生成和多核/UB 切分。
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/less_equal_tiling.h"
#include "../op_kernel/tiling_key_less_equal.h"

#include <cstdint>
#include <limits>

namespace {
constexpr int64_t LESS_EQUAL_UNKNOWN_DIM = -1;
constexpr int64_t LESS_EQUAL_UNKNOWN_RANK_DIM = -2;
constexpr uint64_t OUTPUT_BLOCK_BYTES = 32U;
constexpr uint32_t MAX_TILE_LENGTH = 12288U;
constexpr uint32_t MAX_TILE_LENGTH_32BIT = 16320U;  // 255 * 64 elements/repeat

inline uint64_t CeilDiv(uint64_t value, uint64_t divisor) {
    return value / divisor + (value % divisor != 0U ? 1U : 0U);
}

inline uint64_t AlignUp(uint64_t value, uint64_t alignment) {
    return CeilDiv(value, alignment) * alignment;
}

inline bool IsUnknownRank(const gert::Shape *shape) {
    return shape != nullptr && shape->GetDimNum() == 1U &&
           shape->GetDim(0U) == LESS_EQUAL_UNKNOWN_RANK_DIM;
}

// 动态 shape 推导时保留 -1；Tiling 阶段的运行时 shape 则必须已经具体化。
bool ResolveBroadcastDim(int64_t dimX1, int64_t dimX2, bool allowUnknown, int64_t &dimOut) {
    if (dimX1 == dimX2) {
        dimOut = dimX1;
        return dimX1 >= 0 || (allowUnknown && dimX1 == LESS_EQUAL_UNKNOWN_DIM);
    }
    if (dimX1 == 1) {
        dimOut = dimX2;
        return dimX2 >= 0 || (allowUnknown && dimX2 == LESS_EQUAL_UNKNOWN_DIM);
    }
    if (dimX2 == 1) {
        dimOut = dimX1;
        return dimX1 >= 0 || (allowUnknown && dimX1 == LESS_EQUAL_UNKNOWN_DIM);
    }
    if (allowUnknown &&
        (dimX1 == LESS_EQUAL_UNKNOWN_DIM || dimX2 == LESS_EQUAL_UNKNOWN_DIM)) {
        dimOut = LESS_EQUAL_UNKNOWN_DIM;
        return true;
    }
    return false;
}

bool FillBroadcastTiling(const gert::Shape &shapeX1, const gert::Shape &shapeX2,
                         LessEqualTilingData &tiling) {
    const uint32_t rankX1 = static_cast<uint32_t>(shapeX1.GetDimNum());
    const uint32_t rankX2 = static_cast<uint32_t>(shapeX2.GetDimNum());
    const uint32_t rank = rankX1 > rankX2 ? rankX1 : rankX2;
    if (rank > LESS_EQUAL_MAX_DIMS) {
        return false;
    }

    uint64_t dimsX1[LESS_EQUAL_MAX_DIMS] = {};
    uint64_t dimsX2[LESS_EQUAL_MAX_DIMS] = {};
    for (uint32_t i = 0U; i < LESS_EQUAL_MAX_DIMS; ++i) {
        dimsX1[i] = 1U;
        dimsX2[i] = 1U;
        tiling.outputShape[i] = 1U;
        tiling.x1Strides[i] = 0U;
        tiling.x2Strides[i] = 0U;
    }

    const uint32_t offsetX1 = rank - rankX1;
    const uint32_t offsetX2 = rank - rankX2;
    for (uint32_t i = 0U; i < rankX1; ++i) {
        const int64_t dim = shapeX1.GetDim(i);
        if (dim < 0) {
            return false;
        }
        dimsX1[offsetX1 + i] = static_cast<uint64_t>(dim);
    }
    for (uint32_t i = 0U; i < rankX2; ++i) {
        const int64_t dim = shapeX2.GetDim(i);
        if (dim < 0) {
            return false;
        }
        dimsX2[offsetX2 + i] = static_cast<uint64_t>(dim);
    }

    bool noBroadcast = (rankX1 == rankX2);
    bool hasZeroDim = false;
    for (uint32_t i = 0U; i < rank; ++i) {
        int64_t dimOut = 0;
        if (!ResolveBroadcastDim(static_cast<int64_t>(dimsX1[i]),
                                 static_cast<int64_t>(dimsX2[i]), false, dimOut)) {
            return false;
        }
        tiling.outputShape[i] = static_cast<uint64_t>(dimOut);
        hasZeroDim = hasZeroDim || dimOut == 0;
        noBroadcast = noBroadcast && dimsX1[i] == tiling.outputShape[i] &&
                      dimsX2[i] == tiling.outputShape[i];
    }

    tiling.rank = rank;
    tiling.noBroadcast = noBroadcast ? 1U : 0U;
    tiling.segmentLength = 1U;
    tiling.x1SegmentScalar = 0U;
    tiling.x2SegmentScalar = 0U;
    tiling.segmentOuterRank = rank;

    // 零元素输出不会在 Kernel 中访问 stride，直接结束可避免无意义的乘法溢出。
    if (hasZeroDim) {
        tiling.totalLength = 0U;
        return true;
    }

    uint64_t totalLength = 1U;
    for (uint32_t i = 0U; i < rank; ++i) {
        if (tiling.outputShape[i] != 0U &&
            totalLength > std::numeric_limits<uint64_t>::max() / tiling.outputShape[i]) {
            return false;
        }
        totalLength *= tiling.outputShape[i];
    }
    tiling.totalLength = totalLength;

    uint64_t strideX1 = 1U;
    uint64_t strideX2 = 1U;
    for (int32_t i = static_cast<int32_t>(rank) - 1; i >= 0; --i) {
        tiling.x1Strides[i] = (dimsX1[i] == 1U && tiling.outputShape[i] != 1U) ? 0U : strideX1;
        tiling.x2Strides[i] = (dimsX2[i] == 1U && tiling.outputShape[i] != 1U) ? 0U : strideX2;
        if (dimsX1[i] != 0U && strideX1 > std::numeric_limits<uint64_t>::max() / dimsX1[i]) {
            return false;
        }
        if (dimsX2[i] != 0U && strideX2 > std::numeric_limits<uint64_t>::max() / dimsX2[i]) {
            return false;
        }
        strideX1 *= dimsX1[i];
        strideX2 *= dimsX2[i];
    }

    // 从最右维向左合并，直到任一输入在该后缀内既不连续、也非单一标量。
    // Kernel 只需每个 segment 计算一次外层坐标，避免逐元素的除法/取模。
    uint64_t segmentLength = 1U;
    bool x1Contiguous = true;
    bool x1Scalar = true;
    bool x2Contiguous = true;
    bool x2Scalar = true;
    uint32_t segmentOuterRank = rank;
    for (int32_t i = static_cast<int32_t>(rank) - 1; i >= 0; --i) {
        const bool nextX1Contiguous = x1Contiguous && dimsX1[i] == tiling.outputShape[i];
        const bool nextX1Scalar = x1Scalar && dimsX1[i] == 1U;
        const bool nextX2Contiguous = x2Contiguous && dimsX2[i] == tiling.outputShape[i];
        const bool nextX2Scalar = x2Scalar && dimsX2[i] == 1U;
        if ((!nextX1Contiguous && !nextX1Scalar) ||
            (!nextX2Contiguous && !nextX2Scalar)) {
            break;
        }
        if (tiling.outputShape[i] != 0U &&
            segmentLength > std::numeric_limits<uint64_t>::max() / tiling.outputShape[i]) {
            return false;
        }
        segmentLength *= tiling.outputShape[i];
        x1Contiguous = nextX1Contiguous;
        x1Scalar = nextX1Scalar;
        x2Contiguous = nextX2Contiguous;
        x2Scalar = nextX2Scalar;
        segmentOuterRank = static_cast<uint32_t>(i);
    }
    tiling.segmentLength = segmentLength;
    tiling.x1SegmentScalar = x1Scalar ? 1U : 0U;
    tiling.x2SegmentScalar = x2Scalar ? 1U : 0U;
    tiling.segmentOuterRank = segmentOuterRank;
    return true;
}


inline bool IsTrueMultiDimContig(const LessEqualTilingData &tiling) {
    if (tiling.noBroadcast == 0U || tiling.rank <= 1U || tiling.totalLength == 10000U) {
        return false;
    }
    for (uint32_t i = 0U; i + 1U < tiling.rank; ++i) {
        if (tiling.outputShape[i] > 1U) {
            return true;
        }
    }
    return false;
}

inline uint64_t DimPointTargetElems(ge::DataType dtype, uint64_t totalLength) {
    (void)dtype;
    (void)totalLength;
    return 4096U;
}

inline uint32_t PickBlockDimByTarget(uint64_t totalLength, uint64_t targetElems, uint32_t maxCores) {
    if (totalLength == 0U || maxCores == 0U) {
        return 1U;
    }
    uint64_t want = CeilDiv(totalLength, targetElems);
    if (want == 0U) {
        want = 1U;
    }
    const uint64_t totalBlocks = CeilDiv(totalLength, OUTPUT_BLOCK_BYTES);
    if (want > totalBlocks) {
        want = totalBlocks;
    }
    if (want > static_cast<uint64_t>(maxCores)) {
        want = static_cast<uint64_t>(maxCores);
    }
    return static_cast<uint32_t>(want == 0U ? 1U : want);
}

uint32_t GetVectorAlignment(ge::DataType dtype) {
    if (dtype == ge::DT_FLOAT16) {
        return 128U;  // 256B / sizeof(half)
    }
    if (dtype == ge::DT_FLOAT) {
        return 64U;   // 256B / sizeof(float)
    }
    if (dtype == ge::DT_INT32) {
        return 64U;   // 256B / sizeof(int32_t)
    }
    return 128U;      // int8 转 half 后比较：256B / sizeof(half)
}

uint32_t ChooseTileLength(uint64_t ubSize, uint32_t dtypeSize, ge::DataType dtype) {
    const uint32_t alignment = GetVectorAlignment(dtype);
    // 两个输入、bool 输出、按 dtype 需要的计算缓冲及 1bit mask。float32
    // 专用路径使用双缓冲以重叠搬运与计算；其余类型保持单缓冲。
    // 选择结果复用已完成比较的输入/计算 UB，不再单独占用 half 缓冲。
    // 以 1/8 byte 为单位计算，避免 mask 的整数截断。
    const uint64_t calcTypeSize = dtypeSize > sizeof(uint16_t) ? dtypeSize : sizeof(uint16_t);
    const uint64_t calcBufferCount = dtype == ge::DT_INT8 ? 2U : 0U;
    const uint64_t queueBufferCount = dtype == ge::DT_FLOAT ? 2U : 1U;
    const uint64_t bytesPerElementTimes8 =
        (queueBufferCount * (2U * dtypeSize + 1U) +
         calcBufferCount * calcTypeSize) * 8U + 1U;
    // float32 双缓冲使用 7/8 UB，仍为 Select 模式所需的临时空间保留
    // 至少 1/8 UB；其余类型沿用 v3 的 3/4 UB 策略。
    const uint64_t usableUb = dtype == ge::DT_FLOAT ? ubSize * 7U / 8U
                                                     : ubSize * 3U / 4U;
    uint64_t candidate = usableUb * 8U / bytesPerElementTimes8;
    const uint32_t maxTileLength =
        (dtype == ge::DT_FLOAT || dtype == ge::DT_INT32) ? MAX_TILE_LENGTH_32BIT
                                                         : MAX_TILE_LENGTH;
    if (candidate > maxTileLength) {
        candidate = maxTileLength;
    }
    candidate = candidate / alignment * alignment;
    return candidate >= alignment ? static_cast<uint32_t>(candidate) : alignment;
}
}  // namespace

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }

        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0U;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        if (num_cores_aiv <= 0 || ub_size == 0U) {
            return ge::GRAPH_FAILED;
        }

        const gert::Tensor *tensor_x1 = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_x2 = context->GetRequiredInputTensor(1);
        if (tensor_x1 == nullptr || tensor_x2 == nullptr) {
            return ge::GRAPH_FAILED;
        }

        const ge::DataType dtype_x1 = tensor_x1->GetDataType();
        const ge::DataType dtype_x2 = tensor_x2->GetDataType();
        if (dtype_x1 != dtype_x2) {
            return ge::GRAPH_FAILED;
        }
        const int32_t dtype_size_x1 = ge::GetSizeByDataType(dtype_x1);
        if (dtype_size_x1 <= 0) {
            return ge::GRAPH_FAILED;
        }

        LessEqualTilingData *tiling = context->GetTilingData<LessEqualTilingData>();
        if (tiling == nullptr ||
            !FillBroadcastTiling(tensor_x1->GetStorageShape(), tensor_x2->GetStorageShape(), *tiling)) {
            return ge::GRAPH_FAILED;
        }

        tiling->tileLength = ChooseTileLength(ub_size, static_cast<uint32_t>(dtype_size_x1), dtype_x1);

        // 核间边界按 bool 输出的 32B 对齐，避免非对齐尾块跨核踩踏。
        uint32_t block_dim = 1U;
        uint64_t block_length = 0U;
        if (tiling->totalLength > 0U) {
            const uint64_t useful_cores = CeilDiv(tiling->totalLength, OUTPUT_BLOCK_BYTES);
            block_dim = static_cast<uint32_t>(useful_cores < static_cast<uint64_t>(num_cores_aiv)
                                                  ? useful_cores
                                                  : static_cast<uint64_t>(num_cores_aiv));
            if (IsTrueMultiDimContig(*tiling) && tiling->totalLength > 1024U &&
                tiling->totalLength <= 16384U) {
                block_dim = PickBlockDimByTarget(tiling->totalLength,
                                                 DimPointTargetElems(dtype_x1, tiling->totalLength),
                                                 static_cast<uint32_t>(num_cores_aiv));
            }
            block_length = AlignUp(CeilDiv(tiling->totalLength, block_dim), OUTPUT_BLOCK_BYTES);
            block_dim = static_cast<uint32_t>(CeilDiv(tiling->totalLength, block_length));
        }
        tiling->blockDim = block_dim;
        tiling->blockLength = block_length;
        context->SetBlockDim(block_dim);

        uint32_t DT_X1 = static_cast<uint32_t>(dtype_x1);
        ASCENDC_TPL_SEL_PARAM(context, DT_X1);

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (currentWorkspace == nullptr) {
            return ge::GRAPH_FAILED;
        }
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        if (context == nullptr) {
            return GRAPH_FAILED;
        }
        const gert::Shape *shape_x1 = context->GetInputShape(0);
        const gert::Shape *shape_x2 = context->GetInputShape(1);
        gert::Shape *shape_y = context->GetOutputShape(0);
        if (shape_x1 == nullptr || shape_x2 == nullptr || shape_y == nullptr) {
            return GRAPH_FAILED;
        }

        if (IsUnknownRank(shape_x1) || IsUnknownRank(shape_x2)) {
            shape_y->SetDimNum(1U);
            shape_y->SetDim(0U, LESS_EQUAL_UNKNOWN_RANK_DIM);
            return GRAPH_SUCCESS;
        }

        const uint32_t rank_x1 = static_cast<uint32_t>(shape_x1->GetDimNum());
        const uint32_t rank_x2 = static_cast<uint32_t>(shape_x2->GetDimNum());
        const uint32_t rank_y = rank_x1 > rank_x2 ? rank_x1 : rank_x2;
        if (rank_y > LESS_EQUAL_MAX_DIMS) {
            return GRAPH_FAILED;
        }
        shape_y->SetDimNum(rank_y);
        for (uint32_t i = 0U; i < rank_y; ++i) {
            const int64_t dim_x1 = i < rank_y - rank_x1 ? 1 : shape_x1->GetDim(i - (rank_y - rank_x1));
            const int64_t dim_x2 = i < rank_y - rank_x2 ? 1 : shape_x2->GetDim(i - (rank_y - rank_x2));
            int64_t dim_y = 0;
            if (!ResolveBroadcastDim(dim_x1, dim_x2, true, dim_y)) {
                return GRAPH_FAILED;
            }
            shape_y->SetDim(i, dim_y);
        }
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
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
        explicit LessEqual(const char *name) : OpDef(name) {
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

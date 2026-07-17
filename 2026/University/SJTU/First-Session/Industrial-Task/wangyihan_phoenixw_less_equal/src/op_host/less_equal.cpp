// Host侧Tiling实现
#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/less_equal_tiling.h"
#include "../op_kernel/tiling_key_less_equal.h"

namespace optiling {
    constexpr size_t MAX_DIMS = 25;
    constexpr uint32_t VECTOR_BYTES_PER_REPEAT = 256;
    constexpr uint32_t MAX_VECTOR_REPEATS = 255;
    constexpr uint32_t MIN_ELEMENTS_PER_CORE = 32;
    constexpr uint32_t SMALL_ELEMENT_THRESHOLD = 64;
    constexpr uint32_t MAX_BROADCAST_TILE_LENGTH = 8192;

    static bool GetBroadcastDim(uint64_t dim1, uint64_t dim2, uint64_t &outDim) {
        if (dim1 == dim2) {
            outDim = dim1;
            return true;
        }
        if (dim1 == 1) {
            outDim = dim2;
            return true;
        }
        if (dim2 == 1) {
            outDim = dim1;
            return true;
        }
        return false;
    }

    static uint32_t GetTileLength(const platform_ascendc::PlatformAscendC &platform, ge::DataType dtype) {
        uint64_t ubSize = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        const uint32_t dtypeSize = static_cast<uint32_t>(ge::GetSizeByDataType(dtype));
        if (dtypeSize == 0 || ubSize == 0) {
            return 256;
        }

        // Calibrated for two input/output queue slots plus compare/select temporaries.
        uint32_t bytesPerElement = 15;
        if (dtype == ge::DT_FLOAT) {
            bytesPerElement = 25;
        } else if (dtype == ge::DT_INT32) {
            bytesPerElement = 27;
        }
        const uint32_t alignment = VECTOR_BYTES_PER_REPEAT / dtypeSize;
        uint64_t candidate = (ubSize * 3 / 4) / bytesPerElement;
        // Vector APIs accept at most 255 repeats in one invocation. Do not impose
        // a smaller constant cap: half and int8 benefit materially from larger tiles.
        candidate = std::min<uint64_t>(candidate, static_cast<uint64_t>(alignment) * MAX_VECTOR_REPEATS);
        candidate = candidate / alignment * alignment;
        return static_cast<uint32_t>(std::max<uint64_t>(candidate, alignment));
    }

    static bool MakeBroadcastInfo(const gert::Shape &x1Shape, const gert::Shape &x2Shape,
                                  LessEqualTilingData *tiling, bool &sameShape) {
        const size_t rank1 = x1Shape.GetDimNum();
        const size_t rank2 = x2Shape.GetDimNum();
        const size_t rank = std::max(rank1, rank2);
        if (rank > MAX_DIMS) {
            return false;
        }

        tiling->rank = static_cast<uint32_t>(rank);
        tiling->length = 1;
        sameShape = rank1 == rank2;

        uint64_t rawX1Shape[MAX_DIMS] = {0};
        uint64_t rawX2Shape[MAX_DIMS] = {0};
        for (size_t i = 0; i < rank1; ++i) {
            const int64_t dim = x1Shape.GetDim(i);
            if (dim < 0) {
                return false;
            }
            rawX1Shape[i] = static_cast<uint64_t>(dim);
        }
        for (size_t i = 0; i < rank2; ++i) {
            const int64_t dim = x2Shape.GetDim(i);
            if (dim < 0) {
                return false;
            }
            rawX2Shape[i] = static_cast<uint64_t>(dim);
        }

        uint64_t x1RawStride[MAX_DIMS] = {0};
        uint64_t x2RawStride[MAX_DIMS] = {0};
        uint64_t stride = 1;
        for (int64_t i = static_cast<int64_t>(rank1) - 1; i >= 0; --i) {
            x1RawStride[i] = stride;
            stride *= rawX1Shape[i];
        }
        stride = 1;
        for (int64_t i = static_cast<int64_t>(rank2) - 1; i >= 0; --i) {
            x2RawStride[i] = stride;
            stride *= rawX2Shape[i];
        }

        for (size_t i = 0; i < MAX_DIMS; ++i) {
            tiling->outShape[i] = 1;
            tiling->x1Stride[i] = 0;
            tiling->x2Stride[i] = 0;
        }

        for (size_t outDim = 0; outDim < rank; ++outDim) {
            const size_t x1DimIndex = outDim + rank1 >= rank ? outDim + rank1 - rank : MAX_DIMS;
            const size_t x2DimIndex = outDim + rank2 >= rank ? outDim + rank2 - rank : MAX_DIMS;
            const uint64_t dim1 = x1DimIndex == MAX_DIMS ? 1 : rawX1Shape[x1DimIndex];
            const uint64_t dim2 = x2DimIndex == MAX_DIMS ? 1 : rawX2Shape[x2DimIndex];
            uint64_t outDimValue = 0;
            if (!GetBroadcastDim(dim1, dim2, outDimValue)) {
                return false;
            }

            tiling->outShape[outDim] = outDimValue;
            tiling->x1Stride[outDim] = (dim1 == 1 || x1DimIndex == MAX_DIMS) ? 0 : x1RawStride[x1DimIndex];
            tiling->x2Stride[outDim] = (dim2 == 1 || x2DimIndex == MAX_DIMS) ? 0 : x2RawStride[x2DimIndex];
            tiling->length *= outDimValue;
            if (dim1 != dim2) {
                sameShape = false;
            }
        }

        // Remove unit dimensions and fold adjacent dimensions when both inputs
        // remain either contiguous or broadcast across the merged interval.
        uint64_t coalescedShape[MAX_DIMS] = {0};
        uint64_t coalescedX1Stride[MAX_DIMS] = {0};
        uint64_t coalescedX2Stride[MAX_DIMS] = {0};
        uint32_t coalescedRank = 0;
        for (size_t dim = 0; dim < rank; ++dim) {
            const uint64_t dimSize = tiling->outShape[dim];
            if (dimSize == 1) {
                continue;
            }
            bool canMerge = false;
            if (coalescedRank > 0) {
                const uint32_t previous = coalescedRank - 1;
                const bool x1Merge = (coalescedX1Stride[previous] == 0 && tiling->x1Stride[dim] == 0) ||
                    coalescedX1Stride[previous] == tiling->x1Stride[dim] * dimSize;
                const bool x2Merge = (coalescedX2Stride[previous] == 0 && tiling->x2Stride[dim] == 0) ||
                    coalescedX2Stride[previous] == tiling->x2Stride[dim] * dimSize;
                canMerge = x1Merge && x2Merge;
            }
            if (canMerge) {
                const uint32_t previous = coalescedRank - 1;
                coalescedShape[previous] *= dimSize;
                coalescedX1Stride[previous] = tiling->x1Stride[dim];
                coalescedX2Stride[previous] = tiling->x2Stride[dim];
            } else {
                coalescedShape[coalescedRank] = dimSize;
                coalescedX1Stride[coalescedRank] = tiling->x1Stride[dim];
                coalescedX2Stride[coalescedRank] = tiling->x2Stride[dim];
                ++coalescedRank;
            }
        }
        tiling->rank = coalescedRank;
        for (size_t dim = 0; dim < MAX_DIMS; ++dim) {
            tiling->outShape[dim] = dim < coalescedRank ? coalescedShape[dim] : 1;
            tiling->x1Stride[dim] = dim < coalescedRank ? coalescedX1Stride[dim] : 0;
            tiling->x2Stride[dim] = dim < coalescedRank ? coalescedX2Stride[dim] : 0;
        }
        return true;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        const gert::Tensor *tensor_x1 = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_x2 = context->GetRequiredInputTensor(1);
        const gert::StorageShape *shape_x1 = context->GetRequiredInputShape(0);
        const gert::StorageShape *shape_x2 = context->GetRequiredInputShape(1);
        if (tensor_x1 == nullptr || tensor_x2 == nullptr || shape_x1 == nullptr || shape_x2 == nullptr) {
            return ge::GRAPH_FAILED;
        }

        ge::DataType dtype_x1 = tensor_x1->GetDataType(); // 获取数据类型
        if (dtype_x1 != tensor_x2->GetDataType()) {
            return ge::GRAPH_FAILED;
        }

        LessEqualTilingData *tiling = context->GetTilingData<LessEqualTilingData>();
        bool sameShape = false;
        if (tiling == nullptr ||
            !MakeBroadcastInfo(shape_x1->GetStorageShape(), shape_x2->GetStorageShape(), tiling, sameShape)) {
            return ge::GRAPH_FAILED;
        }
        uint32_t DT_X1 = static_cast<uint32_t>(dtype_x1);
        uint32_t IS_BROADCAST = sameShape ? 0 : 1;
        uint32_t IS_SMALL = tiling->length <= SMALL_ELEMENT_THRESHOLD ? 1 : 0;
        ASCENDC_TPL_SEL_PARAM(context, DT_X1, IS_BROADCAST, IS_SMALL);
        tiling->tileLength = GetTileLength(platform, dtype_x1);
        // Large tiles reduce loop overhead for contiguous data. Broadcast work is
        // divided by tiles, so capping its tile preserves enough independent work
        // units to keep AIV cores occupied on a long single row.
        if (!sameShape) {
            tiling->tileLength = std::min(tiling->tileLength, MAX_BROADCAST_TILE_LENGTH);
        }
        if (tiling->length > 0) {
            tiling->tileLength = static_cast<uint32_t>(std::min<uint64_t>(tiling->tileLength, tiling->length));
        }

        const uint64_t desiredBlocks = tiling->length <= SMALL_ELEMENT_THRESHOLD ? 1 :
            (tiling->length + MIN_ELEMENTS_PER_CORE - 1) / MIN_ELEMENTS_PER_CORE;
        uint64_t availableWorkUnits = desiredBlocks;
        if (tiling->length > 0 && !sameShape && tiling->rank > 0) {
            const uint64_t lastDimLength = tiling->outShape[tiling->rank - 1];
            const uint64_t totalRows = tiling->length / lastDimLength;
            const uint64_t chunksPerRow = (lastDimLength + tiling->tileLength - 1) / tiling->tileLength;
            availableWorkUnits = totalRows * chunksPerRow;
        }
        const uint32_t blockDim = static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(
            static_cast<uint64_t>(num_cores_aiv), std::min(desiredBlocks, availableWorkUnits))));
        context->SetBlockDim(blockDim == 0 ? 1 : blockDim);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *x1Shape = context->GetRequiredInputShape(0);
        const gert::Shape *x2Shape = context->GetRequiredInputShape(1);
        gert::Shape *yShape = context->GetOutputShape(0);
        if (x1Shape == nullptr || x2Shape == nullptr || yShape == nullptr) {
            return GRAPH_FAILED;
        }

        const size_t rank1 = x1Shape->GetDimNum();
        const size_t rank2 = x2Shape->GetDimNum();
        const size_t rank = std::max(rank1, rank2);
        if (rank > gert::Shape::kMaxDimNum) {
            return GRAPH_FAILED;
        }

        yShape->SetDimNum(0);
        for (size_t outDim = 0; outDim < rank; ++outDim) {
            const bool hasDim1 = outDim + rank1 >= rank;
            const bool hasDim2 = outDim + rank2 >= rank;
            const int64_t dim1 = hasDim1 ? x1Shape->GetDim(outDim + rank1 - rank) : 1;
            const int64_t dim2 = hasDim2 ? x2Shape->GetDim(outDim + rank2 - rank) : 1;
            if (dim1 < 0 || dim2 < 0) {
                return GRAPH_FAILED;
            }
            uint64_t outputDim = 0;
            if (!optiling::GetBroadcastDim(static_cast<uint64_t>(dim1), static_cast<uint64_t>(dim2), outputDim)) {
                return GRAPH_FAILED;
            }
            yShape->AppendDim(static_cast<int64_t>(outputDim));
        }
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        if (context == nullptr || context->GetRequiredInputDataType(0) != context->GetRequiredInputDataType(1)) {
            return ge::GRAPH_FAILED;
        }
        return context->SetOutputDataType(0, ge::DT_BOOL);
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

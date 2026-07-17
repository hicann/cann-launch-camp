#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/less_equal_tiling.h"
#include "../op_kernel/tiling_key_less_equal.h"

namespace optiling {
    namespace {
    constexpr uint64_t ELEMENTS_PER_CORE = 8192;

    bool BuildBroadcastInfo(const gert::Shape &x1Shape, const gert::Shape &x2Shape,
                            LessEqualTilingData &tiling) {
        const size_t x1Rank = x1Shape.GetDimNum();
        const size_t x2Rank = x2Shape.GetDimNum();
        const size_t rank = std::max(x1Rank, x2Rank);
        if (rank > LESS_EQUAL_MAX_DIMS) {
            return false;
        }

        tiling = {};

        uint64_t x1Dims[LESS_EQUAL_MAX_DIMS] = {1, 1, 1, 1, 1, 1, 1, 1};
        uint64_t x2Dims[LESS_EQUAL_MAX_DIMS] = {1, 1, 1, 1, 1, 1, 1, 1};
        for (size_t i = 0; i < x1Rank; ++i) {
            const int64_t dim = x1Shape.GetDim(i);
            if (dim < 0) {
                return false;
            }
            x1Dims[rank - x1Rank + i] = static_cast<uint64_t>(dim);
        }
        for (size_t i = 0; i < x2Rank; ++i) {
            const int64_t dim = x2Shape.GetDim(i);
            if (dim < 0) {
                return false;
            }
            x2Dims[rank - x2Rank + i] = static_cast<uint64_t>(dim);
        }

        tiling.outputLength = 1;
        tiling.rank = static_cast<uint32_t>(rank);
        bool sameShape = x1Rank == x2Rank;
        for (size_t i = 0; i < rank; ++i) {
            const uint64_t x1Dim = x1Dims[i];
            const uint64_t x2Dim = x2Dims[i];
            uint64_t outputDim = 0;
            if (x1Dim == x2Dim) {
                outputDim = x1Dim;
            } else if (x1Dim == 1) {
                outputDim = x2Dim;
                sameShape = false;
            } else if (x2Dim == 1) {
                outputDim = x1Dim;
                sameShape = false;
            } else {
                return false;
            }
            tiling.outputShape[i] = outputDim;
            if (outputDim != 0 &&
                tiling.outputLength > std::numeric_limits<uint64_t>::max() / outputDim) {
                return false;
            }
            tiling.outputLength *= outputDim;
        }

        uint64_t x1Stride = 1;
        uint64_t x2Stride = 1;
        for (int32_t i = static_cast<int32_t>(rank) - 1; i >= 0; --i) {
            tiling.x1Stride[i] = x1Dims[i] == 1 && tiling.outputShape[i] != 1 ? 0 : x1Stride;
            tiling.x2Stride[i] = x2Dims[i] == 1 && tiling.outputShape[i] != 1 ? 0 : x2Stride;
            if ((x1Dims[i] != 0 && x1Stride > std::numeric_limits<uint64_t>::max() / x1Dims[i]) ||
                (x2Dims[i] != 0 && x2Stride > std::numeric_limits<uint64_t>::max() / x2Dims[i])) {
                return false;
            }
            x1Stride *= x1Dims[i];
            x2Stride *= x2Dims[i];
            if (x1Dims[i] != x2Dims[i]) {
                sameShape = false;
            }
        }
        tiling.x1Length = x1Stride;
        tiling.x2Length = x2Stride;

        if (sameShape) {
            tiling.mode = LESS_EQUAL_NO_BROADCAST;
        } else if (tiling.x1Length == 1) {
            tiling.mode = LESS_EQUAL_X1_SCALAR;
        } else if (tiling.x2Length == 1) {
            tiling.mode = LESS_EQUAL_X2_SCALAR;
        } else {
            tiling.mode = LESS_EQUAL_GENERAL_BROADCAST;
        }
        return true;
    }
    }  // namespace

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
        const gert::Tensor *x1 = context->GetRequiredInputTensor(0);
        const gert::Tensor *x2 = context->GetRequiredInputTensor(1);
        if (x1 == nullptr || x2 == nullptr || x1->GetDataType() != x2->GetDataType()) {
            return ge::GRAPH_FAILED;
        }

        LessEqualTilingData *tiling = context->GetTilingData<LessEqualTilingData>();
        if (tiling == nullptr || !BuildBroadcastInfo(x1->GetOriginShape(), x2->GetOriginShape(), *tiling)) {
            return ge::GRAPH_FAILED;
        }

        const uint32_t DT_X1 = static_cast<uint32_t>(x1->GetDataType());
        ASCENDC_TPL_SEL_PARAM(context, DT_X1);

        const uint64_t usefulCores = tiling->outputLength == 0
                                         ? 0
                                         : 1 + (tiling->outputLength - 1) / ELEMENTS_PER_CORE;
        coreNum = static_cast<uint32_t>(std::min<uint64_t>(std::max<uint64_t>(usefulCores, 1),
                                                           std::max<uint32_t>(coreNum, 1)));
        context->SetBlockDim(coreNum);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *x1Shape = context->GetInputShape(0);
        const gert::Shape *x2Shape = context->GetInputShape(1);
        gert::Shape *yShape = context->GetOutputShape(0);
        if (x1Shape == nullptr || x2Shape == nullptr || yShape == nullptr) {
            return GRAPH_FAILED;
        }

        const size_t x1Rank = x1Shape->GetDimNum();
        const size_t x2Rank = x2Shape->GetDimNum();
        const size_t rank = std::max(x1Rank, x2Rank);
        if (rank > LESS_EQUAL_MAX_DIMS) {
            return GRAPH_FAILED;
        }
        yShape->SetDimNum(rank);
        for (size_t i = 0; i < rank; ++i) {
            const int64_t x1Dim = i < rank - x1Rank ? 1 : x1Shape->GetDim(i - (rank - x1Rank));
            const int64_t x2Dim = i < rank - x2Rank ? 1 : x2Shape->GetDim(i - (rank - x2Rank));
            int64_t outputDim = 0;
            if (x1Dim == x2Dim) {
                outputDim = x1Dim;
            } else if (x1Dim == 1) {
                outputDim = x2Dim;
            } else if (x2Dim == 1) {
                outputDim = x1Dim;
            } else {
                return GRAPH_FAILED;
            }
            yShape->SetDim(i, outputDim);
        }
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        if (context->GetInputDataType(0) != context->GetInputDataType(1)) {
            return GRAPH_FAILED;
        }
        context->SetOutputDataType(0, ge::DT_BOOL);
        return ge::GRAPH_SUCCESS;
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

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/less_equal_tiling.h"
#include "../op_kernel/tiling_key_less_equal.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const auto *x1StorageShape = context->GetInputShape(0);
    const auto *x2StorageShape = context->GetInputShape(1);
    if (x1StorageShape == nullptr || x2StorageShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &x1Shape = x1StorageShape->GetOriginShape();
    const auto &x2Shape = x2StorageShape->GetOriginShape();
    const uint32_t x1Rank = static_cast<uint32_t>(x1Shape.GetDimNum());
    const uint32_t x2Rank = static_cast<uint32_t>(x2Shape.GetDimNum());
    const uint32_t rank = x1Rank > x2Rank ? x1Rank : x2Rank;
    if (rank > LESS_EQUAL_MAX_DIMS) {
        return ge::GRAPH_FAILED;
    }

    LessEqualTilingData *tiling = context->GetTilingData<LessEqualTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->rank = rank;
    tiling->noBroadcast = 1;
    for (uint32_t i = 0; i < LESS_EQUAL_MAX_DIMS; ++i) {
        tiling->outShape[i] = 1;
        tiling->outStride[i] = 0;
        tiling->x1Stride[i] = 0;
        tiling->x2Stride[i] = 0;
    }

    uint32_t x1Dims[LESS_EQUAL_MAX_DIMS] = {0};
    uint32_t x2Dims[LESS_EQUAL_MAX_DIMS] = {0};
    for (uint32_t i = 0; i < rank; ++i) {
        const int32_t x1Index = static_cast<int32_t>(i) -
                                static_cast<int32_t>(rank - x1Rank);
        const int32_t x2Index = static_cast<int32_t>(i) -
                                static_cast<int32_t>(rank - x2Rank);
        const uint32_t d1 = x1Index >= 0
            ? static_cast<uint32_t>(x1Shape.GetDim(static_cast<size_t>(x1Index))) : 1;
        const uint32_t d2 = x2Index >= 0
            ? static_cast<uint32_t>(x2Shape.GetDim(static_cast<size_t>(x2Index))) : 1;
        x1Dims[i] = d1;
        x2Dims[i] = d2;
        if (d1 == d2) {
            tiling->outShape[i] = d1;
        } else if (d1 == 1) {
            tiling->outShape[i] = d2;
            tiling->noBroadcast = 0;
        } else if (d2 == 1) {
            tiling->outShape[i] = d1;
            tiling->noBroadcast = 0;
        } else {
            return ge::GRAPH_FAILED;
        }
        if (d1 != tiling->outShape[i] || d2 != tiling->outShape[i]) {
            tiling->noBroadcast = 0;
        }
    }

    uint64_t outRunning = 1;
    uint64_t x1Running = 1;
    uint64_t x2Running = 1;
    for (int32_t i = static_cast<int32_t>(rank) - 1; i >= 0; --i) {
        tiling->outStride[i] = static_cast<uint32_t>(outRunning);
        const bool x1Padded = static_cast<uint32_t>(i) < rank - x1Rank;
        const bool x2Padded = static_cast<uint32_t>(i) < rank - x2Rank;
        tiling->x1Stride[i] = (x1Padded ||
            (x1Dims[i] == 1 && tiling->outShape[i] != 1))
            ? 0 : static_cast<uint32_t>(x1Running);
        tiling->x2Stride[i] = (x2Padded ||
            (x2Dims[i] == 1 && tiling->outShape[i] != 1))
            ? 0 : static_cast<uint32_t>(x2Running);
        outRunning *= tiling->outShape[i];
        if (!x1Padded) {
            x1Running *= x1Dims[i];
        }
        if (!x2Padded) {
            x2Running *= x2Dims[i];
        }
        if (outRunning > 0xFFFFFFFFULL || x1Running > 0xFFFFFFFFULL ||
            x2Running > 0xFFFFFFFFULL) {
            return ge::GRAPH_FAILED;
        }
    }
    tiling->outputLength = static_cast<uint32_t>(outRunning);

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t numCores = platform.GetCoreNumAiv();
    const gert::Tensor *tensorX1 = context->GetRequiredInputTensor(0);
    if (tensorX1 == nullptr || numCores <= 0) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t DT_X1 = static_cast<uint32_t>(tensorX1->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X1);

    uint32_t blockDim = 1;
if (tiling->outputLength > 0) {
    // 原来4095 → 改成8191，单片更大
    blockDim = (tiling->outputLength + 8191) / 8192;
    if (blockDim > static_cast<uint32_t>(numCores)) {
        blockDim = static_cast<uint32_t>(numCores);
    }
}
tiling->blockLength = tiling->outputLength == 0 ? 0 :
    (tiling->outputLength + blockDim - 1) / blockDim;
context->SetBlockDim(blockDim);
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
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
    const size_t rank = x1Rank > x2Rank ? x1Rank : x2Rank;
    yShape->SetDimNum(0);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t x1Index = static_cast<int64_t>(i) -
                                static_cast<int64_t>(rank - x1Rank);
        const int64_t x2Index = static_cast<int64_t>(i) -
                                static_cast<int64_t>(rank - x2Rank);
        const int64_t d1 = x1Index >= 0 ? x1Shape->GetDim(x1Index) : 1;
        const int64_t d2 = x2Index >= 0 ? x2Shape->GetDim(x2Index) : 1;
        if (d1 == d2) {
            yShape->AppendDim(d1);
        } else if (d1 == 1) {
            yShape->AppendDim(d2);
        } else if (d2 == 1) {
            yShape->AppendDim(d1);
        } else {
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
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
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/less_equal_tiling.h"
#include "../op_kernel/tiling_key_less_equal.h"

#include <limits>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t availableCores = platform.GetCoreNumAiv();

    const gert::Tensor *tensorX1 = context->GetRequiredInputTensor(0);
    const gert::StorageShape *storageX1 = context->GetRequiredInputShape(0);
    const gert::StorageShape *storageX2 = context->GetRequiredInputShape(1);
    if (tensorX1 == nullptr || storageX1 == nullptr || storageX2 == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType dtypeX1 = tensorX1->GetDataType();
    uint32_t dtypeKey = static_cast<uint32_t>(dtypeX1);
    ASCENDC_TPL_SEL_PARAM(context, dtypeKey);

    LessEqualTilingData *tiling = context->GetTilingData<LessEqualTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &shapeX1 = storageX1->GetOriginShape();
    const gert::Shape &shapeX2 = storageX2->GetOriginShape();
    const size_t rank1 = shapeX1.GetDimNum();
    const size_t rank2 = shapeX2.GetDimNum();
    const size_t outRank = rank1 > rank2 ? rank1 : rank2;
    if (outRank > LESS_EQUAL_MAX_DIMS) {
        return ge::GRAPH_FAILED;
    }

    for (uint32_t i = 0; i < LESS_EQUAL_MAX_DIMS; ++i) {
        tiling->outDims[i] = 1;
        tiling->x1Strides[i] = 0;
        tiling->x2Strides[i] = 0;
    }

    const size_t offset1 = outRank - rank1;
    const size_t offset2 = outRank - rank2;
    uint64_t outputLength = 1;
    bool isBroadcast = rank1 != rank2;
    for (size_t axis = 0; axis < outRank; ++axis) {
        const int64_t dim1 = axis < offset1 ? 1 : shapeX1.GetDim(axis - offset1);
        const int64_t dim2 = axis < offset2 ? 1 : shapeX2.GetDim(axis - offset2);
        if (dim1 < 0 || dim2 < 0 ||
            (dim1 != dim2 && dim1 != 1 && dim2 != 1)) {
            return ge::GRAPH_FAILED;
        }
        const int64_t outDim = dim1 == 1 ? dim2 : dim1;
        tiling->outDims[axis] = static_cast<uint32_t>(outDim);
        outputLength *= static_cast<uint64_t>(outDim);
        if (outputLength > std::numeric_limits<uint32_t>::max()) {
            return ge::GRAPH_FAILED;
        }
        if (dim1 != outDim || dim2 != outDim) {
            isBroadcast = true;
        }
    }

    uint64_t stride1 = 1;
    uint64_t stride2 = 1;
    for (size_t reverse = 0; reverse < outRank; ++reverse) {
        const size_t axis = outRank - 1 - reverse;
        const int64_t dim1 = axis < offset1 ? 1 : shapeX1.GetDim(axis - offset1);
        const int64_t dim2 = axis < offset2 ? 1 : shapeX2.GetDim(axis - offset2);
        if (axis >= offset1) {
            tiling->x1Strides[axis] =
                (dim1 == 1 && tiling->outDims[axis] != 1) ? 0 : static_cast<uint32_t>(stride1);
            stride1 *= static_cast<uint64_t>(dim1);
        }
        if (axis >= offset2) {
            tiling->x2Strides[axis] =
                (dim2 == 1 && tiling->outDims[axis] != 1) ? 0 : static_cast<uint32_t>(stride2);
            stride2 *= static_cast<uint64_t>(dim2);
        }
    }

    const uint32_t length = static_cast<uint32_t>(outputLength);

    constexpr uint32_t minElementsPerCore = 4096;
    uint32_t neededCores = static_cast<uint32_t>(
        (static_cast<uint64_t>(length) + minElementsPerCore - 1) /
        minElementsPerCore);
    if (neededCores == 0) {
        neededCores = 1;
    }
    uint32_t coreCount = static_cast<uint32_t>(availableCores > 0 ? availableCores : 1);
    if (coreCount > neededCores) {
        coreCount = neededCores;
    }

    tiling->length = length;
    if (length == 0) {
        tiling->blockLength = 0;
    } else {
        tiling->blockLength = static_cast<uint32_t>(
            (static_cast<uint64_t>(length) + coreCount - 1) / coreCount);
    }
    tiling->rank = static_cast<uint32_t>(outRank);
    tiling->isBroadcast = isBroadcast ? 1U : 0U;
    tiling->scalarBroadcast = 0;
    if (isBroadcast && stride1 == 1 && stride2 == outputLength) {
        tiling->scalarBroadcast = 1;
    } else if (isBroadcast && stride2 == 1 && stride1 == outputLength) {
        tiling->scalarBroadcast = 2;
    }

    context->SetBlockDim(coreCount);
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x1Shape = context->GetInputShape(0);
    const gert::Shape *x2Shape = context->GetInputShape(1);
    gert::Shape *yShape = context->GetOutputShape(0);
    if (x1Shape == nullptr || x2Shape == nullptr || yShape == nullptr) {
        return GRAPH_FAILED;
    }

    const size_t rank1 = x1Shape->GetDimNum();
    const size_t rank2 = x2Shape->GetDimNum();
    const size_t outRank = rank1 > rank2 ? rank1 : rank2;
    yShape->SetDimNum(0);

    for (size_t outAxis = 0; outAxis < outRank; ++outAxis) {
        const size_t fromRight = outRank - 1 - outAxis;
        const int64_t dim1 = fromRight < rank1
            ? x1Shape->GetDim(rank1 - 1 - fromRight) : 1;
        const int64_t dim2 = fromRight < rank2
            ? x2Shape->GetDim(rank2 - 1 - fromRight) : 1;

        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            return GRAPH_FAILED;
        }
        const int64_t outDim = dim1 == 1 ? dim2 : dim1;
        yShape->AppendDim(outDim);
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const ge::DataType x1Type = context->GetInputDataType(0);
    const ge::DataType x2Type = context->GetInputDataType(1);
    if (x1Type == ge::DT_UNDEFINED || x1Type != x2Type) {
        return GRAPH_FAILED;
    }
    return context->SetOutputDataType(0, ge::DT_BOOL);
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
}

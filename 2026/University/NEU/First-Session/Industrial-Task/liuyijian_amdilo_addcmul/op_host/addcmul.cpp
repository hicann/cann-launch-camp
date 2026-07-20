// Host侧Tiling实现
#include "register/op_def_registry.h"

#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/addcmul_tiling.h"

#include "../op_kernel/tiling_key_addcmul.h"

namespace {
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 2;

static bool BroadcastTwoShapes(const gert::Shape &a, const gert::Shape &b, gert::Shape &out)
{
    const int64_t dimA = static_cast<int64_t>(a.GetDimNum());
    const int64_t dimB = static_cast<int64_t>(b.GetDimNum());
    const int64_t dim = dimA > dimB ? dimA : dimB;
    out.SetDimNum(static_cast<size_t>(dim));
    for (int64_t i = 0; i < dim; ++i) {
        const int64_t aIdx = dimA - 1 - i;
        const int64_t bIdx = dimB - 1 - i;
        const int64_t aDim = aIdx >= 0 ? a.GetDim(aIdx) : 1;
        const int64_t bDim = bIdx >= 0 ? b.GetDim(bIdx) : 1;
        if (aDim == bDim) {
            out.SetDim(static_cast<size_t>(dim - 1 - i), aDim);
        } else if (aDim == 1) {
            out.SetDim(static_cast<size_t>(dim - 1 - i), bDim);
        } else if (bDim == 1) {
            out.SetDim(static_cast<size_t>(dim - 1 - i), aDim);
        } else {
            return false;
        }
    }
    return true;
}

static void PadShapeTo(const gert::Shape &src, uint32_t dimNum, uint32_t *dst)
{
    const uint32_t srcDim = static_cast<uint32_t>(src.GetDimNum());
    for (uint32_t i = 0; i < dimNum; ++i) {
        const int32_t srcIdx = static_cast<int32_t>(i) - static_cast<int32_t>(dimNum - srcDim);
        dst[i] = srcIdx >= 0 ? static_cast<uint32_t>(src.GetDim(static_cast<size_t>(srcIdx))) : 1U;
    }
}

static void CalcBroadcastStride(const uint32_t *shape, uint32_t dimNum, uint32_t *stride)
{
    uint32_t acc = 1;
    for (int32_t i = static_cast<int32_t>(dimNum) - 1; i >= 0; --i) {
        stride[i] = (shape[i] == 1U) ? 0U : acc;
        acc *= shape[i];
    }
}

static void CalcNormalStride(const uint32_t *shape, uint32_t dimNum, uint32_t *stride)
{
    if (dimNum == 0) {
        return;
    }
    stride[dimNum - 1] = 1;
    for (int32_t i = static_cast<int32_t>(dimNum) - 2; i >= 0; --i) {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
}

static bool IsSameShape(const uint32_t *a, const uint32_t *b, uint32_t dimNum)
{
    for (uint32_t i = 0; i < dimNum; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

static uint32_t CeilAlign(uint32_t value, uint32_t align)
{
    if (align == 0) {
        return value;
    }
    return (value + align - 1) / align * align;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t numCoresAiv = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const gert::StorageShape *inputStorage = context->GetInputShape(0);
    const gert::StorageShape *x1Storage = context->GetInputShape(1);
    const gert::StorageShape *x2Storage = context->GetInputShape(2);
    const gert::Shape &inputShape = inputStorage->GetStorageShape();
    const gert::Shape &x1Shape = x1Storage->GetStorageShape();
    const gert::Shape &x2Shape = x2Storage->GetStorageShape();

    gert::Shape tmpShape;
    gert::Shape outShape;
    if (!BroadcastTwoShapes(inputShape, x1Shape, tmpShape) || !BroadcastTwoShapes(tmpShape, x2Shape, outShape)) {
        return ge::GRAPH_FAILED;
    }

    const gert::Tensor *tensorInput = context->GetRequiredInputTensor(0);
    ge::DataType dtype = tensorInput->GetDataType();
    int32_t dtypeSize = ge::GetSizeByDataType(dtype);
    if (dtypeSize <= 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t DT_INPUT_DATA = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_DATA);

    AddcmulTilingData *tiling = context->GetTilingData<AddcmulTilingData>();
    const uint32_t dimNum = static_cast<uint32_t>(outShape.GetDimNum());
    if (dimNum > ADDCMUL_MAX_DIMS) {
        return ge::GRAPH_FAILED;
    }

    tiling->dimNum = dimNum;
    for (uint32_t i = 0; i < ADDCMUL_MAX_DIMS; ++i) {
        tiling->outShape[i] = 1;
        tiling->inputShape[i] = 1;
        tiling->x1Shape[i] = 1;
        tiling->x2Shape[i] = 1;
        tiling->outStride[i] = 0;
        tiling->inputStride[i] = 0;
        tiling->x1Stride[i] = 0;
        tiling->x2Stride[i] = 0;
    }

    if (dimNum > 0) {
        for (uint32_t i = 0; i < dimNum; ++i) {
            tiling->outShape[i] = static_cast<uint32_t>(outShape.GetDim(i));
        }
        PadShapeTo(inputShape, dimNum, tiling->inputShape);
        PadShapeTo(x1Shape, dimNum, tiling->x1Shape);
        PadShapeTo(x2Shape, dimNum, tiling->x2Shape);
        CalcNormalStride(tiling->outShape, dimNum, tiling->outStride);
        CalcBroadcastStride(tiling->inputShape, dimNum, tiling->inputStride);
        CalcBroadcastStride(tiling->x1Shape, dimNum, tiling->x1Stride);
        CalcBroadcastStride(tiling->x2Shape, dimNum, tiling->x2Stride);
    }

    uint64_t totalLength = 1;
    for (uint32_t i = 0; i < dimNum; ++i) {
        totalLength *= tiling->outShape[i];
    }
    for (uint32_t i = 0; i < dimNum; ++i) {
        if (tiling->outShape[i] == 0) {
            totalLength = 0;
            break;
        }
    }
    tiling->totalLength = static_cast<uint32_t>(totalLength);

    const bool sameShape = (dimNum == 0) ||
        (IsSameShape(tiling->inputShape, tiling->outShape, dimNum) &&
         IsSameShape(tiling->x1Shape, tiling->outShape, dimNum) &&
         IsSameShape(tiling->x2Shape, tiling->outShape, dimNum));
    tiling->isBroadcast = sameShape ? 0U : 1U;

    if (totalLength == 0) {
        tiling->tileLength = 0;
        tiling->formerNum = 0;
        tiling->formerLength = 0;
        tiling->tailBlockLength = 0;
        context->SetBlockDim(1);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    const uint32_t alignNum = BLOCK_SIZE / static_cast<uint32_t>(dtypeSize);
    // 4路双缓冲
    constexpr uint64_t ubDataNumber = 4;
    uint64_t tileBlockNum = (ubSize / BLOCK_SIZE / BUFFER_NUM) / ubDataNumber;
    if (tileBlockNum == 0) {
        tileBlockNum = 1;
    }
    uint32_t tileLength = static_cast<uint32_t>((tileBlockNum * BLOCK_SIZE) / static_cast<uint32_t>(dtypeSize));
    tileLength = tileLength / alignNum * alignNum;
    if (tileLength < alignNum) {
        tileLength = alignNum;
    }
    // tileLength 保持 32B 对齐；小数据时向上对齐，禁止缩成未对齐长度
    const uint32_t totalAligned = CeilAlign(static_cast<uint32_t>(totalLength), alignNum);
    if (tileLength > totalAligned) {
        tileLength = totalAligned;
    }
    tiling->tileLength = tileLength;

    int32_t blockDim = numCoresAiv;
    // 数据量较小时单核，避免多分核边界问题
    if (static_cast<uint32_t>(totalLength) <= tileLength) {
        blockDim = 1;
    } else if (blockDim > static_cast<int32_t>(totalLength)) {
        blockDim = static_cast<int32_t>(totalLength);
    }
    if (blockDim <= 0) {
        blockDim = 1;
    }

    const uint32_t baseLen = static_cast<uint32_t>(totalLength / static_cast<uint64_t>(blockDim));
    const uint32_t remainder = static_cast<uint32_t>(totalLength % static_cast<uint64_t>(blockDim));
    tiling->formerNum = remainder;
    tiling->formerLength = baseLen + 1;
    tiling->tailBlockLength = baseLen;

    context->SetBlockDim(blockDim);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    const gert::Shape *x1Shape = context->GetInputShape(1);
    const gert::Shape *x2Shape = context->GetInputShape(2);
    gert::Shape *yShape = context->GetOutputShape(0);
    if (inputShape == nullptr || x1Shape == nullptr || x2Shape == nullptr || yShape == nullptr) {
        return GRAPH_FAILED;
    }
    gert::Shape tmpShape;
    if (!BroadcastTwoShapes(*inputShape, *x1Shape, tmpShape) || !BroadcastTwoShapes(tmpShape, *x2Shape, *yShape)) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Addcmul : public OpDef {
public:
    explicit Addcmul(const char *name) : OpDef(name)
    {
        this->Input("input_data")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Addcmul);
}  // namespace ops

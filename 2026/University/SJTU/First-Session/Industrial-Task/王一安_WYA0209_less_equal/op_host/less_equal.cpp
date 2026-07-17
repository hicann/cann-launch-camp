// LessEqual Host 侧：算子注册、Shape/类型推导及 Tiling。
// 支持广播机制，输出始终为 bool 类型。
#include <algorithm>
#include <cstdint>
#include <vector>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/less_equal_tiling.h"
#include "../op_kernel/tiling_key_less_equal.h"

namespace optiling {
namespace {
constexpr uint32_t DATA_BLOCK_BYTES = 32U;
constexpr uint32_t BUFFER_FACTOR = 4U;       // VECIN 双缓冲 + VECOUT 双缓冲
constexpr uint32_t UB_SAFETY_DIVISOR = 2U;   // 仅使用约一半 UB
constexpr uint32_t MAX_TILE_ELEMENTS = 16384U;
constexpr uint32_t MIN_ELEMENTS_PER_CORE = 4096U;

inline uint64_t CeilDiv(uint64_t value, uint64_t divisor) {
    return (value + divisor - 1U) / divisor;
}

inline uint64_t AlignDown(uint64_t value, uint64_t alignment) {
    return value / alignment * alignment;
}

// 计算 NumPy 风格的广播形状，同时计算两个输入在广播空间中的 stride。
// stride[i] = 该输入在第 i 维上，flat index 增加 1 时对应的元素偏移。
// 若某维度大小为 1（广播维度），stride 设为 0。
static bool ComputeBroadcastShapeAndStrides(
    const std::vector<int64_t> &shape0,
    const std::vector<int64_t> &shape1,
    std::vector<int64_t> &outShape,
    std::vector<uint32_t> &stride0,
    std::vector<uint32_t> &stride1)
{
    // 从右向左对齐两个 shape
    size_t ndim0 = shape0.size();
    size_t ndim1 = shape1.size();
    size_t maxNdim = std::max(ndim0, ndim1);

    outShape.resize(maxNdim);
    stride0.resize(maxNdim, 0);
    stride1.resize(maxNdim, 0);

    // 先计算输出形状
    for (size_t i = 0; i < maxNdim; i++) {
        // 从右往左数第 i 维
        size_t idx0 = (i < ndim0) ? (ndim0 - 1 - i) : (size_t)-1;
        size_t idx1 = (i < ndim1) ? (ndim1 - 1 - i) : (size_t)-1;

        int64_t d0 = (idx0 != (size_t)-1) ? shape0[idx0] : 1;
        int64_t d1 = (idx1 != (size_t)-1) ? shape1[idx1] : 1;

        if (d0 != d1 && d0 != 1 && d1 != 1) {
            return false;  // 无法广播
        }
        outShape[maxNdim - 1 - i] = std::max(d0, d1);
    }

    // 计算 stride（从最外层到最内层）
    // stride[i] 表示在 flat index 中，第 i 维每增加 1，对应输入 flat index 的增量
    uint32_t runningStride = 1;
    for (int i = (int)maxNdim - 1; i >= 0; i--) {
        int64_t d0 = (i < (int)ndim0) ? shape0[i] : 1;
        int64_t d1 = (i < (int)ndim1) ? shape1[i] : 1;

        stride0[i] = (d0 == 1) ? 0 : runningStride;
        stride1[i] = (d1 == 1) ? 0 : runningStride;

        runningStride *= (uint32_t)outShape[i];
    }

    return true;
}
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    int32_t availableCores = platform.GetCoreNumAiv();
    if (availableCores <= 0) {
        availableCores = 1;
    }

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // 获取两个输入的信息
    const gert::Tensor *tensor0 = context->GetRequiredInputTensor(0);
    const gert::Tensor *tensor1 = context->GetRequiredInputTensor(1);

    ge::DataType dtype0 = tensor0->GetDataType();
    int32_t dtypeSize = ge::GetSizeByDataType(dtype0);
    if (dtypeSize <= 0) {
        return ge::GRAPH_FAILED;
    }

    // 获取两个输入的 shape
    const gert::Shape *shapePtr0 = context->GetInputShape(0);
    const gert::Shape *shapePtr1 = context->GetInputShape(1);

    auto storageShape0 = shapePtr0->GetStorageShape();
    auto storageShape1 = shapePtr1->GetStorageShape();

    std::vector<int64_t> shape0, shape1;
    for (int i = 0; i < storageShape0.GetDimNum(); i++) {
        shape0.push_back(storageShape0.GetDim(i));
    }
    for (int i = 0; i < storageShape1.GetDimNum(); i++) {
        shape1.push_back(storageShape1.GetDim(i));
    }

    // 计算广播形状和 stride
    std::vector<int64_t> outShape;
    std::vector<uint32_t> stride0, stride1;
    if (!ComputeBroadcastShapeAndStrides(shape0, shape1, outShape, stride0, stride1)) {
        return ge::GRAPH_FAILED;
    }

    // 计算总元素数
    uint64_t totalLength = 1;
    for (auto d : outShape) {
        totalLength *= (uint64_t)d;
    }

    if (totalLength == 0U) {
        // 空张量
        LessEqualTilingData *tiling = context->GetTilingData<LessEqualTilingData>();
        tiling->totalLength = 0;
        tiling->blockLength = 1;
        tiling->tileLength = DATA_BLOCK_BYTES / (uint32_t)dtypeSize;
        tiling->ndim = (uint32_t)outShape.size();
        for (size_t i = 0; i < outShape.size() && i < MAX_NDIM; i++) {
            tiling->outShape[i] = (uint32_t)outShape[i];
            tiling->stride0[i] = stride0[i];
            tiling->stride1[i] = stride1[i];
        }
        context->SetBlockDim(1);
        size_t *workspace = context->GetWorkspaceSizes(1);
        workspace[0] = 0U;
        return ge::GRAPH_SUCCESS;
    }

    // 按数据规模选择实际启动核数
    uint64_t desiredCores = CeilDiv(totalLength, (uint64_t)MIN_ELEMENTS_PER_CORE);
    desiredCores = std::max<uint64_t>(1U, desiredCores);
    const uint32_t usedCores = static_cast<uint32_t>(
        std::min<uint64_t>(desiredCores, static_cast<uint64_t>(availableCores)));

    const uint64_t blockLength = CeilDiv(totalLength, usedCores);

    // 计算 tile 大小
    const uint32_t alignElements = DATA_BLOCK_BYTES / static_cast<uint32_t>(dtypeSize);
    uint64_t tileElementsByUb = ubSize /
        (BUFFER_FACTOR * UB_SAFETY_DIVISOR * static_cast<uint64_t>(dtypeSize));
    tileElementsByUb = AlignDown(tileElementsByUb, alignElements);

    uint32_t tileLength = static_cast<uint32_t>(
        std::min<uint64_t>(tileElementsByUb, (uint64_t)MAX_TILE_ELEMENTS));
    if (tileLength < alignElements) {
        tileLength = alignElements;
    }

    // 设置 tiling 数据
    const uint32_t dtX = static_cast<uint32_t>(dtype0);
    ASCENDC_TPL_SEL_PARAM(context, dtX);

    LessEqualTilingData *tiling = context->GetTilingData<LessEqualTilingData>();
    tiling->totalLength = totalLength;
    tiling->blockLength = blockLength;
    tiling->tileLength = tileLength;
    tiling->ndim = static_cast<uint32_t>(outShape.size());

    for (size_t i = 0; i < outShape.size() && i < MAX_NDIM; i++) {
        tiling->outShape[i] = static_cast<uint32_t>(outShape[i]);
        tiling->stride0[i] = stride0[i];
        tiling->stride1[i] = stride1[i];
    }

    context->SetBlockDim(usedCores);
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0U;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *inputShape0 = context->GetInputShape(0);
    const gert::Shape *inputShape1 = context->GetInputShape(1);
    gert::Shape *outputShape = context->GetOutputShape(0);

    if (!inputShape0 || !inputShape1 || !outputShape) {
        return GRAPH_FAILED;
    }

    auto storageShape0 = inputShape0->GetStorageShape();
    auto storageShape1 = inputShape1->GetStorageShape();

    // 收集维度信息
    std::vector<int64_t> shape0, shape1;
    for (int i = 0; i < storageShape0.GetDimNum(); i++) {
        shape0.push_back(storageShape0.GetDim(i));
    }
    for (int i = 0; i < storageShape1.GetDimNum(); i++) {
        shape1.push_back(storageShape1.GetDim(i));
    }

    // 计算广播输出形状
    std::vector<int64_t> outShape;
    std::vector<uint32_t> stride0, stride1;
    if (!optiling::ComputeBroadcastShapeAndStrides(shape0, shape1, outShape, stride0, stride1)) {
        return GRAPH_FAILED;
    }

    // 设置输出 shape
    outputShape->SetDimNum(static_cast<int>(outShape.size()));
    for (size_t i = 0; i < outShape.size(); i++) {
        outputShape->SetDim(static_cast<int>(i), outShape[i]);
    }

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    // 输出始终为 bool 类型
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
            .DataType({ge::DT_BOOL})
            .Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(LessEqual);
}  // namespace ops

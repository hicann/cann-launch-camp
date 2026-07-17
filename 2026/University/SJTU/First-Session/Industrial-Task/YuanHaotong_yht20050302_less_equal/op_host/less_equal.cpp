#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/less_equal_tiling.h"
#include "../op_kernel/tiling_key_less_equal.h"

namespace {
constexpr uint32_t LESS_EQUAL_BUFFER_NUM = 2;
constexpr uint32_t LESS_EQUAL_TILE_ALIGN = 32;
constexpr uint32_t LESS_EQUAL_MIN_TILE_LENGTH = 32;
constexpr uint32_t LESS_EQUAL_SMALL_BLOCK_TILE_LENGTH = 256;
constexpr uint32_t LESS_EQUAL_SMALL_BLOCK_THRESHOLD = 128;
constexpr uint32_t LESS_EQUAL_INT32_SMALL_BLOCK_THRESHOLD = 216;
constexpr uint64_t LESS_EQUAL_SMALL_TOTAL_THRESHOLD = 2048;
constexpr uint32_t LESS_EQUAL_SMALL_TOTAL_BLOCK_DIM = 1;
constexpr uint64_t LESS_EQUAL_FULL_CORE_TOTAL_THRESHOLD = 4096;
constexpr uint64_t LESS_EQUAL_FLOAT32_FULL_CORE_MIN_TOTAL = 4097;
constexpr uint64_t LESS_EQUAL_TARGET_BLOCK_LENGTH = 1024;

uint32_t GetTypeSize(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
            return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        case ge::DT_INT8:
        case ge::DT_BOOL:
            return 1;
        default:
            return 4;
    }
}

uint32_t AlignUp32(uint32_t bytes)
{
    return (bytes + 31U) & ~31U;
}

uint32_t AlignDown(uint32_t value, uint32_t align)
{
    return value / align * align;
}

uint64_t GetUbBytesForTile(uint32_t tileLength, ge::DataType dtype)
{
    const uint32_t inputTypeSize = GetTypeSize(dtype);
    uint64_t bytes = 0;
    bytes += static_cast<uint64_t>(LESS_EQUAL_BUFFER_NUM) * AlignUp32(tileLength * inputTypeSize);
    bytes += static_cast<uint64_t>(LESS_EQUAL_BUFFER_NUM) * AlignUp32(tileLength * inputTypeSize);
    bytes += static_cast<uint64_t>(LESS_EQUAL_BUFFER_NUM) * AlignUp32(tileLength * GetTypeSize(ge::DT_BOOL));

    if (dtype == ge::DT_FLOAT16) {
        bytes += AlignUp32(tileLength * GetTypeSize(ge::DT_FLOAT16));
    } else if (dtype == ge::DT_FLOAT) {
        bytes += AlignUp32(tileLength * GetTypeSize(ge::DT_FLOAT));
        bytes += AlignUp32(tileLength * GetTypeSize(ge::DT_FLOAT16));
    } else if (dtype == ge::DT_INT8) {
        bytes += 3ULL * AlignUp32(tileLength * GetTypeSize(ge::DT_FLOAT16));
    } else {
        bytes += AlignUp32(tileLength * GetTypeSize(ge::DT_INT32));
        bytes += AlignUp32(tileLength * GetTypeSize(ge::DT_FLOAT16));
        bytes += AlignUp32(tileLength * GetTypeSize(ge::DT_FLOAT));
    }
    return bytes;
}

uint32_t CalcTileLength(uint64_t ubSize, ge::DataType dtype)
{
    const uint32_t inputTypeSize = GetTypeSize(dtype);
    uint32_t low = LESS_EQUAL_MIN_TILE_LENGTH;
    uint32_t high = AlignDown(static_cast<uint32_t>(std::max<uint64_t>(ubSize / inputTypeSize, low)),
        LESS_EQUAL_TILE_ALIGN);

    if (high < low || GetUbBytesForTile(low, dtype) > ubSize) {
        return low;
    }

    uint32_t best = low;
    while (low <= high) {
        const uint32_t mid = AlignDown(low + (high - low) / 2, LESS_EQUAL_TILE_ALIGN);
        if (mid < LESS_EQUAL_MIN_TILE_LENGTH) {
            break;
        }
        if (GetUbBytesForTile(mid, dtype) <= ubSize) {
            best = mid;
            low = mid + LESS_EQUAL_TILE_ALIGN;
        } else {
            high = mid - LESS_EQUAL_TILE_ALIGN;
        }
    }
    return best;
}

uint32_t SelectTileLength(uint64_t ubSize, ge::DataType dtype, uint64_t maxBlockLength)
{
    if (dtype == ge::DT_INT32 && maxBlockLength <= LESS_EQUAL_INT32_SMALL_BLOCK_THRESHOLD) {
        return LESS_EQUAL_SMALL_BLOCK_TILE_LENGTH;
    }
    if (maxBlockLength <= LESS_EQUAL_SMALL_BLOCK_THRESHOLD) {
        return LESS_EQUAL_SMALL_BLOCK_TILE_LENGTH;
    }
    return CalcTileLength(ubSize, dtype);
}

uint32_t SelectBlockDim(uint64_t total, int32_t coreNum, ge::DataType dtype)
{
    const uint32_t availableCores = static_cast<uint32_t>(coreNum > 0 ? coreNum : 1);
    if (total == 0) {
        return 1;
    }
    if (dtype == ge::DT_FLOAT16) {
        return static_cast<uint32_t>(std::min<uint64_t>(LESS_EQUAL_SMALL_TOTAL_BLOCK_DIM, total));
    }
    if (dtype == ge::DT_INT32) {
        return static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(availableCores, total)));
    }
    if (dtype == ge::DT_FLOAT && total >= LESS_EQUAL_FLOAT32_FULL_CORE_MIN_TOTAL) {
        return static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(availableCores, total)));
    }
    if (total <= 1024) {
        return static_cast<uint32_t>(std::min<uint64_t>(LESS_EQUAL_SMALL_TOTAL_BLOCK_DIM, total));
    }
    if (total <= LESS_EQUAL_FULL_CORE_TOTAL_THRESHOLD) {
        return static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(availableCores, total)));
    }
    const uint64_t neededCores = (total + LESS_EQUAL_TARGET_BLOCK_LENGTH - 1) / LESS_EQUAL_TARGET_BLOCK_LENGTH;
    return static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(availableCores, neededCores)));
}

std::vector<int64_t> GetShapeDims(const gert::Shape *shape)
{
    std::vector<int64_t> dims;
    if (shape == nullptr) {
        return dims;
    }
    const size_t rank = shape->GetDimNum();
    dims.reserve(rank);
    for (size_t i = 0; i < rank; ++i) {
        dims.push_back(shape->GetDim(i));
    }
    return dims;
}

std::vector<int64_t> PreferStorageShape(const gert::StorageShape *shape)
{
    if (shape == nullptr) {
        return {};
    }
    const gert::Shape &storageShape = shape->GetStorageShape();
    if (storageShape.GetDimNum() != 0) {
        return GetShapeDims(&storageShape);
    }
    const gert::Shape &originShape = shape->GetOriginShape();
    return GetShapeDims(&originShape);
}

bool SameShape(const std::vector<int64_t> &x1Shape, const std::vector<int64_t> &x2Shape)
{
    if (x1Shape.size() != x2Shape.size()) {
        return false;
    }
    for (size_t i = 0; i < x1Shape.size(); ++i) {
        if (x1Shape[i] != x2Shape[i]) {
            return false;
        }
    }
    return true;
}

uint64_t ShapeSize(const std::vector<int64_t> &shape)
{
    uint64_t total = 1;
    for (int64_t dim : shape) {
        total *= static_cast<uint64_t>(dim);
    }
    return total;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const int32_t numCores = platform.GetCoreNumAiv();

    const gert::Tensor *tensorX1 = context->GetRequiredInputTensor(0);
    if (tensorX1 == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType dtypeX1 = tensorX1->GetDataType();
    uint32_t DT_X1 = static_cast<uint32_t>(dtypeX1);
    ASCENDC_TPL_SEL_PARAM(context, DT_X1);

    const gert::StorageShape *x1StorageShape = context->GetInputShape(0);
    const gert::StorageShape *x2StorageShape = context->GetInputShape(1);
    if (x1StorageShape == nullptr || x2StorageShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const std::vector<int64_t> x1Shape = PreferStorageShape(x1StorageShape);
    const std::vector<int64_t> x2Shape = PreferStorageShape(x2StorageShape);
    if (!SameShape(x1Shape, x2Shape)) {
        return ge::GRAPH_FAILED;
    }

    LessEqualTilingData *tiling = context->GetTilingData<LessEqualTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->total = ShapeSize(x1Shape);

    const uint32_t blockDim = SelectBlockDim(tiling->total, numCores, dtypeX1);
    tiling->baseBlockLength = tiling->total / blockDim;
    tiling->tailBlockNum = tiling->total % blockDim;
    const uint64_t maxBlockLength = tiling->baseBlockLength + (tiling->tailBlockNum > 0 ? 1 : 0);
    tiling->tileLength = SelectTileLength(ubSize, dtypeX1, maxBlockLength);
    tiling->lastTileLength = 0;
    if (maxBlockLength != 0) {
        tiling->tileNum = static_cast<uint32_t>((maxBlockLength + tiling->tileLength - 1) / tiling->tileLength);
        const uint32_t tail = static_cast<uint32_t>(maxBlockLength % tiling->tileLength);
        tiling->lastTileLength = (tail == 0) ? tiling->tileLength : tail;
    } else {
        tiling->tileNum = 0;
        tiling->lastTileLength = 0;
    }
    tiling->bufferNum = LESS_EQUAL_BUFFER_NUM;
    tiling->inputBufferBytes = AlignUp32(tiling->tileLength * GetTypeSize(dtypeX1));
    tiling->outputBufferBytes = AlignUp32(tiling->tileLength * GetTypeSize(ge::DT_BOOL));

    context->SetBlockDim(blockDim);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x1ShapeRaw = context->GetInputShape(0);
    const gert::Shape *x2ShapeRaw = context->GetInputShape(1);
    gert::Shape *yShape = context->GetOutputShape(0);
    if (x1ShapeRaw == nullptr || x2ShapeRaw == nullptr || yShape == nullptr) {
        return GRAPH_FAILED;
    }

    const std::vector<int64_t> x1Shape = GetShapeDims(x1ShapeRaw);
    const std::vector<int64_t> x2Shape = GetShapeDims(x2ShapeRaw);
    if (!SameShape(x1Shape, x2Shape)) {
        return GRAPH_FAILED;
    }

    yShape->SetDimNum(0);
    if (!x1ShapeRaw->IsScalar()) {
        for (int64_t dim : x1Shape) {
            yShape->AppendDim(dim);
        }
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
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

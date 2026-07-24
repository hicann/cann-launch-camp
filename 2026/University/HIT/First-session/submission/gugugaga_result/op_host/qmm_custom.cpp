#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include "../op_kernel/qmm_custom_tiling.h"

namespace {
constexpr uint32_t SMALL_BASE_M = 16;
constexpr uint32_t SMALL_BASE_N = 128;
constexpr uint32_t MEDIUM_BASE_M = 64;
constexpr uint32_t LARGE_BASE_M = 128;
constexpr uint32_t LARGE_BASE_N = 256;
constexpr size_t USER_WORKSPACE_SIZE = 32;

inline uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1U) / divisor;
}

inline uint32_t AlignUp(uint32_t value, uint32_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}

struct MatmulPlan {
    uint32_t blockDim;
    uint32_t singleM;
    uint32_t singleN;
    uint32_t baseM;
    uint32_t baseN;
};

// Small-M inference is most efficient when every core owns the full M axis and
// the N axis alone is distributed.  This also avoids repeatedly loading the
// same FRACTAL_NZ weight block for several tiny M partitions.
MatmulPlan SelectSmallMPlan(uint32_t m, uint32_t n, uint32_t maxCoreNum)
{
    const uint32_t baseM = (m <= 1U) ? SMALL_BASE_M : MEDIUM_BASE_M;
    const uint32_t baseN = (m <= 1U) ? SMALL_BASE_N : LARGE_BASE_N;
    const uint32_t maxNBlocks = CeilDiv(n, baseN);
    const uint32_t requestedBlocks = std::max<uint32_t>(1U, std::min(maxCoreNum, maxNBlocks));
    const uint32_t singleN = AlignUp(CeilDiv(n, requestedBlocks), baseN);

    MatmulPlan result {};
    result.baseM = baseM;
    result.baseN = baseN;
    result.singleM = AlignUp(m, 16U);
    result.singleN = singleN;
    result.blockDim = CeilDiv(n, singleN);
    return result;
}

// For large matrices, first use as many AICs as possible, then choose the M/N
// grid which minimizes repeated reads of A and B.  For an mBlocks*nBlocks grid,
// A is read nBlocks times and B is read mBlocks times.
MatmulPlan SelectLargeMPlan(uint32_t m, uint32_t n, uint32_t maxCoreNum)
{
    constexpr uint32_t baseM = LARGE_BASE_M;
    constexpr uint32_t baseN = LARGE_BASE_N;

    uint32_t bestBlocks = 1U;
    uint32_t bestMBlocks = 1U;
    uint32_t bestNBlocks = 1U;
    uint64_t bestTraffic = std::numeric_limits<uint64_t>::max();
    uint64_t bestPadding = std::numeric_limits<uint64_t>::max();
    uint64_t bestTileImbalance = std::numeric_limits<uint64_t>::max();

    const uint32_t maxMBlocks = std::min(maxCoreNum, CeilDiv(m, baseM));
    const uint32_t maxNBlocks = std::min(maxCoreNum, CeilDiv(n, baseN));

    for (uint32_t requestedMBlocks = 1U; requestedMBlocks <= maxMBlocks; ++requestedMBlocks) {
        const uint32_t singleM = AlignUp(CeilDiv(m, requestedMBlocks), baseM);
        const uint32_t actualMBlocks = CeilDiv(m, singleM);
        for (uint32_t requestedNBlocks = 1U; requestedNBlocks <= maxNBlocks; ++requestedNBlocks) {
            const uint32_t singleN = AlignUp(CeilDiv(n, requestedNBlocks), baseN);
            const uint32_t actualNBlocks = CeilDiv(n, singleN);
            const uint32_t blocks = actualMBlocks * actualNBlocks;
            if (blocks == 0U || blocks > maxCoreNum) {
                continue;
            }

            const uint64_t traffic = static_cast<uint64_t>(actualNBlocks) * m +
                                     static_cast<uint64_t>(actualMBlocks) * n;
            const uint64_t paddedM = static_cast<uint64_t>(actualMBlocks) * singleM;
            const uint64_t paddedN = static_cast<uint64_t>(actualNBlocks) * singleN;
            const uint64_t padding = paddedM * paddedN - static_cast<uint64_t>(m) * n;
            const uint64_t mTiles = CeilDiv(singleM, baseM);
            const uint64_t nTiles = CeilDiv(singleN, baseN);
            const uint64_t tileImbalance = mTiles > nTiles ? mTiles - nTiles : nTiles - mTiles;

            const bool better =
                blocks > bestBlocks ||
                (blocks == bestBlocks && traffic < bestTraffic) ||
                (blocks == bestBlocks && traffic == bestTraffic && padding < bestPadding) ||
                (blocks == bestBlocks && traffic == bestTraffic && padding == bestPadding &&
                 tileImbalance < bestTileImbalance);
            if (better) {
                bestBlocks = blocks;
                bestMBlocks = actualMBlocks;
                bestNBlocks = actualNBlocks;
                bestTraffic = traffic;
                bestPadding = padding;
                bestTileImbalance = tileImbalance;
            }
        }
    }

    MatmulPlan result {};
    result.baseM = baseM;
    result.baseN = baseN;
    result.singleM = AlignUp(CeilDiv(m, bestMBlocks), baseM);
    result.singleN = AlignUp(CeilDiv(n, bestNBlocks), baseN);
    result.blockDim = CeilDiv(m, result.singleM) * CeilDiv(n, result.singleN);
    return result;
}

MatmulPlan SelectPlan(uint32_t m, uint32_t n, uint32_t maxCoreNum)
{
    maxCoreNum = std::max<uint32_t>(1U, maxCoreNum);
    return m <= 64U ? SelectSmallMPlan(m, n, maxCoreNum)
                    : SelectLargeMPlan(m, n, maxCoreNum);
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    const gert::Tensor *tensorX1 = context->GetRequiredInputTensor(0);
    const gert::Tensor *tensorX2 = context->GetRequiredInputTensor(1);
    const gert::Tensor *tensorPertokenScale = context->GetOptionalInputTensor(3);
    if (tensorX1 == nullptr || tensorX2 == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &x1Shape = tensorX1->GetStorageShape();
    const gert::Shape &x2Shape = tensorX2->GetStorageShape();
    if (x1Shape.GetDimNum() != 2U || x2Shape.GetDimNum() != 2U) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t m = static_cast<uint32_t>(x1Shape.GetDim(0));
    const uint32_t k = static_cast<uint32_t>(x1Shape.GetDim(1));
    const uint32_t n = static_cast<uint32_t>(x2Shape.GetDim(1));
    if (m == 0U || n == 0U || k == 0U) {
        return ge::GRAPH_FAILED;
    }

    const bool isPertoken = tensorPertokenScale != nullptr;
    const uint32_t maxCoreNum =
        static_cast<uint32_t>(std::max<uint32_t>(1U, platform.GetCoreNumAic()));
    const MatmulPlan plan = SelectPlan(m, n, maxCoreNum);

    QmmCustomTilingData *tilingData = context->GetTilingData<QmmCustomTilingData>();
    if (tilingData == nullptr) {
        return ge::GRAPH_FAILED;
    }

    matmul_tiling::MultiCoreMatmulTiling tilingApi(platform);
    tilingApi.SetDim(static_cast<int32_t>(plan.blockDim));
    tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                       matmul_tiling::DataType::DT_INT8, false);
    tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ,
                       matmul_tiling::DataType::DT_INT8, false);
    // INT32 mode is written straight from L0C to GM.  Only the BF16 path needs
    // the INT32 result in VECIN for dequantization.
    tilingApi.SetCType(isPertoken ? matmul_tiling::TPosition::VECIN
                                  : matmul_tiling::TPosition::GM,
                       matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    tilingApi.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                          matmul_tiling::DataType::DT_INT32);
    tilingApi.SetOrgShape(static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k));
    tilingApi.SetShape(static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k));
    tilingApi.SetSingleShape(static_cast<int32_t>(plan.singleM),
                             static_cast<int32_t>(plan.singleN), static_cast<int32_t>(k));
    tilingApi.SetBias(false);
    tilingApi.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTM);
    if (tilingApi.SetFixSplit(static_cast<int32_t>(plan.baseM),
                              static_cast<int32_t>(plan.baseN), -1) == -1) {
        return ge::GRAPH_FAILED;
    }
    // The large BF16 path intentionally allocates a full 128x256 INT32 result
    // tile in VECIN.  Do not artificially shrink the UB visible to Matmul.
    tilingApi.SetBufferSpace(-1, -1, -1);

    if (tilingApi.GetTiling(tilingData->cubeTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }
    tilingData->cubeTilingData.stepM = 1;
    tilingData->cubeTilingData.stepN = 1;
    tilingData->isPertoken = isPertoken ? 1U : 0U;
    tilingData->workspaceSize = static_cast<uint32_t>(USER_WORKSPACE_SIZE);

    context->SetBlockDim(plan.blockDim);
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = static_cast<size_t>(platform.GetLibApiWorkSpaceSize()) + USER_WORKSPACE_SIZE;
    context->GetRawTilingData()->SetDataSize(sizeof(QmmCustomTilingData));
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x1Shape = context->GetInputShape(0);
    const gert::Shape *x2Shape = context->GetInputShape(1);
    gert::Shape *yShape = context->GetOutputShape(0);
    if (x1Shape == nullptr || x2Shape == nullptr || yShape == nullptr ||
        x1Shape->GetDimNum() != 2U || x2Shape->GetDimNum() != 2U) {
        return GRAPH_FAILED;
    }
    yShape->SetDimNum(2U);
    yShape->SetDim(0U, x1Shape->GetDim(0U));
    yShape->SetDim(1U, x2Shape->GetDim(1U));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const ge::DataType pertokenType = context->GetOptionalInputDataType(3);
    return context->SetOutputDataType(0, pertokenType == ge::DT_UNDEFINED ? ge::DT_INT32 : ge::DT_BF16);
}
}  // namespace ge

namespace ops {
class QmmCustom : public OpDef {
public:
    explicit QmmCustom(const char *name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("pertoken_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(QmmCustom);
}  // namespace ops

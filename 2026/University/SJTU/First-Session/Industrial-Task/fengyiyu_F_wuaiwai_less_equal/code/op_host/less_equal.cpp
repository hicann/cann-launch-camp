/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/less_equal_tiling.h"
#include "../op_kernel/tiling_key_less_equal.h"

namespace {
constexpr uint32_t LE_RAW_MAX_DIM = 64;
constexpr uint32_t LE_TILE_ALIGNMENT = 256;
constexpr uint32_t LE_MAX_TILE = 4096;
constexpr uint32_t LE_GM_ALIGNMENT_ELEMENTS = 32;
constexpr uint64_t LE_UB_RESERVED_BYTES = 2048;

static bool CheckedMultiply(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

static uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

static uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return alignment == 0 ? value : CeilDiv(value, alignment) * alignment;
}

static bool IsSupportedType(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT ||
           dtype == ge::DT_INT32 || dtype == ge::DT_INT8;
}

static uint64_t GetAlignedDim(const gert::Shape &shape, uint32_t outputRank,
                              uint32_t outputAxis)
{
    const uint32_t inputRank = static_cast<uint32_t>(shape.GetDimNum());
    if (outputAxis + inputRank < outputRank) {
        return 1;
    }
    const int64_t dimension = shape.GetDim(outputAxis + inputRank - outputRank);
    return dimension < 0 ? std::numeric_limits<uint64_t>::max()
                         : static_cast<uint64_t>(dimension);
}

static bool BuildBroadcastShape(const gert::Shape &x1Shape, const gert::Shape &x2Shape,
                                gert::Shape *outputShape,
                                uint64_t rawOutput[LE_RAW_MAX_DIM],
                                uint32_t &outputRank, uint64_t &totalElements)
{
    const uint32_t x1Rank = static_cast<uint32_t>(x1Shape.GetDimNum());
    const uint32_t x2Rank = static_cast<uint32_t>(x2Shape.GetDimNum());
    outputRank = std::max(x1Rank, x2Rank);
    if (outputRank > LE_RAW_MAX_DIM) {
        return false;
    }
    if (outputShape != nullptr) {
        outputShape->SetDimNum(outputRank);
    }

    totalElements = 1;
    for (uint32_t axis = 0; axis < outputRank; ++axis) {
        const uint64_t x1Dim = GetAlignedDim(x1Shape, outputRank, axis);
        const uint64_t x2Dim = GetAlignedDim(x2Shape, outputRank, axis);
        if (x1Dim == std::numeric_limits<uint64_t>::max() ||
            x2Dim == std::numeric_limits<uint64_t>::max() ||
            (x1Dim != x2Dim && x1Dim != 1 && x2Dim != 1)) {
            return false;
        }
        const uint64_t outputDim = x1Dim == 1 ? x2Dim : x1Dim;
        rawOutput[axis] = outputDim;
        if (!CheckedMultiply(totalElements, outputDim, totalElements) ||
            outputDim > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return false;
        }
        if (outputShape != nullptr) {
            outputShape->SetDim(axis, static_cast<int64_t>(outputDim));
        }
    }
    return true;
}

static bool BuildBroadcastStrides(const gert::Shape &inputShape,
                                  const uint64_t rawOutput[LE_RAW_MAX_DIM],
                                  uint32_t outputRank,
                                  uint64_t strides[LE_RAW_MAX_DIM])
{
    const uint32_t inputRank = static_cast<uint32_t>(inputShape.GetDimNum());
    if (inputRank > outputRank || outputRank > LE_RAW_MAX_DIM) {
        return false;
    }
    for (uint32_t axis = 0; axis < outputRank; ++axis) {
        strides[axis] = 0;
    }

    uint64_t nativeStride = 1;
    for (uint32_t reverse = inputRank; reverse > 0; --reverse) {
        const uint32_t inputAxis = reverse - 1;
        const uint32_t outputAxis = outputRank - inputRank + inputAxis;
        const int64_t signedDim = inputShape.GetDim(inputAxis);
        if (signedDim < 0) {
            return false;
        }
        const uint64_t inputDim = static_cast<uint64_t>(signedDim);
        strides[outputAxis] = inputDim == 1 && rawOutput[outputAxis] != 1
                                  ? 0
                                  : nativeStride;
        if (!CheckedMultiply(nativeStride, inputDim, nativeStride)) {
            return false;
        }
    }
    return true;
}

static bool IsSameShape(const gert::Shape &lhs, const gert::Shape &rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t axis = 0; axis < lhs.GetDimNum(); ++axis) {
        if (lhs.GetDim(axis) != rhs.GetDim(axis)) {
            return false;
        }
    }
    return true;
}

static bool CollapseBroadcastDimensions(
    const uint64_t rawShape[LE_RAW_MAX_DIM],
    const uint64_t rawX1Stride[LE_RAW_MAX_DIM],
    const uint64_t rawX2Stride[LE_RAW_MAX_DIM], uint32_t rawRank,
    LessEqualTilingData &tiling)
{
    if (rawRank == 0) {
        tiling.ndim = 0;
        return true;
    }

    uint64_t shape[LE_RAW_MAX_DIM] = {};
    uint64_t x1Stride[LE_RAW_MAX_DIM] = {};
    uint64_t x2Stride[LE_RAW_MAX_DIM] = {};
    uint32_t collapsedRank = 1;
    shape[0] = rawShape[0];
    x1Stride[0] = rawX1Stride[0];
    x2Stride[0] = rawX2Stride[0];

    for (uint32_t axis = 1; axis < rawRank; ++axis) {
        const uint32_t outer = collapsedRank - 1;
        const uint64_t innerExtent = rawShape[axis];
        const bool bothBroadcast = x1Stride[outer] == 0 && rawX1Stride[axis] == 0 &&
                                   x2Stride[outer] == 0 && rawX2Stride[axis] == 0;
        const bool bothContiguous =
            x1Stride[outer] != 0 && rawX1Stride[axis] != 0 &&
            x1Stride[outer] == rawX1Stride[axis] * innerExtent &&
            x2Stride[outer] != 0 && rawX2Stride[axis] != 0 &&
            x2Stride[outer] == rawX2Stride[axis] * innerExtent;
        if (bothBroadcast || bothContiguous) {
            if (!CheckedMultiply(shape[outer], innerExtent, shape[outer])) {
                return false;
            }
            x1Stride[outer] = rawX1Stride[axis];
            x2Stride[outer] = rawX2Stride[axis];
        } else {
            shape[collapsedRank] = innerExtent;
            x1Stride[collapsedRank] = rawX1Stride[axis];
            x2Stride[collapsedRank] = rawX2Stride[axis];
            ++collapsedRank;
        }
    }

    if (collapsedRank > LE_MAX_DIM) {
        return false;
    }
    const uint64_t uint32Max = std::numeric_limits<uint32_t>::max();
    tiling.ndim = collapsedRank;
    for (uint32_t axis = 0; axis < collapsedRank; ++axis) {
        if (shape[axis] > uint32Max || x1Stride[axis] > uint32Max ||
            x2Stride[axis] > uint32Max) {
            return false;
        }
        tiling.outShape[axis] = static_cast<uint32_t>(shape[axis]);
        tiling.x1Stride[axis] = static_cast<uint32_t>(x1Stride[axis]);
        tiling.x2Stride[axis] = static_cast<uint32_t>(x2Stride[axis]);
    }
    return true;
}

static uint32_t ChooseTileSize(uint64_t perCoreWork)
{
    const uint64_t work = std::max<uint64_t>(perCoreWork, LE_TILE_ALIGNMENT);
    return static_cast<uint32_t>(
        std::min<uint64_t>(LE_MAX_TILE, AlignUp(work, LE_TILE_ALIGNMENT)));
}

static uint64_t RequiredUbBytes(ge::DataType dtype, uint32_t tile)
{
    const uint32_t inputBytes = static_cast<uint32_t>(ge::GetSizeByDataType(dtype));
    const uint64_t queueBytes = static_cast<uint64_t>(tile) * (4ULL * inputBytes + 2ULL);
    const uint64_t commonComputeBytes = static_cast<uint64_t>(tile) * 6ULL;
    const uint64_t dtypeComputeBytes =
        (dtype == ge::DT_INT8 || dtype == ge::DT_INT32)
            ? static_cast<uint64_t>(tile) * 4ULL
            : 0ULL;
    const uint64_t maskBytes = AlignUp(CeilDiv(tile, 8), 32);
    return queueBytes + commonComputeBytes + dtypeComputeBytes + maskBytes +
           LE_UB_RESERVED_BYTES;
}

static bool ConfigureAdaptiveTile(ge::DataType dtype, uint64_t perCoreWork,
                                  uint64_t ubSize, uint32_t &tile)
{
    tile = ChooseTileSize(perCoreWork);
    while (tile > LE_TILE_ALIGNMENT && RequiredUbBytes(dtype, tile) > ubSize) {
        tile -= LE_TILE_ALIGNMENT;
    }
    return RequiredUbBytes(dtype, tile) <= ubSize;
}

static void ClearTiling(LessEqualTilingData &tiling)
{
    tiling = {};
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Tensor *tensorX1 = context->GetRequiredInputTensor(0);
    const gert::Tensor *tensorX2 = context->GetRequiredInputTensor(1);
    const auto *x1Storage = context->GetInputShape(0);
    const auto *x2Storage = context->GetInputShape(1);
    LessEqualTilingData *tiling = context->GetTilingData<LessEqualTilingData>();
    if (tensorX1 == nullptr || tensorX2 == nullptr || x1Storage == nullptr ||
        x2Storage == nullptr || tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtypeX1 = tensorX1->GetDataType();
    const ge::DataType dtypeX2 = tensorX2->GetDataType();
    if (dtypeX1 != dtypeX2 || !IsSupportedType(dtypeX1)) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t DT_X1 = static_cast<uint32_t>(dtypeX1);
    ASCENDC_TPL_SEL_PARAM(context, DT_X1);

    const gert::Shape &x1Shape = x1Storage->GetStorageShape();
    const gert::Shape &x2Shape = x2Storage->GetStorageShape();
    uint64_t rawShape[LE_RAW_MAX_DIM] = {};
    uint64_t rawX1Stride[LE_RAW_MAX_DIM] = {};
    uint64_t rawX2Stride[LE_RAW_MAX_DIM] = {};
    uint32_t outputRank = 0;
    uint64_t totalElements = 0;
    if (!BuildBroadcastShape(x1Shape, x2Shape, nullptr, rawShape, outputRank,
                             totalElements) ||
        !BuildBroadcastStrides(x1Shape, rawShape, outputRank, rawX1Stride) ||
        !BuildBroadcastStrides(x2Shape, rawShape, outputRank, rawX2Stride) ||
        totalElements > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    ClearTiling(*tiling);
    tiling->totalElements = static_cast<uint32_t>(totalElements);
    if (totalElements == 0) {
        tiling->mode = LE_MODE_FAST;
        tiling->tileSize = LE_TILE_ALIGNMENT;
        tiling->blockDim = 1;
        context->SetBlockDim(1);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (currentWorkspace == nullptr) {
            return ge::GRAPH_FAILED;
        }
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t platformCores = platform.GetCoreNumAiv();
    const uint32_t coreCount = platformCores > 0 ? static_cast<uint32_t>(platformCores) : 1;
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    uint64_t perCoreWork = 0;
    if (IsSameShape(x1Shape, x2Shape)) {
        tiling->mode = LE_MODE_FAST;
        const uint64_t target = CeilDiv(totalElements, coreCount);
        const uint64_t alignedPerCore = AlignUp(target, LE_GM_ALIGNMENT_ELEMENTS);
        if (alignedPerCore > std::numeric_limits<uint32_t>::max()) {
            return ge::GRAPH_FAILED;
        }
        tiling->perCore = static_cast<uint32_t>(alignedPerCore);
        tiling->blockDim = static_cast<uint32_t>(
            CeilDiv(totalElements, tiling->perCore));
        perCoreWork = std::min<uint64_t>(tiling->perCore, totalElements);
    } else {
        tiling->mode = LE_MODE_BCAST;
        if (!CollapseBroadcastDimensions(rawShape, rawX1Stride, rawX2Stride,
                                         outputRank, *tiling) ||
            tiling->ndim == 0) {
            return ge::GRAPH_FAILED;
        }
        tiling->lastDimLen = tiling->outShape[tiling->ndim - 1];
        uint64_t totalRows = 1;
        for (uint32_t axis = 0; axis + 1 < tiling->ndim; ++axis) {
            if (!CheckedMultiply(totalRows, tiling->outShape[axis], totalRows)) {
                return ge::GRAPH_FAILED;
            }
        }
        if (totalRows > std::numeric_limits<uint32_t>::max()) {
            return ge::GRAPH_FAILED;
        }
        tiling->totalRows = static_cast<uint32_t>(totalRows);
        tiling->perCore = static_cast<uint32_t>(CeilDiv(totalRows, coreCount));
        tiling->blockDim = static_cast<uint32_t>(CeilDiv(totalRows, tiling->perCore));
        perCoreWork = std::min<uint64_t>(tiling->lastDimLen,
                                         static_cast<uint64_t>(tiling->perCore) *
                                             tiling->lastDimLen);
    }

    uint32_t tile = 0;
    if (!ConfigureAdaptiveTile(dtypeX1, perCoreWork, ubSize, tile)) {
        return ge::GRAPH_FAILED;
    }
    tiling->tileSize = tile;
    context->SetBlockDim(tiling->blockDim);
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
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape *x1Shape = context->GetInputShape(0);
    const gert::Shape *x2Shape = context->GetInputShape(1);
    gert::Shape *outputShape = context->GetOutputShape(0);
    uint64_t rawShape[LE_RAW_MAX_DIM] = {};
    uint32_t outputRank = 0;
    uint64_t totalElements = 0;
    if (x1Shape == nullptr || x2Shape == nullptr || outputShape == nullptr ||
        !BuildBroadcastShape(*x1Shape, *x2Shape, outputShape, rawShape,
                             outputRank, totalElements)) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
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

#pragma once

#include <cstdint>

#define LE_MAX_DIM 16

struct LessEqualTilingData {
    uint32_t mode;
    uint32_t totalElements;
    uint32_t tileSize;
    uint32_t blockDim;
    uint32_t perCore;
    uint32_t ndim;
    uint32_t lastDimLen;
    uint32_t totalRows;
    uint32_t outShape[LE_MAX_DIM];
    int32_t  x1Stride[LE_MAX_DIM];
    int32_t  x2Stride[LE_MAX_DIM];
};

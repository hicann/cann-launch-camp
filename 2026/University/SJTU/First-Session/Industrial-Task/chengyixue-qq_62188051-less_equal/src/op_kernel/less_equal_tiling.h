// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

#define LE_MAX_DIM 16

struct LessEqualTilingData {
    uint32_t mode;          // 0 = FAST (no broadcast), 1 = BCAST
    uint32_t totalElements; // FAST: total output elems (== each input's elem count)
    uint32_t tileSize;      // elements per tile (multiple of 128)
    uint32_t blockDim;      // launched AIV cores
    uint32_t perCore;       // FAST: elems per core (multiple of 256); BCAST: rows per core
    uint32_t ndim;          // BCAST: number of collapsed output dims (<= LE_MAX_DIM)
    uint32_t lastDimLen;    // BCAST: L = outShape[ndim-1]
    uint32_t totalRows;     // BCAST: product(outShape[0..ndim-2]) (=1 if ndim==1)
    uint32_t outShape[LE_MAX_DIM];
    int32_t  x1Stride[LE_MAX_DIM]; // broadcast strides into raw x1 (0 on broadcast dims)
    int32_t  x2Stride[LE_MAX_DIM]; // broadcast strides into raw x2
    uint32_t bufferNum;            // physical buffer count: 1=single-tile, 2=multi-tile
};

#ifndef FAST_GELU_TILING_H
#define FAST_GELU_TILING_H

#include <cstdint>

struct FastGeluTilingData {
    uint32_t length;
    uint32_t tileSize;
    uint32_t blockDim;
};

#endif  // FAST_GELU_TILING_H

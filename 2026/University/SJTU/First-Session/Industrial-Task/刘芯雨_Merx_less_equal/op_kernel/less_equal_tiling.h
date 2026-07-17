// Tiling data shared by host and kernel.
#pragma once

#include <cstdint>

constexpr uint32_t LESS_EQUAL_MAX_DIMS = 16;

struct LessEqualTilingData {
    uint32_t outputLength;
    uint32_t blockLength;
    uint32_t rank;
    uint32_t noBroadcast;
    uint32_t outShape[LESS_EQUAL_MAX_DIMS];
    uint32_t outStride[LESS_EQUAL_MAX_DIMS];
    uint32_t x1Stride[LESS_EQUAL_MAX_DIMS];
    uint32_t x2Stride[LESS_EQUAL_MAX_DIMS];
};

// Host/Kernel shared tiling data.
#pragma once

#include <cstdint>

constexpr uint32_t LESS_EQUAL_MAX_DIMS = 8;

enum LessEqualMode : uint32_t {
    LESS_EQUAL_NO_BROADCAST = 0,
    LESS_EQUAL_X1_SCALAR = 1,
    LESS_EQUAL_X2_SCALAR = 2,
    LESS_EQUAL_GENERAL_BROADCAST = 3,
};

struct LessEqualTilingData {
    uint64_t outputLength;
    uint64_t x1Length;
    uint64_t x2Length;
    uint64_t outputShape[LESS_EQUAL_MAX_DIMS];
    uint64_t x1Stride[LESS_EQUAL_MAX_DIMS];
    uint64_t x2Stride[LESS_EQUAL_MAX_DIMS];
    uint32_t rank;
    uint32_t mode;
};

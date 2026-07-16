// Tiling data for LessEqual.
#pragma once

#include <cstdint>

constexpr uint32_t LESS_EQUAL_MAX_DIMS = 16;

enum LessEqualMode : uint32_t {
    LESS_EQUAL_MODE_FLAT = 0,
    LESS_EQUAL_MODE_X1_SCALAR = 1,
    LESS_EQUAL_MODE_X2_SCALAR = 2,
    LESS_EQUAL_MODE_GENERAL = 3,
    LESS_EQUAL_MODE_X1_ROW = 4,
    LESS_EQUAL_MODE_X2_ROW = 5,
    LESS_EQUAL_MODE_X1_REPEAT = 6,
    LESS_EQUAL_MODE_X2_REPEAT = 7,
};

// Shapes are right-aligned. A zero stride marks a broadcast dimension.
struct LessEqualTilingData {
    uint64_t totalLength;
    uint64_t outputDims[LESS_EQUAL_MAX_DIMS];
    uint64_t x1Strides[LESS_EQUAL_MAX_DIMS];
    uint64_t x2Strides[LESS_EQUAL_MAX_DIMS];
    uint32_t rank;
    uint32_t blockNum;
    uint32_t tileLength;
    uint32_t mode;
};

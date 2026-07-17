#pragma once

#include <cstdint>

constexpr uint32_t LESS_EQUAL_MAX_RANK = 16;

enum LessEqualMode : uint32_t {
    LESS_EQUAL_MODE_LINEAR = 0,
    LESS_EQUAL_MODE_ROW = 1,
    LESS_EQUAL_MODE_REUSE_X1 = 2,
    LESS_EQUAL_MODE_REUSE_X2 = 3
};

struct LessEqualTilingData {
    uint32_t totalLength;
    uint32_t innerLength;
    uint32_t outerLength;
    uint32_t workPerCore;
    uint32_t tileLength;
    uint32_t rank;
    uint32_t mode;
    uint32_t usedCoreNum;
    uint32_t outShape[LESS_EQUAL_MAX_RANK];
    uint32_t x1Stride[LESS_EQUAL_MAX_RANK];
    uint32_t x2Stride[LESS_EQUAL_MAX_RANK];
};

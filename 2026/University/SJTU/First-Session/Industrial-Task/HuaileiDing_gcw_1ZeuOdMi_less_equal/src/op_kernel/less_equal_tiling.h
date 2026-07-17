#pragma once

#include <cstdint>

constexpr uint32_t LESS_EQUAL_MAX_DIMS = 16;

struct LessEqualTilingData {
    uint32_t length;
    uint32_t blockLength;
    uint32_t rank;
    uint32_t isBroadcast;
    uint32_t scalarBroadcast;
    uint32_t outDims[LESS_EQUAL_MAX_DIMS];
    uint32_t x1Strides[LESS_EQUAL_MAX_DIMS];
    uint32_t x2Strides[LESS_EQUAL_MAX_DIMS];
};

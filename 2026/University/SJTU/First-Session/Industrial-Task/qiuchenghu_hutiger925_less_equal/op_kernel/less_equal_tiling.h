// LessEqual tiling data shared by host and kernel.
#pragma once

#include <cstdint>

constexpr uint32_t LESS_EQUAL_MAX_DIMS = 32;

enum LessEqualBroadcastMode : uint32_t {
    LESS_EQUAL_SAME_SHAPE = 0,
    LESS_EQUAL_X1_SCALAR = 1,
    LESS_EQUAL_X2_SCALAR = 2,
    LESS_EQUAL_GENERAL_BROADCAST = 3,
};

struct LessEqualTilingData {
    uint64_t totalLength;
    uint64_t x1Length;
    uint64_t x2Length;
    uint64_t blockLength;
    uint64_t innerLength;

    uint32_t rank;
    uint32_t broadcastMode;
    uint32_t tileLength;
    uint32_t x1InnerContiguous;
    uint32_t x2InnerContiguous;
    uint32_t reserved;

    uint64_t outputShape[LESS_EQUAL_MAX_DIMS];
    uint64_t x1Stride[LESS_EQUAL_MAX_DIMS];
    uint64_t x2Stride[LESS_EQUAL_MAX_DIMS];
};

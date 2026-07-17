// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct LessEqualTilingData {
    uint64_t length;
    uint32_t rank;
    uint32_t tileLength;
    uint64_t outShape[25];
    uint64_t x1Stride[25];
    uint64_t x2Stride[25];
};

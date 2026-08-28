// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct ClipByValueTilingData {
    uint64_t totalLength;
    uint64_t lengthPerCore;
    uint32_t blockDim;
    uint32_t tileLength;
    uint32_t alignElems;
    uint32_t minIsScalar;
    uint32_t maxIsScalar;
};

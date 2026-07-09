// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct FastGeluTilingData {
    uint64_t totalLength;
    uint64_t blockLength;
    uint32_t tileLength;
    uint32_t coreNum;
};

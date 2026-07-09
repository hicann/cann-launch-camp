// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct FastGeluTilingData {
    uint64_t totalLength;

    uint32_t usedCoreNum;

    uint32_t alignElemNum;

    uint64_t baseBlockNum;

    uint32_t tailBlockNum;

    uint32_t tileLength;

    // 1 表示小 shape 轻量模式
    // 0 表示正常多核模式
    uint32_t smallMode;
};
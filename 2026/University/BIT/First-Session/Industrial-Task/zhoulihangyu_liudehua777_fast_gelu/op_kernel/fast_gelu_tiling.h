// Tiling结构体定义的头文件
#pragma once
#include <cstdint>

struct FastGeluTilingData {
    uint32_t totalLength;
    uint32_t alignNum;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;
};
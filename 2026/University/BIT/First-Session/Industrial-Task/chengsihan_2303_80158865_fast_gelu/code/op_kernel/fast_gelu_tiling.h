// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct FastGeluTilingData {
    uint32_t totalLength;     // 当前 AI Core 负责的总元素个数
    uint32_t tileNum;         // 当前 AI Core 内部需要循环处理的 Tiling 块总数
    uint32_t tileLength;      // 每个标准 Tiling 块的元素个数
    uint32_t lastTileLength;  // 最后一个不完整 Tiling 块（尾块）的实际元素个数
};
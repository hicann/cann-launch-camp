// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct FastGeluTilingData {
    uint32_t bigCoreDataNum;
    uint32_t smallCoreDataNum;
    uint32_t tileDataNum;
    uint32_t finalBigTileNum;
    uint32_t finalSmallTileNum;
    uint32_t bigTailDataNum;
    uint32_t smallTailDataNum;
    uint32_t tailBlockNum;
    uint32_t totalLength;    // 总元素数：小 kernel 单发时的搬运/计算长度
    uint32_t isSmallShape;   // 1 = 单核单发小 kernel；0 = 多核流水大 kernel
};

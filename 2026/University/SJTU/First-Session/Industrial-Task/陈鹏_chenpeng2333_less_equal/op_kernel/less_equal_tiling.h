// ============================================================================
//  Tiling结构体定义的头文件
// ============================================================================
#pragma once

#include <cstdint>

struct LessEqualTilingData {
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lasttileLength;
    uint32_t x1Length;    // 0=无需广播, >0=x1实际元素个数
    uint32_t x2Length;    // 0=无需广播, >0=x2实际元素个数
    uint32_t totalLength;  // 输出总元素数，用于尾部边界控制
};
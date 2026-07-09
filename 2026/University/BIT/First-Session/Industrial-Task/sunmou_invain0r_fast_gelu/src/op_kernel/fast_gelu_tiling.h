// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct FastGeluTilingData {
    uint32_t length;       // 总元素个数
    uint32_t blockNum;     // 使用的AI Core数量
    uint32_t tileDataNum;  // 每个完整tile的元素数（UB能容纳的最大值）
};
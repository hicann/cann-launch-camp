// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct FastGeluTilingData {
    uint32_t length;       // 输入/输出张量的元素总数（所有维度展平后的element个数）
    uint32_t tileDataNum;  // 单次UB搬运/计算的元素个数，由Host侧依据实际UB大小动态计算
};

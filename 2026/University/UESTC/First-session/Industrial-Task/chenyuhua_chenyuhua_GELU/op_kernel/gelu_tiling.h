#pragma once
#include <cstdint>

struct GeluTilingData {
    uint32_t totalLength;   // 输入总元素数
    uint32_t blockLength;   // 每个 core 处理的元素数（已对齐）
    uint32_t tileLength;    // 每个 tile 处理的元素数（固定值）
};
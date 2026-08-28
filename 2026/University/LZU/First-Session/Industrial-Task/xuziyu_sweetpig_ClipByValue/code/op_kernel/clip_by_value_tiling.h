// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct ClipByValueTilingData {
    uint32_t length;
    uint32_t blockLength;
    uint32_t tileLength;
    uint8_t  isScalarMin;
    uint8_t  isScalarMax;
};
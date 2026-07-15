// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct GeluTilingData {
    int64_t totalNum = 0;
    int64_t blockFactor = 0;
    int64_t ubFactor = 0;
    int64_t erfTmpBytes = 0;
    int32_t reserved = 0;
};

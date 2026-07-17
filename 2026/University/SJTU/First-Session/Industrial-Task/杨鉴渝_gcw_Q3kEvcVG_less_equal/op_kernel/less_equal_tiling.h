// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct LessEqualTilingData {
    uint32_t blockDim;       // AI Core总数
    uint32_t totalLength;    // 输出张量总元素数
    uint32_t blockLength;    // 每核处理的输出元素数
    uint32_t tileNum;        // 每核tile数
    uint32_t dtype;          // 0=f16, 1=f32, 2=i32, 3=i8
    uint32_t length;         // 总元素数（兼容旧接口）
};

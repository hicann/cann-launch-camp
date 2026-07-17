// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

constexpr int32_t MAX_DIMS = 8;

struct LessEqualTilingData {
    uint64_t totalLength;       // 输出总元素数
    int32_t x1Shape[MAX_DIMS];  // x1的形状（广播对齐后）
    int32_t x2Shape[MAX_DIMS];  // x2的形状（广播对齐后）
    int32_t outShape[MAX_DIMS]; // 输出形状（广播后）
    uint64_t x1Stride[MAX_DIMS];// x1存储stride
    uint64_t x2Stride[MAX_DIMS];// x2存储stride
    int32_t dims;               // 维度数
    int32_t dtype;              // 输入数据类型
    uint32_t blockDim;          // 实际使用的核数
    uint32_t ubChunkSize;       // UB分块大小（元素数）
};

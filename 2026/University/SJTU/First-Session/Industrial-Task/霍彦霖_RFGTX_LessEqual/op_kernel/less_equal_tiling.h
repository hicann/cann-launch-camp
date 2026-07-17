// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// 广播场景下支持的最大（合并后）维度数
constexpr uint32_t LE_MAX_DIM = 8;

struct LessEqualTilingData {
    uint32_t totalLength;   // 广播后输出的总元素个数
    uint32_t mode;          // 0 = 纯逐元素(无广播), 1 = 广播(按行处理)
    uint32_t blockDim;      // 实际启动的核数
    uint32_t rank;          // 合并后维度数 (mode=1 时有效)
    uint32_t lastDim;       // 输出最内层维度大小
    uint32_t numRows;       // total / lastDim, 广播模式下按行分核
    uint32_t x1LastBroad;   // x1 在最内层维度是否为广播(size=1)
    uint32_t x2LastBroad;   // x2 在最内层维度是否为广播(size=1)
    uint32_t outShape[LE_MAX_DIM];  // 合并后每个维度的输出大小
    uint32_t x1Stride[LE_MAX_DIM];  // x1 在各输出维度上的线性步长(广播维为0)
    uint32_t x2Stride[LE_MAX_DIM];  // x2 在各输出维度上的线性步长(广播维为0)
};

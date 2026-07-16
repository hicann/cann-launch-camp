// Tiling 结构体定义（host 与 kernel 的契约）
// 设计：BCAST 不折叠维度（题目维度 <=4D，ndim 不会超 LE_MAX_DIM），
//       广播维的 stride 置 0，这样 kernel 解码坐标时广播维不影响线性下标。
#pragma once

#include <cstdint>

#define LE_MAX_DIM 16

struct LessEqualTilingData {
    uint32_t mode;          // 0 = FAST（两输入同形，无广播）; 1 = BCAST（右对齐广播）; 2 = SCALAR（一个输入为标量）
    uint32_t totalElements; // 输出总元素数
    uint32_t tileSize;      // 每个 tile 的元素数
    uint32_t blockDim;      // 实际启动的 AIV 核数
    uint32_t perCore;       // FAST/SCALAR: 每核元素数(32对齐)；BCAST: 每核行数
    uint32_t ndim;          // BCAST: 输出维度数；SCALAR(mode=2): 1=x1标量, 2=x2标量
    uint32_t lastDimLen;    // BCAST: 最后一维长度 L = outShape[ndim-1]
    uint32_t totalRows;     // BCAST: product(outShape[0..ndim-2])
    uint32_t outShape[LE_MAX_DIM];
    int32_t  x1Stride[LE_MAX_DIM]; // 进入 x1 的广播 stride（广播维为 0）
    int32_t  x2Stride[LE_MAX_DIM]; // 进入 x2 的广播 stride（广播维为 0）
};

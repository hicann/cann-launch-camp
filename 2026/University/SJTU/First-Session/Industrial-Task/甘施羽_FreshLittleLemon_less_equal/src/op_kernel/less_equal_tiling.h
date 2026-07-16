// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// 最大支持维度数（广播场景下按对齐后的维度数处理）
#ifndef LESS_EQUAL_MAX_DIM
#define LESS_EQUAL_MAX_DIM 8
#endif

struct LessEqualTilingData {
    // 通用
    uint32_t totalLength;   // 输出总元素数（广播后）
    uint32_t tileLength;    // 单次处理元素数（UB 分块）
    uint32_t blockDim;      // 使用的核数
    uint32_t needBroadcast; // 0=fast path(等形状), 1=broadcast path

    // 广播场景专用（needBroadcast==1 时有效）
    uint32_t ndim;                       // 对齐后的维度数（<= LESS_EQUAL_MAX_DIM）
    uint32_t shapeOut[LESS_EQUAL_MAX_DIM];  // 广播后各维大小
    uint32_t dimStride[LESS_EQUAL_MAX_DIM]; // 输出各维累积步长（用于平坦索引反推）
    uint32_t strideX1[LESS_EQUAL_MAX_DIM];  // x1 各维广播步长（被广播维=0）
    uint32_t strideX2[LESS_EQUAL_MAX_DIM];  // x2 各维广播步长（被广播维=0）
};

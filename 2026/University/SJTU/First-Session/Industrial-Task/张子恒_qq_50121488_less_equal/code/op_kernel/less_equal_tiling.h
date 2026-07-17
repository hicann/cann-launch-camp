// Host/Kernel 共享的 LessEqual tiling 数据结构。
#pragma once

#include <cstdint>

// 预留 64 维以覆盖高维批次场景；三组 shape/stride 数组仍保持在 1KB 以内。
constexpr uint32_t LESS_EQUAL_MAX_RANK = 64;

struct LessEqualTilingData {
    // 广播后的输出总元素数。
    uint32_t length;

    // 实际启动核数与单次 UB 分块元素数。
    uint32_t blockDim;
    uint32_t tileLength;

    // 广播后的 rank；contiguous 表示 x1/x2 都可按线性地址连续读取。
    uint32_t rank;
    uint32_t contiguous;

    // 两个输入张量的实际元素数，用于设置 GlobalTensor 边界。
    uint32_t x1Numel;
    uint32_t x2Numel;

    // 每个输入在广播后仍保持连续的最大尾部 span，用于按连续段 DMA 搬运。
    uint32_t x1ContiguousSpan;
    uint32_t x2ContiguousSpan;

    // 广播后的输出 shape。
    uint32_t outputShape[LESS_EQUAL_MAX_RANK];

    // 广播 stride：某一维为广播维时 stride 为 0，否则为原输入连续内存 stride。
    uint32_t x1Stride[LESS_EQUAL_MAX_RANK];
    uint32_t x2Stride[LESS_EQUAL_MAX_RANK];
};

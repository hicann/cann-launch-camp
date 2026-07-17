// LessEqual Host/Kernel 之间传递的 Tiling 参数。
// 支持广播机制：通过预计算的 stride 实现从输出索引到输入索引的映射。
#pragma once

#include <cstdint>

// 最大支持维度数
static constexpr uint32_t MAX_NDIM = 8;

struct LessEqualTilingData {
    uint64_t totalLength;   // 输出张量的总元素数（广播后）
    uint64_t blockLength;   // 每个 AIV 核处理的元素数
    uint32_t tileLength;    // 每次搬入 UB 的元素数
    uint32_t ndim;          // 有效维度数
    // 输出形状（广播后），从最内层到最外层
    uint32_t outShape[MAX_NDIM];
    // x1 在各维度上的 stride（用于广播寻址，维度为1时 stride=0）
    uint32_t stride0[MAX_NDIM];
    // x2 在各维度上的 stride（用于广播寻址，维度为1时 stride=0）
    uint32_t stride1[MAX_NDIM];
    uint32_t reserved;      // 保留字段，保证对齐
};

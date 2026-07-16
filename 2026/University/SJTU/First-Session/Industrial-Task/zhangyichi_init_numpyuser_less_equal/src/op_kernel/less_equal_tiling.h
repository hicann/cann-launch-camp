// Host 和 Kernel 共用的 Tiling 数据。
#pragma once

#include <cstdint>

// gert::Shape 在本工程配套的 CANN 版本中最多支持 25 维。
static constexpr uint32_t LESS_EQUAL_MAX_DIMS = 25U;

struct LessEqualTilingData {
    uint64_t totalLength;   // 广播后输出张量的元素个数
    uint64_t blockLength;   // 每个 AI Core 处理的最大连续长度（32B 对齐）
    uint64_t segmentLength; // 广播映射在本段内可化为连续块或标量复用
    uint32_t rank;          // 广播后的维数
    uint32_t blockDim;      // 实际启动的 AI Core 数
    uint32_t tileLength;    // 单次放入 UB 处理的元素个数
    uint32_t noBroadcast;   // 1：两输入形状相同，可直接连续搬运
    uint32_t x1SegmentScalar; // 1：x1 在 segment 内为同一标量
    uint32_t x2SegmentScalar; // 1：x2 在 segment 内为同一标量
    uint32_t segmentOuterRank; // segment 左侧仍需做广播坐标换算的维数

    uint64_t outputShape[LESS_EQUAL_MAX_DIMS];
    uint64_t x1Strides[LESS_EQUAL_MAX_DIMS];
    uint64_t x2Strides[LESS_EQUAL_MAX_DIMS];
};

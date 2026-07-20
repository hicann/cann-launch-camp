// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

constexpr uint32_t ADDCMUL_MAX_DIMS = 16;

struct AddcmulTilingData {
    uint32_t totalLength;       // 输出元素总数
    uint32_t tileLength;        // 单次搬运/计算长度（32B对齐元素数）
    uint32_t formerNum;         // 大核数量
    uint32_t formerLength;      // 大核处理元素数
    uint32_t tailBlockLength;   // 小核处理元素数
    uint32_t isBroadcast;       // 是否需要广播
    uint32_t dimNum;            // 广播后维度数
    uint32_t outShape[ADDCMUL_MAX_DIMS];
    uint32_t inputShape[ADDCMUL_MAX_DIMS];
    uint32_t x1Shape[ADDCMUL_MAX_DIMS];
    uint32_t x2Shape[ADDCMUL_MAX_DIMS];
    uint32_t outStride[ADDCMUL_MAX_DIMS];
    uint32_t inputStride[ADDCMUL_MAX_DIMS];
    uint32_t x1Stride[ADDCMUL_MAX_DIMS];
    uint32_t x2Stride[ADDCMUL_MAX_DIMS];
};

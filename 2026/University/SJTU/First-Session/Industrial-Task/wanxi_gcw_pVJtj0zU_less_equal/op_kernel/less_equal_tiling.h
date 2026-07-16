#pragma once
#include <cstdint>

#define LE_MAX_DIM 16

struct LessEqualTilingData {
    uint64_t totalLen;       // 总元素数
    uint32_t tileLen;        // 每 tile 元素数
    uint32_t blockDim;       // 使用的核心数
    uint32_t perCore;        // 每个核心处理的元素数(fast)或行数(broadcast)
    uint32_t mode;           // 0=fast同形, 1=broadcast
    uint32_t ndim;           // 广播后的维数
    uint32_t lastDimLen;     // 最后一维长度（broadcast模式）
    uint32_t totalRows;      // 总行数（broadcast模式）
    uint32_t outShape[LE_MAX_DIM];
    int32_t  x1Stride[LE_MAX_DIM];
    int32_t  x2Stride[LE_MAX_DIM];
};

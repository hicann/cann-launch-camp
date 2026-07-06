#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H

#include <cstdint>

struct LogSigmoidCustomTilingData {
    uint32_t totalLength;   // 总数据量
    uint32_t blockLength;   // 每个Block处理的数据量
    uint32_t tileLength;    // 每次搬运的数据量
    uint32_t dataType;      // 数据类型: 0=float32, 1=float16, 2=bfloat16
};

#endif  // LOG_SIGMOID_CUSTOM_TILING_H

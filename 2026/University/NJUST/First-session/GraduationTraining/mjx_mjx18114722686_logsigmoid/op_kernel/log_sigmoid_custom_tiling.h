#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H
#include <cstdint>

struct LogSigmoidCustomTilingData {
    uint32_t totalElements;   // 总元素个数
    uint32_t dataTypeSize;    // 数据类型大小: 2=float16/bfloat16, 4=float32
    uint32_t dataType;        // 数据类型: 0=float16, 1=float32, 2=bfloat16
};

#endif // LOG_SIGMOID_CUSTOM_TILING_H

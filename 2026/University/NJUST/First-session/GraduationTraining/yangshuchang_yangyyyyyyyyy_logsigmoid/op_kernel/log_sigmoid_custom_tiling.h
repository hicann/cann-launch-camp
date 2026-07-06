#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H
#include <cstdint>

struct LogSigmoidCustomTilingData {
    // 修改Tiling结构体定义
    uint32_t size;
    uint32_t dataType;  // 0: float16, 1: float32, 2: bfloat16
};

#endif // LOG_SIGMOID_CUSTOM_TILING_H

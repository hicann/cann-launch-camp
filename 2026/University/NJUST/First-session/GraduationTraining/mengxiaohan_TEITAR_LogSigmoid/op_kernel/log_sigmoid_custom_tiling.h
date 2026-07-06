#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H
#include <cstdint>

struct LogSigmoidCustomTilingData {
    uint32_t size;
    uint32_t dtype;  // 0=float16, 1=float32, 2=bfloat16
};

#endif // LOG_SIGMOID_CUSTOM_TILING_H

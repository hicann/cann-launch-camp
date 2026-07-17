#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H
#include <cstdint>

struct LogSigmoidCustomTilingData {
    uint32_t size;
    // 0:float16 half  1:float32 float  2:bfloat16_t
    uint32_t dataType;
};

#endif // LOG_SIGMOID_CUSTOM_TILING_H

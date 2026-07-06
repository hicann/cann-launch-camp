#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H

#include <cstdint>

struct LogSigmoidCustomTilingData {
    uint32_t size;
    uint32_t blockDim;
    uint32_t tileLength;
    uint32_t dataType; // 0: float32, 1: float16, 2: bfloat16
};

#endif // LOG_SIGMOID_CUSTOM_TILING_H

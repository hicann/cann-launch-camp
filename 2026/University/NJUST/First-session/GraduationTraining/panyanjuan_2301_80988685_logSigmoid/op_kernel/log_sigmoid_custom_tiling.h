#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H
#include <cstdint>

struct LogSigmoidCustomTilingData {
    uint32_t elem_total;
    uint32_t dtype_code; // 0 half / 1 float / 2 bfloat16
};
#endif

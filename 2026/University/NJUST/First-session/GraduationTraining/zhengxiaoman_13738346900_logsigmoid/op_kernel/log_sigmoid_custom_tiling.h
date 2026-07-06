#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H
#include <cstdint>

struct LogSigmoidCustomTilingData {
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t dataType;
};

#endif // LOG_SIGMOID_CUSTOM_TILING_H

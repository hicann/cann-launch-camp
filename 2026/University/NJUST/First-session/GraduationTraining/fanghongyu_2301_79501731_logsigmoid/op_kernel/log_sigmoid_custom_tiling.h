%%writefile Sources/test/custom_op/op_kernel/log_sigmoid_custom_tiling.h
#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H
#include <cstdint>

struct LogSigmoidCustomTilingData {
    uint32_t totalLength;
    uint32_t ALIGN_NUM;
};

#endif // LOG_SIGMOID_CUSTOM_TILING_H
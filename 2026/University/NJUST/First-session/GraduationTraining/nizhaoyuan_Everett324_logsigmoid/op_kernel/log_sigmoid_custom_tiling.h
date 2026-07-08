#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H
#include <cstdint>


struct LogSigmoidCustomTilingData {
    // 输入张量的元素总数
    uint32_t size;

    // host侧设置的核数，kernel侧按这个值进行数据切分
    uint32_t core_num;
};

#endif // LOG_SIGMOID_CUSTOM_TILING_H

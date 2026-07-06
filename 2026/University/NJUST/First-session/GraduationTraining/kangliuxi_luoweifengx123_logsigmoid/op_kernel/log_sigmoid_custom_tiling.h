#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H

#include <cstdint>

struct LogSigmoidCustomTilingData {
    // 小核（无余数块）处理的总元素数
    uint32_t smallCoreDataNum;
    // 大核（带余数块）处理的总元素数
    uint32_t bigCoreDataNum;
    // 大核的 tile 循环总次数
    uint32_t finalBigTileNum;
    // 小核的 tile 循环总次数
    uint32_t finalSmallTileNum;
    // 单次常规 tile 处理的元素数
    uint32_t tileDataNum;
    // 小核最后一次 tile 实际处理的元素数
    uint32_t smallTailDataNum;
    // 大核最后一次 tile 实际处理的元素数
    uint32_t bigTailDataNum;
    // 大核数量（余数块个数）
    uint32_t tailBlockNum;
};

#endif // LOG_SIGMOID_CUSTOM_TILING_H
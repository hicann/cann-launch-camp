#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H
#include <cstdint>

struct LogSigmoidCustomTilingData {
    // 小核处理的元素个数
    uint32_t smallCoreDataNum;

    // 大核处理的元素个数
    uint32_t bigCoreDataNum;

    // 大核需要执行的 tile 循环次数
    uint32_t finalBigTileNum;

    // 小核需要执行的 tile 循环次数
    uint32_t finalSmallTileNum;

    // 每次 tile 常规处理的元素个数
    uint32_t tileDataNum;

    // 小核最后一次 tile 实际处理的元素个数
    uint32_t smallTailDataNum;

    // 大核最后一次 tile 实际处理的元素个数
    uint32_t bigTailDataNum;

    // 大核数量，也就是核间切分后的余数块数量
    uint32_t tailBlockNum;
};

#endif // LOG_SIGMOID_CUSTOM_TILING_H

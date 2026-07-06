#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H

#include <cstdint>

struct LogSigmoidCustomTilingData {
    // small core 处理的数据量，单位：元素个数
    uint32_t smallCoreDataNum;

    // big core 处理的数据量，单位：元素个数
    uint32_t bigCoreDataNum;

    // big core 需要处理的 tile 数
    uint32_t finalBigTileNum;

    // small core 需要处理的 tile 数
    uint32_t finalSmallTileNum;

    // 每个完整 tile 处理的数据量，单位：元素个数
    uint32_t tileDataNum;

    // small core 最后一个 tile 的数据量
    uint32_t smallTailDataNum;

    // big core 最后一个 tile 的数据量
    uint32_t bigTailDataNum;

    // 前 tailBlockNum 个 Core 是 big core
    uint32_t tailBlockNum;
};

#endif // LOG_SIGMOID_CUSTOM_TILING_H

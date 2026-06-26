#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H

#include <cstdint>

struct LogSigmoidCustomTilingData {
    uint32_t smallCoreDataNum;   // 小核处理的元素个数
    uint32_t bigCoreDataNum;     // 大核处理的元素个数
    uint32_t finalBigTileNum;    // 大核的Tile循环次数
    uint32_t finalSmallTileNum;  // 小核的Tile循环次数
    uint32_t tileDataNum;        // 每次Tile常规处理元素个数
    uint32_t smallTailDataNum;   // 小核最后一次Tile实际处理元素个数
    uint32_t bigTailDataNum;     // 大核最后一次Tile实际处理元素个数
    uint32_t tailBlockNum;       // 大核数量（余数块数）
};

#endif // LOG_SIGMOID_CUSTOM_TILING_H

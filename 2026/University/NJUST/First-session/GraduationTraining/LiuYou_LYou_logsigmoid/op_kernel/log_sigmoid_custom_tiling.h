#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H
#include <cstdint>

struct LogSigmoidCustomTilingData {
    uint32_t smallCoreDataNum;
    uint32_t bigCoreDataNum;
    uint32_t tileDataNum;
    uint32_t smallTailDataNum;
    uint32_t bigTailDataNum;
    uint32_t finalSmallTileNum;
    uint32_t finalBigTileNum;
    uint32_t tailBlockNum;
};

#endif // LOG_SIGMOID_CUSTOM_TILING_H

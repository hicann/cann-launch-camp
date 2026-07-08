

#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H

#include <cstdint>

struct LogSigmoidCustomTilingData
{
    uint32_t normalCoreElemNum;
    uint32_t largeCoreElemNum;
    uint32_t tileElemNum;
    uint32_t normalTileNum;
    uint32_t largeTileNum;
    uint32_t normalTailElemNum;
    uint32_t largeTailElemNum;
    uint32_t largeCoreNum;
};

#endif
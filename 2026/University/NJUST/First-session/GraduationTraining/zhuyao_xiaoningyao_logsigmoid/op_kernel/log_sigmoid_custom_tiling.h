#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H
#include <cstdint>

struct LogSigmoidCustomTilingData {
    uint32_t size;              
    uint32_t coreDataNum;       
    uint32_t tileNum;           
    uint32_t tileDataNum;      
    uint32_t tailDataNum;      
    uint32_t tailBlockNum;     
    uint32_t smallCoreDataNum;
    uint32_t bigCoreDataNum;
    uint32_t smallTailDataNum;
    uint32_t bigTailDataNum;
    uint32_t finalSmallTileNum;
    uint32_t finalBigTileNum;
};

#endif // LOG_SIGMOID_CUSTOM_TILING_H

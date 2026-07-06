#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H
#include <cstdint>

struct LogSigmoidCustomTilingData {
    uint32_t size;              // 总数据量
    uint32_t coreDataNum;       // 每个core处理的数据量（实际使用）
    uint32_t tileNum;           // 每个core的tile数量
    uint32_t tileDataNum;       // 每个tile的数据量
    uint32_t tailDataNum;       // 最后一个tile的数据量
    uint32_t tailBlockNum;      // 尾块数量
    // 为了兼容host代码，添加这些成员
    uint32_t smallCoreDataNum;
    uint32_t bigCoreDataNum;
    uint32_t smallTailDataNum;
    uint32_t bigTailDataNum;
    uint32_t finalSmallTileNum;
    uint32_t finalBigTileNum;
};

#endif // LOG_SIGMOID_CUSTOM_TILING_H

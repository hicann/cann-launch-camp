#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H
#include <cstdint>

struct LogSigmoidCustomTilingData {
    uint32_t smallCoreDataNum;     // 小核处理的总数据量（元素个数）
    uint32_t bigCoreDataNum;       // 大核处理的总数据量（元素个数）
    uint32_t finalBigTileNum;      // 大核tile循环次数（包含尾块）
    uint32_t finalSmallTileNum;    // 小核tile循环次数（包含尾块）
    uint32_t tileDataNum;          // 每个tile最大元素数
    uint32_t smallTailDataNum;     // 小核尾块元素数
    uint32_t bigTailDataNum;       // 大核尾块元素数
    uint32_t tailBlockNum;         // 大核个数（额外分配1个32B块的核数）
};

#endif // LOG_SIGMOID_CUSTOM_TILING_H

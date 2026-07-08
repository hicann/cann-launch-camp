#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H
#include <cstdint>

struct LogSigmoidCustomTilingData {
    uint32_t totalLength;        // 输入元素总数（保留字段，本实现未强制使用）
    uint32_t smallCoreDataNum;   // 小核处理的元素数
    uint32_t bigCoreDataNum;     // 大核处理的元素数
    uint32_t finalBigTileNum;    // 大核核内切分次数
    uint32_t finalSmallTileNum;  // 小核核内切分次数
    uint32_t tileDataNum;        // 单次处理的元素数
    uint32_t smallTailDataNum;   // 小核尾块元素数
    uint32_t bigTailDataNum;     // 大核尾块元素数
    uint32_t tailBlockNum;       // 大核个数
};
#endif // LOG_SIGMOID_CUSTOM_TILING_H

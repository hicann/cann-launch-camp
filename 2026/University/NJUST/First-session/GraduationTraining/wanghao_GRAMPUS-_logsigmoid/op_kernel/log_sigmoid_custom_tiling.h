#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H
#include <cstdint>

struct LogSigmoidCustomTilingData {
    uint32_t smallCoreDataNum;   // 普通核单核算素总数
    uint32_t bigCoreDataNum;     // 大核单核算素总数（多处理1块）
    uint32_t finalBigTileNum;    // 大核总分片数
    uint32_t finalSmallTileNum;  // 普通核总分片数
    uint32_t tileDataNum;        // 单个分片的元素数
    uint32_t smallTailDataNum;   // 普通核最后一片的元素数
    uint32_t bigTailDataNum;     // 大核最后一片的元素数
    uint32_t tailBlockNum;       // 大核的数量
};

#endif // LOG_SIGMOID_CUSTOM_TILING_H

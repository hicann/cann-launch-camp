#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H

#include <cstdint>

// 存储由Host侧计算的Tiling参数，用于Kernel侧分配数据和循环控制
struct LogSigmoidCustomTilingData {
    uint32_t smallCoreDataNum;    // 无尾块核的处理元素数
    uint32_t bigCoreDataNum;      // 有尾块核的处理元素数
    uint32_t finalBigTileNum;     // 大核的tile总数
    uint32_t finalSmallTileNum;   // 小核的tile总数
    uint32_t tileDataNum;         // 每个完整tile的元素数
    uint32_t smallTailDataNum;    // 小核最后一个tile的元素数
    uint32_t bigTailDataNum;      // 大核最后一个tile的元素数
    uint32_t tailBlockNum;        // 尾核个数（前tailBlockNum个核为大核）
};

#endif

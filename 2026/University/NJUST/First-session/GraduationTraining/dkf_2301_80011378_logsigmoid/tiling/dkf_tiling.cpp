%%writefile  Sources/test/custom_op/op_kernel/log_sigmoid_custom_tiling.h
#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H
#include <cstdint>

struct LogSigmoidCustomTilingData {
    uint32_t smallCoreDataNum;    // small core 处理的总元素数
    uint32_t bigCoreDataNum;      // big core 处理的总元素数
    uint32_t finalBigTileNum;     // big core 的 tile 数量
    uint32_t finalSmallTileNum;   // small core 的 tile 数量
    uint32_t tileDataNum;         // 每个 tile 处理的元素数（满 tile）
    uint32_t smallTailDataNum;    // small core 尾 tile 元素数
    uint32_t bigTailDataNum;      // big core 尾 tile 元素数
    uint32_t tailBlockNum;        // 多处理一个数据块的 core 数量（big core 个数）
};

#endif // LOG_SIGMOID_CUSTOM_TILING_H
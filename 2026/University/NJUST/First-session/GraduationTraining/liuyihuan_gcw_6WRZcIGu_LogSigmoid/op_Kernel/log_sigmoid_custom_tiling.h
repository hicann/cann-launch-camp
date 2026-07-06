%%writefile  Sources/test/custom_op/op_kernel/log_sigmoid_custom_tiling.h
#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H

#include <cstdint>

struct LogSigmoidCustomTilingData {
    uint32_t smallCoreDataNum;   // 小核处理的数据量（对齐后）
    uint32_t bigCoreDataNum;     // 大核处理的数据量
    uint32_t finalBigTileNum;    // 大核的 tile 总数
    uint32_t finalSmallTileNum;  // 小核的 tile 总数
    uint32_t tileDataNum;        // 每个 tile 处理的数据个数
    uint32_t smallTailDataNum;   // 小核最后一个 tile 的数据个数
    uint32_t bigTailDataNum;     // 大核最后一个 tile 的数据个数
    uint32_t tailBlockNum;       // 多分配一个 tile 的核数
};

#endif // LOG_SIGMOID_CUSTOM_TILING_H

#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H

#include <cstdint>

struct LogSigmoidCustomTilingData {
    // 输入张量的实际元素总数
    uint32_t size;

    // Host 侧为每个 AI Core 划分的数据长度
    // 最后一个有效核实际处理的数据可能小于该值
    uint32_t blockLength;

    // 单次从 GM 搬入 UB 进行计算的元素数量
    // 最后一个 Tile 实际处理的数据可能小于该值
    uint32_t tileLength;
};

#endif  // LOG_SIGMOID_CUSTOM_TILING_H

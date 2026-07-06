#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H

#include <cstdint>

struct LogSigmoidCustomTilingData {
    // 基础核负责处理的总元素数
    uint32_t baseCoreElemTotal;
    // 满载核负责处理的总元素数
    uint32_t fullCoreElemTotal;
    // 满载核需要执行的tile循环总次数
    uint32_t fullCoreTileTotal;
    // 基础核需要执行的tile循环总次数
    uint32_t baseCoreTileTotal;
    // 单次常规tile处理的元素数量
    uint32_t baseTileElemCount;
    // 基础核最后一次tile实际处理的元素数
    uint32_t baseTailElemCount;
    // 满载核最后一次tile实际处理的元素数
    uint32_t fullTailElemCount;
    // 满载核的数量（即余数块的个数）
    uint32_t remainBlockCount;
};

#endif // LOG_SIGMOID_CUSTOM_TILING_H
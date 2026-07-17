#pragma once

#include <cstdint>

/**
 * @brief 定义算子支持的最大维度数。
 */
constexpr uint32_t LESS_EQUAL_MAX_DIMS = 8;

/**
 * @brief 定义算子的计算模式常量。
 */
constexpr uint32_t LESS_EQUAL_DIRECT = 0;             // 直接计算模式（无广播）
constexpr uint32_t LESS_EQUAL_X1_SCALAR = 1;          // x1为标量计算模式
constexpr uint32_t LESS_EQUAL_X2_SCALAR = 2;          // x2为标量计算模式
constexpr uint32_t LESS_EQUAL_GENERAL_BROADCAST = 3;  // 通用广播计算模式

/**
 * @brief 定义算子的数据运行模式常量。
 */
constexpr uint32_t LESS_EQUAL_RUN_CONTIGUOUS = 0;  // 连续数据运行模式
constexpr uint32_t LESS_EQUAL_RUN_SCALAR = 1;      // 标量数据运行模式

/**
 * @brief LessEqual算子的Tiling参数结构体。
 * 该结构体用于在Host侧计算并传递给Device侧，指导Kernel进行数据分块、内存管理和计算执行。
 */
struct LessEqualTilingData {
    uint64_t totalLength;      // 总计算元素个数，即输出Tensor的总元素数量
    uint64_t blockLength;      // 每个核（Block）处理的元素个数，用于多核切分
    uint64_t vectorRunLength;  // 单次Vector指令计算处理的元素长度，用于核内循环切分

    uint32_t tileLength;  // 每次搬移（DataCopy）的元素个数，即一个Tile的大小
    uint32_t rank;        // 输入输出的维度数（Rank）
    uint32_t mode;        // 计算模式，取值为 LESS_EQUAL_DIRECT, LESS_EQUAL_X1_SCALAR, LESS_EQUAL_X2_SCALAR 或
                          // LESS_EQUAL_GENERAL_BROADCAST
    uint32_t x1RunMode;   // x1输入的运行模式，取值为 LESS_EQUAL_RUN_CONTIGUOUS 或 LESS_EQUAL_RUN_SCALAR
    uint32_t x2RunMode;   // x2输入的运行模式，取值为 LESS_EQUAL_RUN_CONTIGUOUS 或 LESS_EQUAL_RUN_SCALAR

    uint64_t outDims[LESS_EQUAL_MAX_DIMS];  // 输出Tensor的各维度大小数组，用于广播计算时的坐标映射
    uint64_t x1Strides[LESS_EQUAL_MAX_DIMS];  // x1输入Tensor的各维度步长数组，用于非连续数据或广播场景的偏移计算
    uint64_t x2Strides[LESS_EQUAL_MAX_DIMS];  // x2输入Tensor的各维度步长数组，用于非连续数据或广播场景的偏移计算
};

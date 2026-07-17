#pragma once
#include <cstdint>

/**
 * @file less_equal_tiling.h
 * @brief LessEqual算子Tiling数据结构定义
 * 
 * 该文件定义了LessEqual算子的Tiling配置数据结构，
 * 用于在Host侧计算并传递Tiling参数给Kernel侧。
 */

/**
 * @brief LessEqual算子支持的最大维度数
 * 
 * 限制广播后的张量最大维度为16维，
 * 超过此限制的高维张量将被合并到低维处理。
 */
#define LE_MAX_DIM 16

/**
 * @brief LessEqual算子Tiling配置数据结构
 * 
 * 该结构体承载了Host侧Tiling计算的结果，
 * 包括数据分块策略、并行处理配置和内存访问参数，
 * 用于指导Kernel侧进行高效的数据处理。
 */
struct LessEqualTilingData {
    uint64_t totalLen;       /**< 输出张量的总元素数 */
    uint32_t tileLen;        /**< 每个Tile处理的元素数，由UB容量决定 */
    uint32_t blockDim;       /**< 并行计算使用的AI Core核心数 */
    uint32_t perCore;        /**< 每个核心处理的元素数(fast模式)或行数(broadcast模式) */
    uint32_t mode;           /**< 处理模式：0=fast同形模式(输入形状相同)，1=broadcast广播模式 */
    uint32_t ndim;           /**< 广播后的张量维度数 */
    uint32_t lastDimLen;     /**< 最后一维的长度（broadcast模式专用） */
    uint32_t totalRows;      /**< 总行数（broadcast模式专用，即除最后一维外的元素总数） */
    uint32_t outShape[LE_MAX_DIM];    /**< 输出张量各维度的大小 */
    int32_t  x1Stride[LE_MAX_DIM];    /**< 输入张量x1在各维度上的步长（用于广播内存访问） */
    int32_t  x2Stride[LE_MAX_DIM];    /**< 输入张量x2在各维度上的步长（用于广播内存访问） */
};

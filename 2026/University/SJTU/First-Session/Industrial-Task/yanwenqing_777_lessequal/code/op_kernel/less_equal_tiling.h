#pragma once

#include <cstdint>

constexpr uint32_t LESS_EQUAL_MAX_DIM = 25;

constexpr uint32_t LESS_EQUAL_BUFFER_NUM = 1;

constexpr uint32_t LESS_EQUAL_STATIC_HALF_TILE_ELEMENTS = 4096;

constexpr uint32_t LESS_EQUAL_STATIC_FLOAT_TILE_ELEMENTS = 4096;
constexpr uint32_t LESS_EQUAL_STATIC_WORK_BLOCK_ELEMENTS = 32;
constexpr uint32_t LESS_EQUAL_STATIC_FLOAT_WORK_BLOCK_ELEMENTS = 64;

constexpr uint32_t LESS_EQUAL_MODE_SAME_SHAPE = 0;
constexpr uint32_t LESS_EQUAL_MODE_X1_SCALAR = 1;
constexpr uint32_t LESS_EQUAL_MODE_X2_SCALAR = 2;
constexpr uint32_t LESS_EQUAL_MODE_BROADCAST = 3;

constexpr uint32_t LESS_EQUAL_PATH_DIRECT_SCALAR = 0;
constexpr uint32_t LESS_EQUAL_PATH_CONTIGUOUS_VECTOR = 1;
constexpr uint32_t LESS_EQUAL_PATH_BROADCAST_INNER_DIRECT = 2;
constexpr uint32_t LESS_EQUAL_PATH_BROADCAST_INNER_VECTOR = 3;
constexpr uint32_t LESS_EQUAL_PATH_BROADCAST_GENERIC = 4;

constexpr uint32_t LESS_EQUAL_PATH_GLOBAL_SCALAR_VECTOR = 5;

constexpr uint32_t LESS_EQUAL_PATH_CONTIGUOUS_PIPELINE = 6;
constexpr uint32_t LESS_EQUAL_PATH_STATIC_HALF_SAME_SHAPE = 7;

constexpr uint32_t LESS_EQUAL_PATH_STATIC_FLOAT_SAME_SHAPE = 8;

constexpr uint32_t LESS_EQUAL_INNER_CONTIGUOUS = 0;
constexpr uint32_t LESS_EQUAL_INNER_SCALAR = 1;

// 以下宏已删除（未使用）：
// constexpr uint32_t LESS_EQUAL_REUSE_NONE = 0;
// constexpr uint32_t LESS_EQUAL_REUSE_X1 = 1;
// constexpr uint32_t LESS_EQUAL_REUSE_X2 = 2;

struct LessEqualTilingData {
    uint64_t totalLength;

    uint64_t workUnitNum;
    uint64_t smallCoreWorkNum;
    uint64_t bigCoreWorkNum;

    uint64_t innerDataNum;
    uint64_t outerDataNum;
    uint64_t innerSegmentDataNum;
    uint64_t innerSegmentNum;

    
    uint64_t x1OuterStep;
    uint64_t x2OuterStep;

    uint32_t blockDim;
    uint32_t tailBlockNum;
    uint32_t tileDataNum;
    uint32_t dimNum;
    uint32_t mode;
    uint32_t pathMode;
    uint32_t innerStartDim;
    uint32_t x1InnerMode;
    uint32_t x2InnerMode;
    uint32_t workBlockElements;
    uint32_t outerLinear;
    uint32_t reuseInput;   // 保留字段，不再使用但保持结构体布局

    uint64_t outputShape[LESS_EQUAL_MAX_DIM];
    uint64_t outputStride[LESS_EQUAL_MAX_DIM];
    uint64_t x1Stride[LESS_EQUAL_MAX_DIM];
    uint64_t x2Stride[LESS_EQUAL_MAX_DIM];
};
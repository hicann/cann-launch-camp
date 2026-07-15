#pragma once

#include <cstdint>

constexpr uint32_t TILE_LENGTH_FP32 = 8192;
constexpr uint32_t TILE_LENGTH_FP16 = 10240;
constexpr uint32_t TILE_LENGTH_FP16_LARGE = 20480;
constexpr uint32_t DOUBLE_BUFFER = 2;

struct GeluTilingData {
    uint32_t blockNum;
    uint32_t totalLength;
    uint32_t alignedLength;
    uint32_t alignUnit;
    uint32_t numPerCore;
    uint32_t tailNumLastCore;
    uint32_t tileLength;
    uint32_t tanhTmpSize;
};

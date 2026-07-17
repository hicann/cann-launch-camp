#pragma once

#include <cstdint>

constexpr uint32_t LESS_EQUAL_MAX_DIMS = 16;

struct LessEqualTilingData {
    uint64_t total;
    uint64_t baseBlockLength;
    uint64_t tailBlockNum;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;
    uint32_t bufferNum;
    uint32_t inputBufferBytes;
    uint32_t outputBufferBytes;
};

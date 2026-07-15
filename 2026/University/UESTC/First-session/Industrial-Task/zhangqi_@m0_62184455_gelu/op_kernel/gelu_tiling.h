#pragma once

#include <cstdint>

constexpr uint32_t MAX_TILE_LENGTH = 4096;

struct GeluTilingData {
    uint32_t totalLength;
    uint32_t perCoreLength;
    uint32_t tileLength;
};

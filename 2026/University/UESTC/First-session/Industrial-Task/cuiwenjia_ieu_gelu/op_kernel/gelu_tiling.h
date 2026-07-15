#pragma once

#include <cstdint>

struct GeluTilingData {
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t tileNum;
};

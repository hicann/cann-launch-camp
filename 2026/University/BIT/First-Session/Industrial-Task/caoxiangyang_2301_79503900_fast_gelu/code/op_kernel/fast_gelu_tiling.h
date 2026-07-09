// Tiling data for FastGelu.
#pragma once

#include <cstdint>

struct FastGeluTilingData {
    uint64_t length;
    uint32_t tileDataNum;
    uint32_t blockElements;
};

// Tiling data shared by host and kernel.
#pragma once

#include <cstdint>

struct FastGeluTilingData {
    uint32_t length;
    uint32_t blockLength;
    uint32_t tileLength;
};

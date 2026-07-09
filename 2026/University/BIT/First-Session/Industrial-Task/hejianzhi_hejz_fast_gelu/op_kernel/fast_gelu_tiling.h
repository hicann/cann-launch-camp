#pragma once

#include <cstdint>

struct FastGeluTilingData {
    uint64_t length;
    uint32_t blockDim;
    uint64_t blockLength;
};

#pragma once

#include <cstdint>

struct ClipByValueTilingData {
    uint32_t length;
    uint32_t tileLength;
    uint32_t minIsScalar;
    uint32_t maxIsScalar;
};

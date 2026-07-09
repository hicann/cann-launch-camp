#pragma once

#include <cstdint>

struct FastGeluTilingData {
    uint32_t totalLength;
    uint32_t coreNum;
    uint32_t formerNum;
    uint32_t tailNum;
    uint32_t formerLength;
    uint32_t tailLength;
    uint32_t tileLength;
};
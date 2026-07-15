#pragma once

#include <cstdint>

struct GeluTilingData {
    uint32_t totalDataNum;
    uint32_t smallCoreDataNum;
    uint32_t bigCoreDataNum;
    uint32_t tileDataNum;
    uint32_t tailCoreNum;
};

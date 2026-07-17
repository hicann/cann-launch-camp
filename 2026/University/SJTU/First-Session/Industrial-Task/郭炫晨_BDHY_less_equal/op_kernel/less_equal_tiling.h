#pragma once

#include <cstdint>

struct LessEqualTilingData {
    uint64_t totalLength;
    uint32_t tileNumMean;
    uint32_t tileNumEnd;
    uint32_t tileLengthMean;
    uint32_t tileLengthEnd;
    uint32_t blockLengthMean;
    uint32_t blockLengthEnd;
};

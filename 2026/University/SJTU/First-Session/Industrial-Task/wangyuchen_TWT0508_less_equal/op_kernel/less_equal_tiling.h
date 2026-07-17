#pragma once

#include <cstdint>

// max 8 dims after collapsing, raw input up to 64 dims
constexpr uint32_t LESS_EQUAL_MAX_DIMS = 8;
constexpr uint32_t LESS_EQUAL_RAW_DIMS = 64;

// two modes: same shape or broadcast
constexpr uint32_t LE_MODE_DIRECT    = 0;
constexpr uint32_t LE_MODE_BROADCAST = 1;

// data passed from host to kernel, all uint32_t to save space
struct LessEqualTilingData {
    uint32_t count;        // total number of output elements
    uint32_t tile;         // elements per tile
    uint32_t cores;        // number of cores used
    uint32_t perCore;      // work per core (elements in direct mode, rows in broadcast mode)
    uint32_t mode;         // LE_MODE_DIRECT or LE_MODE_BROADCAST
    uint32_t ndim;         // number of dims after collapsing
    uint32_t tail;         // last dim length (always a contiguous run)
    uint32_t rows;         // product of leading dims = total rows
    uint32_t shape[LESS_EQUAL_MAX_DIMS];
    uint32_t s1[LESS_EQUAL_MAX_DIMS];    // x1 stride per dim (0 = broadcast)
    uint32_t s2[LESS_EQUAL_MAX_DIMS];    // x2 stride per dim (0 = broadcast)
};

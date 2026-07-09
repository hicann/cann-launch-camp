#pragma once
#include <cstdint>

struct FastGeluTilingData {
    uint32_t length;
    uint32_t block_num;
    uint32_t block_len;
    uint32_t remainder;
    uint32_t tile_num;
    uint32_t tile_len;
    uint32_t tail_len;
    uint32_t data_type;
};
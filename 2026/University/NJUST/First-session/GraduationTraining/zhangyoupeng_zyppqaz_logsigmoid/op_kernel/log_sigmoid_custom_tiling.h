#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H

#include <cstdint>

namespace optiling {
struct LogSigmoidCustomTilingData {
    uint32_t small_core_data_num;
    uint32_t big_core_data_num;
    uint32_t final_big_tile_num;
    uint32_t final_small_tile_num;
    uint32_t tile_data_num;
    uint32_t small_tail_data_num;
    uint32_t big_tail_data_num;
    uint32_t tail_block_num;
};
} // namespace optiling

#endif // LOG_SIGMOID_CUSTOM_TILING_H

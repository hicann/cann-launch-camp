#pragma once

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

#pragma pack(push, 8)
struct alignas(8) QmmCustomTilingData {
    TCubeTiling cubeTilingData;
    uint32_t isPertoken;
    uint32_t workspaceSize;
};
#pragma pack(pop)

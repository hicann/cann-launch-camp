#ifndef GELU_TILING_H
#define GELU_TILING_H
#include <cstdint>

struct GeluTilingData {
    uint32_t smallCoreDataNum;
    uint32_t bigCoreDataNum;
    uint32_t finalBigTileNum;
    uint32_t finalSmallTileNum;
    uint32_t tileDataNum;
    uint32_t smallTailDataNum;
    uint32_t bigTailDataNum;
    uint32_t tailBlockNum;
    uint32_t input_dtype_flag;
};

#endif // GELU_TILING_H
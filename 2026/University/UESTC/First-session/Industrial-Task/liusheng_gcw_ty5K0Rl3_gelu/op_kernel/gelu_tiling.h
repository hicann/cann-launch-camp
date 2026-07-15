#pragma once

#include <cstdint>

struct GeluTilingData
{
    // 每个小核处理的数据量；尾部 block 之后的核使用该配置。
    uint32_t smallCoreDataNum;
    // 每个大核处理的数据量；前 tailBlockNum 个核多处理一个 block。
    uint32_t bigCoreDataNum;
    // 大核需要处理的 tile 数量。
    uint32_t finalBigTileNum;
    // 小核需要处理的 tile 数量。
    uint32_t finalSmallTileNum;
    // 常规 tile 中包含的元素数量。
    uint32_t tileDataNum;
    // 小核最后一个 tile 的实际元素数量。
    uint32_t smallTailDataNum;
    // 大核最后一个 tile 的实际元素数量。
    uint32_t bigTailDataNum;
    // 需要按大核配置处理的核数量。
    uint32_t tailBlockNum;
};

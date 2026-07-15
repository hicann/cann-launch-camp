#pragma once
#include <cstdint>

struct GeluTilingData {
    uint32_t totalLength;       // 输入真实元素总数，kernel用于边界钳位
    uint32_t smallCoreDataNum;  // 小核分配元素数(已对齐)
    uint32_t bigCoreDataNum;    // 大核分配元素数(比小核多32B)
    uint32_t tileDataNum;       // 每个tile处理元素数(UB自适应)
    uint32_t smallTailDataNum;  // 小核最后一个tile的元素数
    uint32_t bigTailDataNum;    // 大核最后一个tile的元素数
    uint32_t finalSmallTileNum; // 小核tile总数
    uint32_t finalBigTileNum;   // 大核tile总数
    uint32_t tailBlockNum;      // 前N个核为大核(多分配32B)
};

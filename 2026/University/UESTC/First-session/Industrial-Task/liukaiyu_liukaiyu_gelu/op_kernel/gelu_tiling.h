// ==========================================================
//  GELU Tiling结构体 — 大核/小核模型
// ==========================================================
//  核间: 按余数分配, 前tailBlockNum个大核(多1元素), 其余为小核
//  核内: tileDataNum根据总数据量分级设定, 尾tile做32B上对齐
//  GELU公式: gelu(x) = 0.5 * x * (1 + erf(x / √2))
#pragma once

#include <cstdint>

struct GeluTilingData {
    uint32_t smallCoreDataNum;    // 小核分配元素数
    uint32_t bigCoreDataNum;      // 大核分配元素数 (=smallCoreDataNum+1)
    uint32_t finalBigTileNum;     // 大核迭代轮数
    uint32_t finalSmallTileNum;   // 小核迭代轮数
    uint32_t tileDataNum;         // 每tile元素数 (按数据量分级)
    uint32_t smallTailDataNum;    // 小核尾tile大小 (已32B对齐)
    uint32_t bigTailDataNum;      // 大核尾tile大小 (已32B对齐)
    uint32_t tailBlockNum;        // 大核个数 (=totalElems % coreNum, 0则无大核)
    uint32_t bufferNum;           // 缓冲槽数 (3)
    uint32_t usedCoreNum;         // 实际使用核心数
};
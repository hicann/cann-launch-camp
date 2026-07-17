// EleWise 规范 Tiling 结构体 — 区分首/尾 block，支持多核切分与 UB 切分
#pragma once

#include <cstdint>

struct GeluTilingData {
    uint32_t dim0;                   // 元素总数
    uint32_t coreNum;                // 实际使用的核数
    uint32_t blockFormer;            // 每核处理元素数（512 元素对齐）
    uint32_t blockNum;               // 总 block 数
    uint32_t ubFormer;               // 每 UB 块元素数（256B 对齐）
    uint32_t ubLoopOfFormerBlock;    // 首 block 的 UB 完整循环次数
    uint32_t ubTailOfFormerBlock;    // 首 block 的 UB 尾段元素数
    uint32_t ubLoopOfTailBlock;      // 末 block 的 UB 完整循环次数
    uint32_t ubTailOfTailBlock;      // 末 block 的 UB 尾段元素数
};

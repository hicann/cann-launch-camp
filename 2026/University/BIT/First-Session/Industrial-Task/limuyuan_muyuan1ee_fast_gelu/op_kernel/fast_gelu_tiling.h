// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// 多核 + 多 tile 切分的 Tiling 参数（plain POD，配合 REGISTER_TILING_DEFAULT）。
// 切分策略（按 tile 均分，负载最优）：
//   totalTiles = ceil(totalLength / tileLength)
//   usedCoreNum = min(coreNum, totalTiles)          // 不超过物理核数，也不超过 tile 数
//   tilesPerCore = totalTiles / usedCoreNum          // 基准每核 tile 数
//   tailCoreNum  = totalTiles % usedCoreNum          // 前 tailCoreNum 个核各多扛 1 tile
// 核间 tile 数最多相差 1，负载近乎完美均衡；且只要 totalTiles >= coreNum 就一定用满全部核，
// 不会像"余数全塞末核"方案那样因 blockLength 向上对齐而把核数压低、导致核空闲。
// 全局唯一可能非整 tile 的是末核末 tile（valid < tileLength），其余 tile 全满 -> 走 DataCopy 快路径。
struct FastGeluTilingData {
    uint64_t totalLength;   // 展平后总元素个数；0 表示空张量（kernel no-op）
    uint32_t usedCoreNum;   // 实际使用的核数（= host SetBlockDim 的值）
    uint64_t tileLength;    // 每个 tile 的元素数（32B 对齐）
    uint32_t tilesPerCore;  // 基准每核 tile 数
    uint32_t tailCoreNum;   // 前 tailCoreNum 个核各多 1 tile（取值 [0, usedCoreNum)）
    uint32_t dtypeSize;     // 每元素字节数：fp16=2, fp32=4
};

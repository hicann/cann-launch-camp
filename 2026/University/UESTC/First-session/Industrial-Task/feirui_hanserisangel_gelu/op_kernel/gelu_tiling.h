// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// GELU 算子泛化 Tiling 结构体
// 支持核间不均分（大小核拆分）+ 核内不均分（尾块处理）
struct GeluTilingData {
    uint32_t smallCoreDataNum;   // 小核处理的总数据量（元素个数）
    uint32_t bigCoreDataNum;     // 大核处理的总数据量（元素个数）
    uint32_t finalBigTileNum;    // 大核数据搬运总批次次数
    uint32_t finalSmallTileNum;  // 小核数据搬运总批次次数
    uint32_t tileDataNum;        // 单核单次可搬运数据量（元素个数）
    uint32_t smallTailDataNum;   // 小核最后一批处理数据量
    uint32_t bigTailDataNum;     // 大核最后一批处理数据量
    uint32_t tailBlockNum;       // 大核个数（前 tailBlockNum 个核为大核）
};

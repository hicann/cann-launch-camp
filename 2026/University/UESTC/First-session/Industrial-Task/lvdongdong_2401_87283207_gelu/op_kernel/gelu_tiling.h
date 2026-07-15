// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// 泛化 Tiling(负载均衡版): 按 32B 块在核间均分, 余数由前 tailBlockNum 个"大核"各多担一块;
// 关键点——所有核统一循环 finalTileNum 次, 再把各核的块数在这些片间尽量均摊
// (前 remTiles 片各多担 1 个 32B 块), 使每一轮各核负载几乎相等, 消除"尾轮只有大核在转"的空转.
// host 侧预计算全部分片参数, kernel 侧直接消费, 避免核内标量重算.
// 说明: 元素总数与 GM 偏移相关的字段用 64 位, 避免超大张量(FP16 >~21 亿 / FP32 >~10 亿元素)
// 在 32 位下溢出导致分核/偏移错误; 循环计数与块内小量(受 UB 约束, 远小于 2^32)保持 32 位.
struct GeluTilingData {
    uint64_t length;             // 输入真实元素总数(用于末核尾块非对齐安全裁剪)
    uint64_t bigCoreDataNum;     // 大核处理的元素数(比小核多一个32B块, 用于计算 GM 偏移)
    uint64_t smallCoreDataNum;   // 小核处理的元素数(32B块对齐)
    uint64_t smallCoreBaseOffset; // 所有大核占据的元素总数(= bigCoreDataNum*tailBlockNum), 小核起址基准(host 预算)
    uint32_t tailBlockNum;       // 大核个数(前 tailBlockNum 个核为大核, <= 核数, 恒小)
    uint32_t tileDataNum;        // UB 单缓冲分片上限(元素, 物理最大, 用于开 buffer)
    uint32_t finalTileNum;       // 每个核统一的核内循环次数(大小核一致 => 无尾轮空转)
    uint32_t bigBaseTileBlock;   // 大核每片基础块数(前 bigRemTiles 片各多担 1 块)
    uint32_t bigRemTiles;        // 大核中"多担一块"的片数
    uint32_t smallBaseTileBlock; // 小核每片基础块数(前 smallRemTiles 片各多担 1 块)
    uint32_t smallRemTiles;      // 小核中"多担一块"的片数
};

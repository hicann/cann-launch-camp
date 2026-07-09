// Tiling结构体定义的头文件 - 优化版: 支持大/小核负载均衡
#pragma once

#include <cstdint>

struct FastGeluTilingData {
    uint32_t length;           // 总数据元素个数
    uint32_t blockNum;         // 实际使用的核数
    // 小核(大部分核)的tiling参数
    uint32_t smallCoreDataNum; // 小核处理数据元素数
    uint32_t smallTileNum;     // 小核tile数目
    uint32_t smallTailDataNum; // 小核尾块元素数
    // 大核(少部分核)的tiling参数
    uint32_t bigCoreDataNum;   // 大核处理数据元素数
    uint32_t bigTileNum;       // 大核tile数目
    uint32_t bigTailDataNum;   // 大核尾块元素数
    // 公共参数
    uint32_t tileDataNum;      // 每个tile的标准数据元素数
    uint32_t tailBlockNum;     // 大核的核数
    uint32_t tmpSize;          // FasterGelu临时buffer大小(byte)
};

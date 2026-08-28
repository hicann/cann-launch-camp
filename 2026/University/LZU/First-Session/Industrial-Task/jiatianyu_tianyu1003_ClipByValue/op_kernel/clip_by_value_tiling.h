// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct ClipByValueTilingData {
    // 原始总元素数（未对齐）
    uint32_t totalLength;
    // 32字节对齐对应的元素个数（= 32 / dtypeSize）
    uint32_t alignNum;
    // 每块大小（= alignNum，32B对齐）
    uint32_t blockSize;

    // 总对齐块数
    uint32_t totalBlockNum;
    // 启动核数
    uint32_t coreNum;
    // 每核平均块数
    uint32_t avgBlocksPerCore;
    // 余数块数（前 remBlocks 个核各多处理一块）
    uint32_t remBlocks;

    // 广播模式标识
    uint32_t minIsScalar;    // clip_value_min是否为标量（0=否，1=是）
    uint32_t maxIsScalar;    // clip_value_max是否为标量（0=否，1=是）
};

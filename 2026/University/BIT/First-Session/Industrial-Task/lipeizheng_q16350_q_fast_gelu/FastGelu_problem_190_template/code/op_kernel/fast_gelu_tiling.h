// FastGelu Host/Kernel 之间传递的 Tiling 参数。
#pragma once

#include <cstdint>

struct FastGeluTilingData {
    uint64_t totalLength;  // 输入张量的真实元素总数
    uint64_t blockLength;  // 每个 AIV 核最多处理的真实元素数
    uint32_t tileLength;   // 每次搬入 UB 的元素数（按 32 Byte 对齐）
    uint32_t reserved;     // 保留字段，保证结构体自然对齐并便于扩展
};

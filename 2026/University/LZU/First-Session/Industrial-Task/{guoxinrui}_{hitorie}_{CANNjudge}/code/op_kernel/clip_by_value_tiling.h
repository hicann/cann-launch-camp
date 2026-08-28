// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct ClipByValueTilingData {
    // 可能原本存在的成员
    uint32_t length;   // 已有（从 did you mean 'length' 可知）
    
    // 添加缺失的成员
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileLength;
    bool     minIsScalar;
    bool     maxIsScalar;
    uint32_t bufferDepth;
    
    // ... 其他可能字段
};

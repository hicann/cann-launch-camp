#pragma once  // 防止头文件被重复包含

#include <cstdint>  // 引入标准整数类型（如 uint32_t）

struct GeluTilingData {
    uint32_t length;      ///< 输入张量的总元素个数（整个张量的大小）
    uint32_t tileLength;  ///< 每个 tile（数据块）包含的元素个数，即 Kernel 内单次处理的数据量
};

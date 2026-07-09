// Tiling结构体定义的头文件
#pragma once
#include <cstdint>

struct FastGeluTilingData {
    // 输入Tensor的元素总数。保留 length 字段，兼容原工程。
    uint32_t length;
    // 实际启用的AI Core数量。Host设置，Kernel按该值把总数据均分到多核。
    uint32_t blockDim;
};

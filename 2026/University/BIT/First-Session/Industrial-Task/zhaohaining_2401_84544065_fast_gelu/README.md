# FastGelu Ascend C 算子实验

## 实验简介

本实验基于华为 CANN Ascend C 自定义算子开发框架，实现 FastGelu 激活函数算子。算子对输入张量进行逐元素计算，输出与输入保持相同 shape 和 dtype。

FastGelu 计算公式可写为：

```text
y = x * exp(0.851 * (x - abs(x))) / (1 + exp(-1.702 * abs(x)))
```

在最终实现中，为提升性能，使用等价的快速形式：

```text
y = x / (1 + exp(-1.702 * x))
```

## 支持特性

- 支持数据类型：`float16`、`float32`
- 支持数据格式：`ND`
- 支持任意维度输入张量
- 支持非 32 对齐长度场景
- 输出 shape 与输入 shape 完全一致
- 输出 dtype 与输入 dtype 完全一致

## 文件说明

```text
code/
├── CMakeLists.txt                    # 工程构建入口
├── op_host/
│   ├── CMakeLists.txt                # host 侧构建配置
│   └── fast_gelu.cpp                 # 算子注册、shape/type 推导、tiling 计算
└── op_kernel/
    ├── CMakeLists.txt                # kernel 侧构建配置
    ├── fast_gelu.cpp                 # Ascend C kernel 实现
    ├── fast_gelu_tiling.h            # tiling 数据结构定义
    └── tiling_key_fast_gelu.h        # dtype 模板选择定义
```

## 实现思路

1. Host 侧获取输入总元素数 `length_x`。
2. 根据输入规模动态选择 `block_dim`，实现多核分片。
3. Kernel 侧将输入按一维连续内存处理。
4. 每个 AIV core 负责一段数据。
5. 每段数据再按 tile 搬入 UB，使用向量指令完成计算。
6. 使用 `DataCopyPad` 处理非对齐尾块，保证隐藏测试中的非 32 对齐场景正确。

## 性能优化

本实验主要采用以下优化：

- 使用一次 `Exp` 的快速计算形式，减少向量计算开销。
- 使用多核并行处理大张量。
- 根据输入长度动态调整 `block_dim`，兼顾小规模和大规模测试点性能。
- 使用 UB 分块计算，减少全局内存访问开销。
- 保持 `TQue` 队列方式，确保数据搬运和计算同步稳定。

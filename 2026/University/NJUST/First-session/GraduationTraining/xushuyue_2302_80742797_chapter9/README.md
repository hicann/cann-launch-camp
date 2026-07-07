# Chapter 9：LogSigmoid 自定义算子

## 项目信息

- 目录名称：`xushuyue_2302_80742797_chapter9`
- 任务内容：完成 LogSigmoid 算子的 Kernel 侧核函数与 Host 侧 Tiling 函数
- 目标硬件：Ascend 910B
- 支持输入/输出类型：
  - `float32`
  - `float16`
  - `bfloat16`

## 文件说明

本目录包含以下三个文件：

1. `README.md`：项目概览与实现说明。
2. `代码文件.md`：三个需要写入工程的完整源代码。
3. `运行说明.md`：代码写入、测试运行及常见现象说明。

## 算子功能

LogSigmoid 的计算公式为：

```text
LogSigmoid(x) = log(1 / (1 + exp(-x)))
              = -log(1 + exp(-x))
```

实现中：

- `float32` 数据直接进行向量计算。
- `float16` 和 `bfloat16` 数据先转换为 `float32` 计算，再转换回原始类型。
- Host 侧根据输入总元素数和数据类型选择合适的 Block 数。
- Kernel 侧按 Tile 分块处理数据。

## 代码写入位置

三个源文件分别写入：

```text
Sources/test/custom_op/op_host/log_sigmoid_custom.cpp
Sources/test/custom_op/op_kernel/log_sigmoid_custom_tiling.h
Sources/test/custom_op/op_kernel/log_sigmoid_custom.cpp
```

完整内容见 `代码文件.md`。

## 测试用例

| Case | Shape | 数据类型 |
|---|---|---|
| case1 | `(8, 16, 64)` | `float32` |
| case2 | `(8, 16, 1743)` | `float32` |
| case3 | `(4, 2028)` | `float16` |
| case4 | `(32, 1001, 7763)` | `float16` |
| case5 | `(1, 1024)` | `bfloat16` |
| case6 | `(64, 64, 1024)` | `bfloat16` |

> case4 数据量很大，生成输入数据、Golden 数据以及执行测试都可能耗时较长。

# Gelu 算子任务说明

## 任务目标

实现一个基于 Ascend C 的 `Gelu` 自定义算子，用于对输入张量逐元素执行 GELU 激活函数计算，并将结果写入输出张量。

该算子需要完成 host 侧算子注册、shape/type 推导、tiling 参数计算，以及 device 侧 AICore kernel 计算逻辑。

## 算子功能

`Gelu` 算子对输入张量 `input_x` 中的每个元素执行 GELU 激活计算，输出结果保存到 `output`。

GELU 常用公式如下：

```text
GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

本工程中：

- `float32` 路径使用 erf 形式计算。
- `float16` 路径使用 sigmoid 近似形式计算。

## 输入输出说明

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `input_x` | 输入 | 待计算 GELU 的输入张量 |
| `output` | 输出 | GELU 计算后的输出张量 |

## 数据类型与格式

算子支持的数据类型：

- `float16`
- `float32`

算子支持的数据格式：

- `ND`

输出张量的 shape 与输入张量保持一致，输出数据类型与输入数据类型保持一致。

## 代码模块划分

### host 侧模块

文件位置：

```text
code/op_host/gelu.cpp
```

host 侧主要负责：

- 注册 `Gelu` 算子的输入和输出。
- 声明算子支持的数据类型和数据格式。
- 设置 shape 推导函数。
- 设置数据类型推导函数。
- 根据输入规模、数据类型长度、AICore 数量和 UB 空间计算 tiling 参数。
- 将 tiling 参数写入 `GeluTilingData`，供 device 侧 kernel 使用。

### device 侧模块

文件位置：

```text
code/op_kernel/gelu.cpp
```

device 侧主要负责：

- 根据 block 索引确定当前 core 需要处理的数据范围。
- 按 tile 将输入数据从 GM 搬运到本地 buffer。
- 调用 Ascend C 向量接口完成 GELU 计算。
- 将计算结果从本地 buffer 写回 GM。

### tiling 数据结构

文件位置：

```text
code/op_kernel/gelu_tiling.h
```

`GeluTilingData` 用于保存 host 侧计算出的切分参数：

| 字段 | 说明 |
| --- | --- |
| `totalDataNum` | 输入总元素数量 |
| `smallCoreDataNum` | 普通 core 处理的数据量 |
| `bigCoreDataNum` | 多分配一个 block 的 core 处理的数据量 |
| `tileDataNum` | 单个 tile 处理的数据量 |
| `tailCoreNum` | 需要处理更多数据的 core 数量 |

## 实现流程

1. host 侧获取输入 shape 和数据类型。
2. host 侧根据输入元素数量和数据类型长度计算 32 字节对齐后的 block 数量。
3. host 侧根据平台 AICore 数量和输入 block 数量确定实际使用的 core 数。
4. host 侧根据 UB 空间计算每个 tile 可处理的数据量。
5. host 侧生成 tiling 参数并传递给 device 侧。
6. device 侧每个 core 根据 tiling 参数确定自己的数据范围。
7. device 侧按 tile 循环执行 `CopyIn -> Compute -> CopyOut`。
8. 所有 core 完成后，输出张量得到完整 GELU 结果。

## 计算逻辑

### float16

`float16` 输入使用 sigmoid 近似公式：

```text
y = x / (1 + exp(-1.595769122 * (x + 0.0455399241 * x^3)))
```

该路径减少复杂函数调用，更适合半精度向量计算。

### float32

`float32` 输入使用标准 erf 公式：

```text
y = 0.5 * x * (1 + erf(x * 0.7071067811865475))
```

其中 `0.7071067811865475` 约等于 `1 / sqrt(2)`。

## 构建目标

工程构建后需要生成自定义算子包，包含：

- host 侧 tiling 库。
- ACLNN 调用接口库。
- device 侧 AICore kernel 库。

目标芯片配置：

```text
ascend910b
```

## 约束说明

- 输入和输出 shape 必须一致。
- 输入和输出数据类型必须一致。
- 当前实现支持 `float16` 和 `float32`。
- 当前实现面向 `ascend910b`。
- workspace 大小为 0。

## 验收要点

完成该算子任务后，应重点确认：

- 算子能够完成编译并生成自定义算子包。
- `float16` 和 `float32` 输入均能正确执行。
- 输出 shape 与输入 shape 一致。
- 输出数据类型与输入数据类型一致。
- 大小不整除 core 或 tile 的输入也能正确处理尾块数据。

# LessEqual 自定义算子说明文档

## 1. 算子功能

本项目实现 Ascend C 自定义算子 `LessEqual`，功能等价于 TensorFlow / NumPy 中的逐元素小于等于比较：

```text
out = x1 <= x2
```

输出类型为 `bool`，在算子内部按 `int8_t` / `uint8_t` 字节形式写出，其中 `1` 表示 `true`，`0` 表示 `false`。

本版本对应提交包：`LessEqual_final_v97_v95_v96_blend.zip`。

## 2. 支持范围

### 2.1 输入数据类型

支持以下输入类型：

- `float16`
- `float32`
- `int32`
- `int8`

`x1` 和 `x2` 使用相同输入类型，输出为 `bool`。

### 2.2 形状与广播

支持：

- 一维、二维、三维、四维及更高维张量；
- 连续同形状输入；
- 标量与张量广播；
- 向量与矩阵广播；
- 不同形状高维张量广播；
- 非 32B 对齐长度；
- 空张量输出。

广播规则遵循 NumPy 风格：从右向左对齐维度，当某一维相等或其中一方为 `1` 时可以广播。

## 3. 工程结构

提交包主要结构如下：

```text
code/
├── CMakeLists.txt
├── op_host/
│   ├── CMakeLists.txt
│   └── less_equal.cpp
└── op_kernel/
    ├── CMakeLists.txt
    ├── less_equal.cpp
    ├── less_equal_tiling.h
    └── tiling_key_less_equal.h
```

各文件作用：

| 文件 | 作用 |
|---|---|
| `op_host/less_equal.cpp` | Host 侧 shape 推导、广播 stride 计算、tiling 切分、blockDim / tileLength 设置 |
| `op_kernel/less_equal.cpp` | AI Core kernel 实现，包括连续、标量广播、通用广播和不同 dtype 的比较逻辑 |
| `op_kernel/less_equal_tiling.h` | Host 与 Kernel 共用的 tiling 数据结构 |
| `op_kernel/tiling_key_less_equal.h` | 按输入 dtype 生成模板选择信息 |

## 4. 核心实现思路

### 4.1 Host 侧 tiling

Host 侧主要完成以下工作：

1. 根据 `x1`、`x2` 的 shape 推导输出 shape；
2. 判断是否为连续同形状场景，即 `noBroadcast`；
3. 为广播场景生成 `outputShape`、`x1Strides`、`x2Strides`；
4. 从最右维开始合并可连续处理的后缀维度，生成 `segmentLength`，减少 Kernel 中逐元素计算广播 offset 的开销；
5. 根据 dtype、总元素数、是否多维连续等信息选择 `blockDim` 和 `tileLength`。

本版本对多维连续同形状场景采用中等粒度切核策略，兼顾测试点 2 的维度覆盖和其它测试点的整体稳定性。

### 4.2 Kernel 侧执行路径

Kernel 侧根据 tiling 信息分为三类主要路径：

1. **连续同形状路径**  
   当 `noBroadcast == 1` 时，直接按连续地址搬运 `x1` 和 `x2`，执行向量比较并写出结果。

2. **分段广播路径**  
   对可以在右侧后缀维度内连续复用的广播场景，使用 `segmentLength` 分段处理。这样每个 segment 只需要计算一次外层广播坐标，减少除法和取模开销。

3. **标量广播路径**  
   对 `float16` 和 `float32` 的标量-张量比较，使用 `CompareScalar` 减少 scalar duplicate 的 UB 写入开销。对于整数类型保持稳定计算路线，避免精度和指令支持风险。

## 5. 不同 dtype 的比较策略

### 5.1 float16 / float32

普通向量比较使用：

```text
Compare(LE) -> Select -> Cast(bool)
```

`Compare` 输出为 bit mask，不能直接作为 bool 字节输出，因此需要通过 `Select` 展开为数值 0/1，再 `Cast` 为 bool 字节。

标量广播时，使用等价变换减少 scalar 扩展：

```text
scalar <= tensor  等价于  tensor >= scalar
tensor <= scalar  直接使用 tensor <= scalar
```

### 5.2 int32

Ascend C 中 `int32` 的比较指令对 `LE` 支持有限，因此使用等价转换：

```text
x1 <= x2  等价于  min(x1, x2) == x1
```

该方法对 `INT_MIN`、`INT_MAX` 等边界值也保持精确。

### 5.3 int8

`int8` 先转换为 `half`，再使用向量比较：

```text
int8 -> half -> Compare(LE) -> Select -> Cast(bool)
```

由于 `int8` 到 `half` 是精确转换，因此不会引入比较误差。同时避免使用 `Duplicate<int8_t>`，保证在 CANN 8.5 环境下可编译。

## 6. 边界与正确性处理

本版本重点保证以下情况正确：

- `N = 1`；
- `N = 10000`；
- 非 32B 对齐长度；
- `x1 == x2`；
- `INT_MIN / INT_MAX`、`INT8_MIN / INT8_MAX` 等边界值；
- `float16 / float32` 中的普通值、边界值和 NaN 场景。

对于 `float16 / float32` 的 scalar-only 情况，内部使用 bit-level 比较辅助处理，避免在 AI Core 侧直接进行 half 标量比较，从而规避编译风险。

## 7. 性能优化点

本版本主要采用以下优化策略：

1. **按 dtype 模板实例化**  
   使用 tiling key 按输入 dtype 选择 kernel 模板，减少运行时 dtype 分支。

2. **广播后缀合并**  
   从右向左合并可连续处理的广播维度，通过 `segmentLength` 降低广播 offset 计算频率。

3. **float32 双缓冲**  
   对 `float32` 启用双缓冲，提高 GM 搬运和 Vector 计算的重叠度。

4. **连续路径简化**  
   连续同形状时不做广播 offset 计算，直接按线性地址读写。

5. **中等粒度多维切核**  
   对多维连续同形状场景采用中等目标元素数，减少过多小 core 带来的初始化开销，同时避免过粗切核导致并行度不足。

6. **避免高风险特判**  
   未使用容易引发整体退步或 Wrong Answer 的 `TBuf direct`、复杂 all-true 判断、min/max 边界捷径、row-cache、split-copy 等策略。

## 8. 已知取舍

本版本是综合评分导向的平衡方案：

- 优先保证测试点 1、3、4 的优势路径；
- 测试点 2 通过中等切核策略尽量改善，但不为了单点极限牺牲其它测试点；
- 测试点 5 保持稳定精度路线，不做可能影响正确性的激进优化。

根据调试记录，v97 的一次测试结果为：

| 指标 | 结果 |
|---|---:|
| 综合评分 | 55.97 |
| 测试点 1 | 3.02 μs |
| 测试点 2 | 4.42 μs |
| 测试点 3 | 13.02 μs |
| 测试点 4 | 3.12 μs |
| 测试点 5 | 17.98 μs |

实际分数可能随平台负载、调度波动和隐藏用例抽样略有变化。

## 9. 提交方式

将压缩包中的 `code/` 目录作为最终提交内容即可。提交包应保持如下结构：

```text
code/
├── CMakeLists.txt
├── op_host/
└── op_kernel/
```

不要额外修改 `less_equal_tiling.h` 中的 include 为 kernel-only 头文件，避免 Host 编译阶段找不到 `kernel_operator.h`。

## 10. 注意事项

- 不要使用 `Duplicate<int8_t>`，该指令组合在当前 CANN 8.5 环境中存在编译风险；
- 不要直接在 AI Core 函数中做 half 标量比较；
- `Compare` 的输出是 bit mask，不是 bool 字节，必须经 `Select` / `Cast` 生成 bool 输出；
- 对 `int32` 不直接使用 `Compare(LE)`，采用 `Min + EQ` 保证可编译和正确性；
- 调整性能时应优先修改 Host 侧切核参数，谨慎修改 Kernel 主计算路线。

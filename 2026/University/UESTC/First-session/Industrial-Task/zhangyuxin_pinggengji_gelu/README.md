# GELU 自定义算子实现说明

## 1. 项目概述

本项目实现了 Ascend 910B 平台上的 GELU 自定义算子。算子输入为 `float16` 或 `float32` 类型的 ND Tensor，输出 Tensor 与输入 Tensor 保持相同 shape 和 dtype。

GELU 数学定义为：

```text
GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

本实现以 PyTorch `torch.nn.functional.gelu` 的 exact 计算结果作为参考，重点保证以下要求：

- 支持 `float16` 和 `float32`
- 支持任意 ND 输入
- 支持最后一维 `N ∈ [1, 10240]`
- 支持非 32B 对齐场景
- 输出 shape 与输入完全一致
- 输出 dtype 与输入完全一致
- float32 误差满足 `abs < 1e-4` 且 `rel < 1e-4`
- float16 误差满足 `abs < 1e-3` 且 `rel < 1e-3`

当前提交版本在 CANNJudge 中通过全部 6 个测试点，提交得分约为 40.24。

---

## 2. 文件结构

本次提交基于 CANNJudge 官方模板，仅修改必要文件：

```text
code/
├── CMakeLists.txt
├── op_host/
│   ├── CMakeLists.txt
│   └── gelu.cpp
└── op_kernel/
    ├── CMakeLists.txt
    ├── gelu.cpp
    ├── gelu_tiling.h
    └── tiling_key_gelu.h
```

主要修改文件：

```text
op_host/gelu.cpp
op_kernel/gelu.cpp
op_kernel/gelu_tiling.h
```

保持未修改文件：

```text
CMakeLists.txt
op_host/CMakeLists.txt
op_kernel/CMakeLists.txt
op_kernel/tiling_key_gelu.h
```

---

## 3. Host 侧 Tiling 策略

Host 侧负责计算输入总元素数、多核切分参数，并将 tiling 数据传递给 kernel。

### 3.1 Tiling 字段

`GeluTilingData` 包含以下字段：

```cpp
struct GeluTilingData {
    uint32_t totalLength;
    uint32_t activeBlockNum;
    uint32_t coreLength;
};
```

字段含义：

| 字段 | 含义 |
|---|---|
| `totalLength` | 输入 Tensor flatten 后的总元素数 |
| `activeBlockNum` | 实际启动的 AIV block 数 |
| `coreLength` | 非最后一个 block 处理的元素数 |

### 3.2 多核切分

当前版本采用多核 tiling：

```text
MIN_LENGTH_PER_CORE = 8192
CORE_ALIGN_NUM = 16
```

切分逻辑：

1. 获取输入总元素数 `totalLength`
2. 获取平台 AIV 核数 `coreNum`
3. 对小输入保持较少 block，避免空核和调度开销
4. 对大输入按 `MIN_LENGTH_PER_CORE` 估算启动核数
5. 非最后 block 的长度按 16 元素对齐
6. 最后一个 block 处理所有剩余元素

这样可以避免：

- 空核
- 重复写
- 漏写
- 非最后 block 起点不对齐
- 最后一核尾块处理错误

---

## 4. Kernel 侧实现策略

Kernel 侧采用 flatten 后逐元素处理的方式，每个 block 负责一段连续元素区间。

### 4.1 Tile 设置

当前最终版本使用：

```text
TILE_LENGTH = 2048
BUFFER_NUM = 1
```

说明：

- `TILE_LENGTH=2048` 是在本地与 CANNJudge 实验后选择的较稳定版本
- 当前最终版本未启用 double buffer
- double buffer 版本已验证正确，但在当前评分环境中未体现稳定性能收益，因此不作为最终默认提交版本

### 4.2 block 内循环

每个 block 内按 tile 循环处理：

```text
offset = tileIndex * TILE_LENGTH
curLen = min(TILE_LENGTH, blockLength - offset)
alignedLen = AlignUp(curLen, 32B 对齐元素数)
```

其中：

- float16：32B 对齐对应 16 个元素
- float32：32B 对齐对应 8 个元素

完整对齐 tile 使用普通 `DataCopy`。

非对齐 tail 使用 `DataCopyPad`，避免逐元素 GM 标量读写。

---

## 5. 数据类型处理

### 5.1 float32 路径

float32 路径保留手写 exact erf GELU：

```text
y = 0.5 * x * (1 + erf(x / sqrt(2)))
```

该路径使用 AscendC 向量 API 完成：

- `Muls`
- `Erf`
- `Adds`
- `Mul`
- `Muls`

保留该路径的原因是：CANNJudge 中 `AscendC::Gelu<float, ...>` 实验会导致部分测试点 Wrong Answer，无法满足题目 exact erf 参考要求。

### 5.2 float16 路径

当前 40.24 分版本中，float16 路径使用 AscendC 高阶 GELU API：

```cpp
AscendC::Gelu<half, true, false>(yLocal, xLocal, alignedLen);
```

其中：

```text
highPrecision = true
highPerformance = false
```

该策略在 CANNJudge 中通过全部测试点。相比完全手写 erf 的 half 路径，部分测试点性能有所改善。

### 5.3 禁用的实现

以下实现经过实验后未作为最终版本：

```text
AscendC::Gelu<float, ...>
FasterGelu
FasterGeluV2
tanh 近似 GELU
sigmoid 近似 GELU
快速 erf 近似
```

原因：

- `AscendC::Gelu<float, ...>` 在部分测试点出现 Wrong Answer
- 快速 erf 近似虽可通过，但性能低于当前最高分版本
- 题目要求 exact erf GELU，不能使用不满足精度的近似路径

---

## 6. 非对齐 Tail 处理

非 32B 对齐是本题正确性的重点。

早期版本使用：

```cpp
inputGm.GetValue(...)
outputGm.SetValue(...)
```

逐元素处理 tail，虽然正确，但性能较差。

最终版本将非对齐 tail 改为 `DataCopyPad`：

### 输入 tail

```text
只读取 curLen 个有效元素
UB padding 区补 0
按 alignedLen 参与向量计算
```

### 输出 tail

```text
只写回 curLen 个有效元素
不按 alignedLen 越界写 GM
```

这样同时保证了：

- 不越界读
- 不越界写
- UB padding 不含随机值
- 避免逐元素标量 GM 访问
- 兼容 float16 / float32

---

## 7. 正确性验证

本地测试覆盖了以下场景：

```text
[1]
[5]
[31]
[32]
[33]
[127]
[129]
[10240]
[1025]
[1027]
[2049]
[2, 5]
[3, 31]
[2, 3, 5]
[2, 2, 3, 33]
[2, 1025]
[2, 3, 1027]
[8, 2048]
```

测试数据类型：

```text
float16
float32
```

测试输入类型：

```text
random
fixed_values
large_values
float16_boundaries
```

固定值包括：

```text
0, ±1, ±2
```

大值和边界值用于验证数值稳定性。

所有测试均满足题目精度要求。

---

## 8. CANNJudge 提交结果

当前保留版本为：

```text
half：AscendC::Gelu<half, true, false>
float：手写 exact erf GELU
DataCopyPad tail
TILE_LENGTH = 2048
MIN_LENGTH_PER_CORE = 8192
多核 tiling
```

CANNJudge 结果：

```text
通过：6 / 6
输出错误占比：0.00%
分数：约 40.24
```

各测试点耗时如下：

| 测试点 | 状态 | 用时 |
|---|---|---:|
| 1 | Pass | 3.64 μs |
| 2 | Pass | 12.59 μs |
| 3 | Pass | 33.24 μs |
| 4 | Pass | 7.01 μs |
| 5 | Pass | 24.95 μs |
| 6 | Pass | 164.17 μs |

---

## 9. 优化过程总结

### 9.1 单核正确性 baseline

最初实现单核版本：

```text
blockDim = 1
TILE_LENGTH = 1024
workspace = 0
```

重点解决：

- exact erf GELU
- float16 转 float32 计算
- 非 32B 对齐 tail
- 任意 ND 输入 flatten 处理

### 9.2 多核 tiling

加入多核切分：

```text
activeBlockNum
coreLength
totalLength
```

大输入性能明显提升。

### 9.3 DataCopyPad tail 优化

将 tail 中的 `GetValue / SetValue` 标量 GM 访问替换为 `DataCopyPad`。

该优化显著改善了部分非对齐隐藏测试点性能。

### 9.4 double buffer 实验

实现过 double buffer 版本，正确性通过，但性能收益不稳定，因此未作为最终版本。

### 9.5 高阶 GELU API 实验

实验结果：

- half 高阶 GELU 可通过并带来部分性能改善
- float 高阶 GELU 会导致 Wrong Answer
- 因此最终采用 half 高阶、float 手写 exact erf 的混合策略

---

## 10. 最终结论

当前最终版本优先保证正确性，并在此基础上进行了多核、tail 搬运和 half 高阶 API 优化。

最终采用方案：

```text
多核 tiling
TILE_LENGTH = 2048
BUFFER_NUM = 1
DataCopyPad 处理非对齐 tail
float16 使用 AscendC::Gelu<half, true, false>
float32 使用手写 exact erf GELU
workspace = 0
```

该版本在 CANNJudge 中通过全部测试点，满足题目正确性和精度要求。

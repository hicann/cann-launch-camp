# FastGelu Ascend C 自定义算子实现

## 1. 项目简介

本项目实现 CANNJudge FastGelu 自定义算子，基于 Ascend C 完成 Host 侧 tiling 与 Kernel 侧向量化计算。算子支持 `float16`、`float32` 两种输入类型，支持 ND 格式输入，输出 shape 与 dtype 均与输入保持一致。

---

## 2. 计算公式

题目给出的 FastGelu 公式为：

```text
output = x * exp(0.851 * (x - |x|)) / (1 + exp(-1.702 * |x|))
```

对公式分段化简：

当 `x >= 0` 时：

```text
|x| = x
x - |x| = 0
output = x / (1 + exp(-1.702 * x))
```

当 `x < 0` 时：

```text
|x| = -x
x - |x| = 2x
output = x * exp(1.702 * x) / (1 + exp(1.702 * x))
       = x / (1 + exp(-1.702 * x))
```

因此最终可使用等价公式：

```text
output = x / (1 + exp(-1.702 * x))
```

也可以写成：

```text
output = x * sigmoid(1.702 * x)
```

最终版本中，大多数场景采用单指数形式 `x / (1 + exp(-1.702 * x))`，小规模 `float32` 输入使用 `x * sigmoid(1.702 * x)` 快路径。

---

## 3. 文件结构

核心文件如下：

```text
op_host/fast_gelu.cpp
op_kernel/fast_gelu.cpp
```

其中：

- `op_host/fast_gelu.cpp`：负责算子注册、输入输出类型声明、shape/type 推导、tiling 参数计算和 blockDim 设置。
- `op_kernel/fast_gelu.cpp`：负责 Ascend C Kernel 实现，包括 GM 到 UB 搬运、向量计算和结果写回。

---

## 4. Host 侧设计

Host 侧主要完成以下任务：

1. 获取输入张量 dtype 和 shape size。
2. 根据 dtype 设置模板选择参数。
3. 根据 UB 大小估算 `tileDataNum`。
4. 根据输入长度和 32B 对齐要求设置 blockDim。
5. 将 `length` 和 `tileDataNum` 写入 tiling data。

关键参数如下：

```cpp
constexpr uint32_t TILING_BUFFER_NUM = 2;
constexpr uint32_t TILING_TMP_FLOAT_BUF_CNT = 4;
constexpr uint32_t UB_RESERVED_RATIO_NUM = 9;
constexpr uint32_t UB_RESERVED_RATIO_DEN = 10;
constexpr uint32_t BLOCK_SIZE_BYTES = 32;
constexpr uint32_t SMALL_FLOAT32_SIGMOID_THRESHOLD = 24;
```

### 4.1 tileDataNum 计算

`tileDataNum` 根据 UB 可用空间估算，计算时同时考虑输入队列、输出队列和临时 float buffer。最终版本中保留了较保守的 `TILING_TMP_FLOAT_BUF_CNT = 4`，虽然实际部分路径用不到这么多临时空间，但实测整体性能更加稳定。

### 4.2 对齐分核

最终版本采用对齐分核策略，使每个 core 处理的数据量尽可能按 32B 对齐。这样可以减少非对齐尾块数量，降低 `DataCopyPad` 触发次数，提升整体性能稳定性。

---

## 5. Kernel 侧设计

Kernel 侧采用典型三段式流程：

```text
CopyIn  : GM -> UB
Compute : UB 向量计算
CopyOut : UB -> GM
```

### 5.1 数据切分

每个 core 根据 blockDim 和输入总长度计算自己的处理区间：

```cpp
uint32_t perCore = CeilDiv(length, coreNum);
uint32_t alignedPerCore = AlignUp(perCore, alignElem);
uint32_t coreOffset = coreIdx * alignedPerCore;
```

其中 `alignElem = 32 / sizeof(DT_X)`，用于保证每个 core 的数据切分尽量符合 32B 对齐要求。

### 5.2 数据搬运

对齐场景使用普通 `DataCopy`，非对齐场景使用 `DataCopyPad`：

```cpp
if ((copyBytes % BLOCK_SIZE_BYTES) == 0) {
    AscendC::DataCopy(...);
} else {
    AscendC::DataCopyPad(...);
}
```

调试过程中曾尝试移除 `DataCopyPad`，但非对齐测试点会出现 Wrong Answer 或 Runtime Error，因此最终版本必须保留该逻辑。

### 5.3 计算路径

float32 大张量路径使用单指数公式：

```cpp
AscendC::Muls(down, xLocal, -FASTGELU_ALPHA, curNum);
AscendC::Exp(down, down, curNum);
AscendC::Adds(down, down, 1.0f, curNum);
AscendC::Div(yLocal, xLocal, down, curNum);
```

小规模 float32 路径使用 Sigmoid：

```cpp
AscendC::Muls(yLocal, xLocal, FASTGELU_ALPHA, curNum);
AscendC::Sigmoid(yLocal, yLocal, curNum);
AscendC::Mul(yLocal, yLocal, xLocal, curNum);
```

float16 路径先 Cast 到 float32 中间变量，再进行 Exp、Div 计算，最后 Cast 回 float16：

```cpp
AscendC::Cast(xF, xLocal, AscendC::RoundMode::CAST_NONE, curNum);
AscendC::Muls(down, xF, -FASTGELU_ALPHA, curNum);
AscendC::Exp(down, down, curNum);
AscendC::Adds(down, down, 1.0f, curNum);
AscendC::Div(xF, xF, down, curNum);
AscendC::Cast(yLocal, xF, AscendC::RoundMode::CAST_RINT, curNum);
```

这样可以保证 float16 精度更加稳定。

---

## 6. 最终关键参数

Kernel 侧最终参数：

```cpp
constexpr int32_t BUFFER_NUM = 2;
constexpr float FASTGELU_ALPHA = 1.702f;
constexpr uint32_t BLOCK_SIZE_BYTES = 32;
constexpr uint32_t SMALL_FLOAT32_SIGMOID_THRESHOLD = 24;
```

Host 侧最终参数：

```cpp
constexpr uint32_t TILING_BUFFER_NUM = 2;
constexpr uint32_t TILING_TMP_FLOAT_BUF_CNT = 4;
constexpr uint32_t UB_RESERVED_RATIO_NUM = 9;
constexpr uint32_t UB_RESERVED_RATIO_DEN = 10;
constexpr uint32_t BLOCK_SIZE_BYTES = 32;
constexpr uint32_t SMALL_FLOAT32_SIGMOID_THRESHOLD = 24;
```

---

## 7. 调试与优化过程

### 7.1 初始版本

初始版本直接按照题目公式实现：

```text
output = x * exp(0.851 * (x - |x|)) / (1 + exp(-1.702 * |x|))
```

该版本计算流程包含 `Abs`、`Sub`、两次 `Exp`、`Mul`、`Adds`、`Div` 等操作，指令数量较多。虽然精度正确，但性能不够理想。

### 7.2 公式化简优化

通过数学推导，将公式等价化简为：

```text
output = x / (1 + exp(-1.702 * x))
```

该版本减少了 `Abs`、`Sub` 和一次 `Exp`，计算链路更短，性能明显提升，因此成为最终主路径。

### 7.3 全量 Sigmoid 尝试

曾尝试所有输入都使用：

```text
output = x * sigmoid(1.702 * x)
```

但实测中大规模输入性能下降，说明 `Sigmoid` 算子在当前环境中并不一定优于显式 `Exp + Div`。因此最终只在小规模 float32 场景使用 Sigmoid。

### 7.4 小张量混合策略

最终采用混合策略：

```text
小 float32：x * sigmoid(1.702 * x)
其他场景：x / (1 + exp(-1.702 * x))
```

经过多轮测试，`SMALL_FLOAT32_SIGMOID_THRESHOLD = 24` 整体表现较好。

### 7.5 DataCopyPad 调试

尝试去掉 `DataCopyPad` 后，非 32B 对齐场景出现错误。最终保留：

```text
32B 对齐：DataCopy
非 32B 对齐：DataCopyPad
```

这样既保证性能，也保证非对齐场景正确性。

### 7.6 BUFFER_NUM 调试

尝试将 `BUFFER_NUM = 2` 改成 `BUFFER_NUM = 1`，希望降低 UB 使用量，但实测性能下降。最终保留双缓冲：

```cpp
constexpr int32_t BUFFER_NUM = 2;
```

### 7.7 临时 buffer 数量调试

尝试降低 Host 侧临时 buffer 估算数量，使 `tileDataNum` 更大，但实测部分测试点性能下降。最终保留：

```cpp
constexpr uint32_t TILING_TMP_FLOAT_BUF_CNT = 4;
```

该设置虽然保守，但整体表现更稳定。

### 7.8 downBuf 延迟初始化尝试

曾尝试在小 float32 Sigmoid 路径中不初始化 `downBuf`，但实测性能下降。最终恢复统一初始化：

```cpp
pipe.InitBuffer(downBuf, this->tileDataNum * sizeof(float));
```

### 7.9 单 tile 快路径尝试

曾尝试对 `tileNum == 1` 的场景单独走快路径，减少循环判断开销，但实测属于负优化，可能增加了分支开销和寄存器压力。因此最终保留普通循环结构。

### 7.10 最终版本

最终版本采用以下优化组合：

```text
1. 主路径使用单指数公式。
2. 小 float32 使用 Sigmoid 快路径。
3. float16 使用 float32 中间计算。
4. 保留 DataCopyPad，支持非 32B 对齐。
5. 使用对齐分核，减少非对齐尾块。
6. 保留 BUFFER_NUM = 2。
7. Host 侧保留 TILING_TMP_FLOAT_BUF_CNT = 4。
8. 不额外拆分 Process/Compute 结构，避免负优化。
```

---

## 8. 测试场景

测试需要覆盖：

- `float16`
- `float32`
- 1D、2D、3D、4D 和高维 ND 输入
- 非 32B 对齐输入
- 空张量
- 单元素输入
- 大规模输入
- 正数、负数、0、接近 0 的数值
- 极大正数和极大负数

精度要求：

```text
float32：相对误差和绝对误差小于 1e-4
float16：相对误差和绝对误差小于 1e-3
```

---

## 9. 总结

本项目通过 Ascend C 实现了 FastGelu 自定义算子。最终版本没有直接使用题目原始公式，而是通过数学等价变换，将计算简化为单指数形式：

```text
output = x / (1 + exp(-1.702 * x))
```

同时结合小 float32 Sigmoid 快路径、float16 转 float32 中间计算、对齐分核、非对齐 DataCopyPad 处理和双缓冲队列，在保证正确性和精度的前提下获得了较好的综合性能。

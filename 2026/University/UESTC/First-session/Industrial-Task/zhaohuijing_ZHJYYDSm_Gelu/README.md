# GELU Ascend C 自定义算子

## 1. 项目简介

本项目实现了一个基于 Ascend C 的 GELU（Gaussian Error Linear Unit）自定义算子，用于在昇腾 NPU 上完成逐元素 GELU 激活计算。

GELU 是 Transformer、BERT、GPT、ViT 等模型中常用的激活函数。该算子以 PyTorch 原生 `torch.nn.functional.gelu` / `torch.nn.GELU` 的默认精确模式作为对标目标，要求在 `float16` 和 `float32` 两种数据类型下保持较高数值精度，并兼容 ND 格式、多维 shape 以及非 32 整倍数的数据规模。

本项目当前支持平台配置：

```text
ascend910b
```

---

## 2. 算子功能说明

### 2.1 算子名称

```text
Gelu
```

### 2.2 Kernel 入口

```cpp
extern "C" __global__ __aicore__ void gelu(
    GM_ADDR input_x,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling
);
```

### 2.3 输入输出

| 类型     | 名称        | 数据类型                 | 数据格式 | 说明        |
| ------ | --------- | -------------------- | ---- | --------- |
| Input  | `input_x` | `float16`, `float32` | ND   | 输入张量      |
| Output | `output`  | 与输入一致                | ND   | GELU 计算结果 |

输出张量满足：

```text
output.shape == input_x.shape
output.dtype == input_x.dtype
```

---

## 3. 数学定义

本算子采用 GELU 精确公式：

```text
GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
```

其中：

```text
erf(x)
```

为误差函数。当前 kernel 侧实现直接调用 Ascend C 提供的：

```cpp
AscendC::Gelu(zLocal, xLocal, curProcessNum);
```

完成逐元素 GELU 计算。

---

## 4. 支持范围

### 4.1 数据类型

当前支持：

```text
float16
float32
```

host 侧通过输入 dtype 设置 `input_dtype_flag`：

```text
0: float16
1: float32
```

kernel 侧根据该标志分别实例化：

```cpp
KernelGelu<half, half>
KernelGelu<float, float>
```

### 4.2 Shape 范围

支持 ND 格式输入，shape 可覆盖：

```text
0 维标量
1 维张量
2 维张量
3 维张量
4 维张量
含 batch 维度的 5~8 维张量
```

维度可表示为：

```text
(..., N4, N3, N2, N)
```

其中：

```text
N  ∈ [1, 10000]
N2 ∈ [1, 10000]
N3 ∈ [1, 1000]
N4 ∈ [1, 1000]
```

### 4.3 非对齐场景

算子需要兼容以下非 32 整倍数场景：

```text
N  非 32 整倍数
N2 非 32 整倍数
N3 非 32 整倍数
N4 非 32 整倍数
总元素数非 32B 对齐
```

当前 host 侧 tiling 逻辑以 32 字节为基本 block 单位，对输入总字节数进行向上对齐后分配计算任务。

---

## 5. 精度要求

结果需与 PyTorch 默认精确模式 GELU 对齐：

```python
torch.nn.functional.gelu(x)
```

精度判定标准如下：

| 数据类型    |   绝对误差 |   相对误差 |
| ------- | -----: | -----: |
| float32 | ≤ 1e-5 | ≤ 1e-4 |
| float16 | ≤ 1e-2 | ≤ 1e-3 |

误差计算建议：

```python
abs_error = abs(custom_output - torch_output)
rel_error = abs_error / max(abs(torch_output), eps)
```

其中 `eps` 可取：

```python
1e-12
```

用于避免除零。

---

## 6. 特殊值处理要求

| 输入              | 期望输出           |
| --------------- | -------------- |
| `NaN`           | 输出 `NaN`       |
| `+Inf`          | 输出 `+Inf`      |
| `-Inf`          | 输出趋近 `0` 的负侧结果 |
| 大正数，如 `x >= 6`  | 输出近似 `x`       |
| 大负数，如 `x <= -6` | 输出近似 `0`       |

---

## 7. 工程目录



说明：

```text
op_host/gelu.cpp
```

负责算子注册、shape 推导、dtype 推导和 tiling 计算。

```text
op_kernel/gelu.cpp
```

负责 AICore kernel 实现，包括 GM 数据读取、UB 内计算和结果写回。

```text
op_kernel/gelu_tiling.h
```

定义 host 与 kernel 共用的 tiling 数据结构。

```text
op_kernel/tiling_key_gelu.h
```

定义模板 dtype 选择和 tiling data 注册信息。

---

## 8. Host 侧实现说明

host 侧主要完成以下工作。

### 8.1 获取输入规模

```cpp
uint32_t inputNum = context->GetInputShape(0)
    ->GetStorageShape()
    .GetShapeSize();
```

该值表示输入张量的元素总数。

### 8.2 获取输入 dtype 字节数

```cpp
ge::DataType inputDtype = context->GetInputDesc(0)->GetDataType();
ge::TypeUtils::GetDataTypeLength(inputDtype, typeLength);
```

当前支持：

```text
float16: 2 bytes
float32: 4 bytes
```

### 8.3 32 字节对齐

```cpp
const uint32_t BLOCK_SIZE = 32;

uint32_t inputLength = inputNum * typeLength;
uint32_t inputLengthAlign32 =
    ((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

uint32_t totalBlock = inputLengthAlign32 / BLOCK_SIZE;
```

host 侧以 32B block 为粒度计算总 block 数。

### 8.4 分核策略

当前分核策略如下：

| totalBlock 范围       | 使用核数    |
| ------------------- | ------- |
| `totalBlock <= 32`  | 1 核     |
| `totalBlock <= 64`  | 最多 2 核  |
| `totalBlock <= 128` | 最多 4 核  |
| `totalBlock <= 256` | 最多 8 核  |
| 其他情况                | 最多 20 核 |

实际核数还会受以下条件限制：

```cpp
coreNum <= hwCoreNum
coreNum <= totalBlock
coreNum >= 1
```

然后通过：

```cpp
context->SetBlockDim(coreNum);
```

设置 block dim。

### 8.5 大小核数据划分

host 侧将任务划分为：

```text
大核：处理 everyCoreInputBlockNum + 1 个 block
小核：处理 everyCoreInputBlockNum 个 block
```

对应字段：

```cpp
smallCoreDataNum
bigCoreDataNum
tailBlockNum
```

其中：

```text
tailBlockNum
```

表示前多少个 core 使用大核数据量。

### 8.6 UB tile 划分

host 侧读取 UB 大小：

```cpp
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
```

当前 tile 大小估算逻辑为：

```cpp
tileBlockNum = (ubSize * 8 / 10) / BLOCK_SIZE / 2;
```

含义是：

```text
使用约 80% UB
预留 2 份 buffer：输入 xBuf 和输出 zBuf
```

然后得到：

```cpp
tileDataNum
smallTailDataNum
bigTailDataNum
finalSmallTileNum
finalBigTileNum
```

这些字段会写入 `GeluTilingData`，供 kernel 侧使用。

---

## 9. Kernel 侧实现说明

kernel 侧核心类为：

```cpp
template<typename TYPE_X, typename TYPE_Y>
class KernelGelu
```

### 9.1 Init 阶段

`Init` 主要完成：

```text
1. 获取当前 core id
2. 判断当前 core 是大核还是小核
3. 计算当前 core 的 GM 偏移 gmOffset
4. 设置输入输出 GlobalTensor
5. 初始化 UB buffer
```

大核偏移：

```cpp
gmOffset = coreIdx * bigCoreDataNum;
```

小核偏移：

```cpp
gmOffset = tailBlockNum * bigCoreDataNum +
           (coreIdx - tailBlockNum) * smallCoreDataNum;
```

### 9.2 Process 阶段

每个 core 按 tile 循环处理：

```cpp
for (uint32_t i = 0; i < this->tileNum; i++) {
    uint32_t curProcessNum = this->tileDataNum;

    if (i == this->tileNum - 1) {
        curProcessNum = this->tailDataNum;
    }

    uint32_t offset = i * this->tileDataNum;

    AscendC::DataCopy(xLocal, xGm[offset], curProcessNum);
    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::Gelu(zLocal, xLocal, curProcessNum);
    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::DataCopy(zGm[offset], zLocal, curProcessNum);
    AscendC::PipeBarrier<PIPE_ALL>();
}
```

处理流程为：

```text
GM -> UB
UB 内 GELU 计算
UB -> GM
```

当前实现使用单输入 buffer 和单输出 buffer，暂未启用 double buffer。

---





## 10. 关键注意事项

### 10.1 tiling 数据结构必须保持一致

host 侧和 kernel 侧必须使用完全一致的 `GeluTilingData` 字段顺序：

```cpp
struct GeluTilingData {
    uint32_t smallCoreDataNum;
    uint32_t bigCoreDataNum;
    uint32_t finalBigTileNum;
    uint32_t finalSmallTileNum;
    uint32_t tileDataNum;
    uint32_t smallTailDataNum;
    uint32_t bigTailDataNum;
    uint32_t tailBlockNum;
    uint32_t input_dtype_flag;
};
```

字段顺序、字段类型和字段数量必须严格一致，否则 kernel 侧读取 tiling 数据时会发生错位。

### 10.2 `tiling_key_gelu.h` 中的 tiling data 注册需检查

当前 `tiling_key_gelu.h` 中存在：

```cpp
BEGIN_TILING_DATA_DEF(GeluTilingDataDef)
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Gelu, GeluTilingDataDef)
```

如果工程框架依赖 `GeluTilingDataDef` 进行 tiling 数据注册和序列化，应确保其中定义的字段与 `GeluTilingData` 保持一致。

否则可能出现：

```text
host 侧写入 tiling 正常
kernel 侧读取 tiling 异常
tiling 字段全 0
运行结果错误
kernel 执行异常
```

### 10.3 非对齐 shape 需要重点验证

当前 host 侧以 32B 对齐后的长度计算 `totalBlock`，并据此分配每个 core 的处理元素数。

因此在非 32B 对齐场景中，需要重点确认：

```text
kernel 实际 DataCopy 的元素数是否超过真实 inputNum
最后一个 core / 最后一个 tile 是否正确裁剪
output 是否发生越界写
```

推荐在 tiling 数据中额外保留真实元素数 `inputNum`，并在 kernel 侧处理最后一个 tile 时根据真实元素数进行边界保护。

### 10.4 dtype flag 需要与注册 dtype 对齐

当前 dtype 分支逻辑为：

```cpp
tiling->input_dtype_flag = (inputDtype == ge::DT_FLOAT16) ? 0U : 1U;
```

kernel 侧逻辑为：

```cpp
if (tilingData.input_dtype_flag == 0) {
    KernelGelu<half, half> op;
} else {
    KernelGelu<float, float> op;
}
```

因此需要确保 OpDef 中仅注册：

```text
ge::DT_FLOAT16
ge::DT_FLOAT
```

不应传入 int、double、bool 等不支持类型。

### 10.5 输出 shape 和 dtype 不应改变

host 侧 shape 推导：

```cpp
*y_shape = *x1_shape;
```

host 侧 dtype 推导：

```cpp
context->SetOutputDataType(0, inputDataType);
```

测试时需要确认：

```text
输出 shape 与输入 shape 完全一致
输出 dtype 与输入 dtype 完全一致
```

---

## 11. 常见问题排查

### 11.1 编译报找不到 `gelu_tiling.h`

检查 include 路径：

```cpp
#include "gelu_tiling.h"
#include "../op_kernel/gelu_tiling.h"
```

确认 host 侧和 kernel 侧都能找到该头文件。

### 11.2 编译报找不到 `tiling_key_gelu.h`

检查 host 侧路径：

```cpp
#include "../op_kernel/tiling_key_gelu.h"
```

确保 `tiling_key_gelu.h` 位于 `op_kernel/` 目录下，或修改为工程实际路径。

### 11.3 kernel 运行结果全 0 或随机错误

优先检查：

```text
tiling 数据是否正确传入
GeluTilingData 字段是否 host/kernel 一致
input_dtype_flag 是否正确
coreDataNum 是否为 0
tileDataNum 是否为 0
```

### 11.4 非对齐 shape 结果异常

优先检查：

```text
inputNum 是否被 32B 对齐后扩大
最后一个 tile 是否越界读取
最后一个 tile 是否越界写入
DataCopy 长度是否满足 Ascend C 要求
```

### 11.5 float16 精度不稳定

建议检查：

```text
是否与 PyTorch float16 结果直接对比
是否误用了 float32 参考输出
是否使用了过严的误差阈值
特殊值和极大值是否单独判断
```



## 12. 当前实现摘要

当前 GELU 算子实现采用：

```text
host 侧：
1. 获取输入元素数和 dtype
2. 按 32B 对齐计算 block 数
3. 根据 totalBlock 选择 coreNum
4. 划分大小核任务
5. 根据 UB 大小计算 tileDataNum
6. 写入 GeluTilingData

kernel 侧：
1. 根据 blockIdx 判断大核/小核
2. 计算当前 core 的 GM 偏移
3. 将输入从 GM 拷贝到 UB
4. 调用 AscendC::Gelu 完成逐元素计算
5. 将结果从 UB 拷贝回 GM
6. 根据 input_dtype_flag 分别支持 half 和 float
```


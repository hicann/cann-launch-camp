# FastGelu 算子实现

## 作者

- 姓名：zengyancheng
- 账号：hi497925191
- 平台：CANN ascend910b (Da Vinci 架构)

## 实现概述

基于 Ascend C 语法实现 FastGelu 激活函数算子，核心公式：

$$FastGelu(x) = \frac{x}{1 + e^{-1.702 \times x}}$$

该公式与原始定义 `x × exp(0.851×(x-|x|)) / (1+exp(-1.702×|x|))` 在数学上完全等价，但运算次数减少56%（9步→4步）。

### 数学证明

- **x ≥ 0**：|x| = x，x-|x| = 0，exp(0)=1，原式 = x/(1+exp(-1.702x)) = 简化式 ✓
- **x < 0**（令 a=|x|=-x）：原式 = x·exp(-1.702a)/(1+exp(-1.702a)) = x/(exp(1.702a)+1) = x/(1+exp(-1.702x)) = 简化式 ✓

## 核心优化

### 1. 公式化简（9步→4步，-56%）

```cpp
Muls(yLocal, xLocal, (DT_X)(-1.702), length);  // y = -1.702*x
Exp(yLocal, yLocal, length);                     // y = exp(-1.702*x)
Adds(yLocal, yLocal, (DT_X)1.0, length);         // y = 1+exp(-1.702*x)
Div(yLocal, xLocal, yLocal, length);             // y = x/(1+exp(-1.702*x))
```

### 2. `constexpr` 编译期硬编码

TILE_LENGTH 在编译期确定，消除运行时除法/mod运算带来的 Scalar 开销：

```cpp
// f16: 4*TILE_LEN*2 + TILE_LEN*4 + TILE_LEN*4 = 16*TILE_LEN → 192KB/16 = 12288
// f32: 4*TILE_LEN*4                         = 16*TILE_LEN → 192KB/16 = 12288
static constexpr uint32_t TILE_LEN = 12288;
```

### 3. f16 混合精度计算

f16 输入先 Cast 到 f32 完成 Exp/Div 运算，再 Cast 回 f16，相比直接在 f16 上算4步精度更高：

```cpp
if constexpr (IsSameType<T, half>::value) {
    Cast(xf, xLocal, RoundMode::CAST_NONE, len);
    Muls(tmp, xf, -1.702f, len);
    Exp(tmp, tmp, len);
    Adds(tmp, tmp, 1.0f, len);
    Div(xf, xf, tmp, len);
    Cast(yLocal, xf, RoundMode::CAST_RINT, len);
}
```

### 4. f32 零临时缓冲区计算

f32 路径直接在 yLocal 上复合计算，无需单独的 tmp_ 缓冲区（省 48KB UB）：

```cpp
Muls(yLocal, xLocal, -1.702f, len);   // 直接在 yLocal 中算
Exp(yLocal, yLocal, len);
Adds(yLocal, yLocal, 1.0f, len);
Div(yLocal, xLocal, yLocal, len);
```

### 5. CopyIn/CopyOut Full/Pad 拆分

中间 tile（占绝大多数）走纯 DataCopy 路径，无分支；仅尾 tile 走 DataCopyPad 处理非对齐数据：

```cpp
if (nextLen == TILE_LEN) CopyInFull(nextOff);
else CopyInPad(nextOff, nextLen);
```

### 6. 双缓冲流水线

多 tile 场景下 CopyIn(N+1) 与 Compute(N) 并行执行，DMA 与向量计算重叠。

### 7. 非对齐数据处理

```cpp
DataCopyExtParams cp{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
DataCopyPadExtParams<T> pp{true, 0,
    static_cast<uint8_t>(AlignUp(len, ALIGN_NUM) - len), 0};
DataCopyPad(xLocal, xGm_[offset], cp, pp);
```

## 文件说明

```
src/
├── CMakeLists.txt                    # 顶层构建脚本
├── op_host/
│   ├── CMakeLists.txt                # Host侧构建
│   └── fast_gelu.cpp                 # Host侧Tiling实现（均分+MIN_ELEM=64）
└── op_kernel/
    ├── CMakeLists.txt                # Kernel侧构建
    ├── fast_gelu.cpp                 # Kernel侧核函数
    ├── fast_gelu_tiling.h            # Tiling结构体（3字段）
    └── tiling_key_fast_gelu.h        # TilingKey模板
```

### 数据结构：FastGeluTilingData

| 字段 | 含义 |
|------|------|
| `totalLength` | 总数据元素数 |
| `blockLength` | 每核处理的元素数（32字节对齐） |
| `blockDim` | 使用的核数 |

## 运行方式

### 编译

```bash
cd custom_op
bash build.sh
./build_out/custom_opp*.run --install-path=${HOME}/
```

### 调用

```cpp
#include "aclnn_fast_gelu.h"

uint64_t workspaceSize = 0;
aclOpExecutor *executor;
aclnnFastGeluGetWorkspaceSize(x, out, &workspaceSize, &executor);
aclnnFastGelu(workspace, workspaceSize, executor, stream);
```

### 性能测试

```bash
source ${HOME}/vendors/customize/bin/set_env.bash
msprof op --output=./prof ./execute_op
```

## 性能数据

| 测试点 | 耗时 | 最优 | 状态 |
|--------|------|------|------|
| 1 (小数据) | 2.68μs | 2.20μs | — |
| 2 | 2.30μs | 2.12μs | — |
| 3 | 6.66μs | 5.82μs | — |
| 4 | 6.46μs | 5.66μs | — |
| 5 (大数据) | 8.40μs | 7.26μs | — |

**总分**：待 CI 评测

msprof 分析结论：Vector 计算仅占 12% 耗时（4条向量指令），瓶颈在 Scalar 队列管理和 DMA 延迟。通过 constexpr 硬编码 + Full/Pad 拆分 + f32 零 tmpBuf + 双缓冲流水线，在算子代码层面已做到极限。

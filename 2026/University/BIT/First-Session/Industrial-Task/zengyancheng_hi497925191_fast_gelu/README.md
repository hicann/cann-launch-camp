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
- **x < 0**（令 a=|x|=-x）：原式 = x·exp(-1.702a)/(1+exp(-1.702a)) = x(1/a)/((1+1/a)/a)·a = x/(1+exp(1.702a)) = x/(1+exp(-1.702x)) = 简化式 ✓

## 核心优化

### 1. 公式化简（9步→4步，-56%）

```cpp
// 仅需4条向量指令，无需临时缓冲区
Muls(yLocal, xLocal, (DT_X)(-1.702), length);  // y = -1.702*x
Exp(yLocal, yLocal, length);                     // y = exp(-1.702*x)
Adds(yLocal, yLocal, (DT_X)1.0, length);         // y = 1+exp(-1.702*x)
Div(yLocal, xLocal, yLocal, length);             // y = x/(1+exp(-1.702*x))
```

### 2. 非对齐数据处理

使用 `DataCopyPad` + `DataCopyExtParams` 处理最后一个 tile 的非对齐数据：

```cpp
// DataCopyExtParams 构造函数: (blockCount, blockLen_bytes, srcStride, dstStride, rsv)
DataCopyExtParams params(1, validLen * sizeof(DT_X), 0, 0, 0);
DataCopyPad(yGm[offset], yLocal, params);
DataCopyPadExtParams<DT_X> padParams{true, 0, 0, (DT_X)0};
DataCopyPad(xLocal, xGm[offset], params, padParams);
```

### 3. 动态核数约束

对小数据场景减少核数，避免核间同步开销放大：

```cpp
// 每个核至少处理一个完整 tile 的数据
coreNum = std::min(coreNum, totalBlocks / tileBlockNum);
if (coreNum == 0) coreNum = 1;
```

### 4. 大小核负载均衡

采用 big/small core 分布模式：前 `tailBlockNum` 个核为大核（多处理1个32字节块），其余为小核，确保多核负载均衡。

## 文件说明

```
src/
├── CMakeLists.txt                    # 顶层构建脚本
├── op_host/
│   ├── CMakeLists.txt                # Host侧构建
│   └── fast_gelu.cpp                 # Host侧Tiling实现（核数计算、大小核分配）
└── op_kernel/
    ├── CMakeLists.txt                # Kernel侧构建
    ├── fast_gelu.cpp                 # Kernel侧核函数（CopyIn/Compute/CopyOut）
    ├── fast_gelu_tiling.h            # Tiling数据结构体定义
    └── tiling_key_fast_gelu.h        # TilingKey数据类型模板
```

### 数据结构：FastGeluTilingData

| 字段 | 含义 |
|------|------|
| `smallCoreDataNum` | 小核处理的元素数 |
| `bigCoreDataNum` | 大核处理的元素数（比小核多1个block） |
| `tileDataNum` | 每个tile的元素数（32字节对齐） |
| `finalSmallTileNum` | 小核的tile总数 |
| `finalBigTileNum` | 大核的tile总数 |
| `smallTailDataNum` | 小核最后一个tile的元素数 |
| `bigTailDataNum` | 大核最后一个tile的元素数 |
| `tailBlockNum` | 大核数量（前N个核） |
| `totalDataNum` | 总数据元素数 |

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

// 获取workspace大小
uint64_t workspaceSize = 0;
aclOpExecutor *executor;
aclnnFastGeluGetWorkspaceSize(x, out, &workspaceSize, &executor);

// 执行
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
| 1 (小数据) | 3.44μs | 2.32μs | — |
| 2 | 4.16μs | 2.20μs | — |
| 3 | 5.70μs | 5.70μs | **达最优** |
| 4 | 6.22μs | 5.86μs | — |
| 5 (大数据) | 7.40μs | 7.16μs | — |

**总分**：73.86 / 100

msprof 分析确认：Vector 计算仅占 12% 耗时（4条向量指令、0.37μs），剩余 55% 为 Scalar 队列管理开销（1.7μs）、38% 为 DMA 搬运（1.4μs）。这是一个典型的"计算极轻、搬运主导"的算子，DMA 延迟是主要性能瓶颈。

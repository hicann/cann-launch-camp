# FastGelu 自定义算子

## 1. 算子概述

`FastGelu` 是基于 Ascend C 实现的自定义 AICore 算子，用于计算 Fast GELU 激活函数。该算子支持 `float16` 和 `float32` 输入，输出张量与输入张量保持相同的 shape 和数据类型。

Fast GELU 是 GELU 激活函数的一种高效近似形式，常用于 Transformer、BERT 等神经网络模型中，可在保持较好非线性表达能力的同时降低计算复杂度。

## 2. 计算公式

算子对输入张量 `x` 中的每个元素执行如下计算：

```text
y = x * sigmoid(1.702 * x)
```

其中：

```text
sigmoid(t) = 1 / (1 + exp(-t))
```

## 3. 算子接口

### 3.1 输入

| 参数名 | 类型 | Shape | Format | 说明 |
| --- | --- | --- | --- | --- |
| `x` | `float16` / `float32` | 任意 shape | `ND` | 输入张量 |

### 3.2 输出

| 参数名 | 类型 | Shape | Format | 说明 |
| --- | --- | --- | --- | --- |
| `y` | 与 `x` 相同 | 与 `x` 相同 | `ND` | Fast GELU 计算结果 |

### 3.3 支持约束

- 支持数据类型：`float16`、`float32`
- 支持数据格式：`ND`
- 输入和输出 shape 相同
- 输入和输出数据类型相同
- 当前注册平台配置：`ascend910b`

## 4. 文件结构

```text
.
├── CMakeLists.txt
├── op_host
│   ├── CMakeLists.txt
│   └── fast_gelu.cpp
└── op_kernel
    ├── CMakeLists.txt
    ├── fast_gelu.cpp
    ├── fast_gelu_tiling.h
    └── tiling_key_fast_gelu.h
```

## 5. Host 侧实现说明

Host 侧实现位于 `op_host/fast_gelu.cpp`，主要负责算子注册、shape 推导、数据类型推导和 Tiling 参数计算。

### 5.1 算子注册

算子名为 `FastGelu`，通过 `OP_ADD(FastGelu)` 注册。

算子定义包含：

- 1 个必选输入：`x`
- 1 个必选输出：`y`
- 输入输出均支持 `float16` 和 `float32`
- 输入输出 format 均为 `ND`
- AICore 配置目标为 `ascend910b`

### 5.2 Shape 推导

输出张量 `y` 的 shape 直接继承输入张量 `x` 的 shape：

```text
y.shape = x.shape
```

### 5.3 数据类型推导

输出张量 `y` 的数据类型与输入张量 `x` 保持一致：

```text
y.dtype = x.dtype
```

### 5.4 Tiling 逻辑

Host 侧根据输入长度、数据类型大小、AIV 核数以及 UB 大小计算 Kernel 执行所需的 Tiling 参数。

Tiling 数据结构定义在 `op_kernel/fast_gelu_tiling.h` 中：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `length` | `uint32_t` | 输入张量元素总数 |
| `blockLength` | `uint32_t` | 每个 Core 负责处理的元素数量上限 |
| `tileLength` | `uint32_t` | 每次搬运和计算的 Tile 元素数量 |

Tiling 过程主要包括：

1. 获取输入张量元素总数 `length_x`。
2. 检查输入数据类型，仅允许 `float16` 和 `float32`。
3. 根据数据类型大小计算 32 字节 block 对应的元素个数。
4. 根据输入规模和 AIV Core 数确定 `block_dim`。
5. 小规模张量长度不超过 `8192` 时，最多使用 `8` 个 Core，减少调度开销。
6. 根据 UB 大小计算 `tileLength`，并按 32 字节对齐对应的元素个数向下对齐。
7. 设置 workspace 大小为 `0`。

## 6. Kernel 侧实现说明

Kernel 侧实现位于 `op_kernel/fast_gelu.cpp`，核心类为 `KernelFastGelu<DT_X>`。

Kernel 采用按 Core 切分、按 Tile 循环处理的方式执行计算。每个 Core 只处理自己负责的连续数据片段。

### 6.1 初始化流程

`Init` 函数完成如下工作：

1. 保存 Host 侧传入的 `length`、`blockLength` 和 `tileLength`。
2. 根据当前 Core ID 计算该 Core 的全局偏移 `coreOffset`。
3. 计算当前 Core 实际需要处理的元素数量 `coreLength`。
4. 设置输入、输出 Global Memory Buffer。
5. 初始化输入队列、输出队列和临时计算 buffer。
6. 计算完整 Tile 数量 `fullTileNum` 和尾块元素数量 `tailDataNum`。
7. 判断完整 Tile 是否满足 32 字节对齐，以选择普通搬运或 Pad 搬运路径。

### 6.2 数据处理流程

`Process` 函数为 Kernel 主流程：

1. 处理完整 Tile。
2. 根据对齐情况选择：
   - `ProcessFullTilesAligned`：对齐场景，使用 `DataCopy`。
   - `ProcessFullTilesPad`：非对齐场景，使用 `DataCopyPad`。
3. 处理尾块数据 `ProcessTail`。
4. 尾块同样根据地址和长度是否 32 字节对齐，选择对齐或非对齐搬运方式。

### 6.3 计算流程

每个 Tile 的计算逻辑在 `Compute` 函数中完成：

```text
scale = x * 1.702
y = sigmoid(scale)
y = y * x
```

对应使用的 Ascend C 向量指令包括：

- `Muls`：将输入乘以常数 `1.702`
- `Sigmoid`：计算 sigmoid
- `Mul`：将 sigmoid 结果与原输入相乘

### 6.4 数据搬运策略

Kernel 中根据数据是否满足 32 字节对齐，分别使用两种搬运策略：

| 场景 | 搬入 | 搬出 |
| --- | --- | --- |
| 对齐 | `DataCopy` | `DataCopy` |
| 非对齐或尾块 | `DataCopyPad` | `DataCopyPad` |

这样可以同时支持对齐数据和非对齐尾块数据，避免因 shape 或切分导致的非 32 字节对齐问题。

## 7. 模板与数据类型选择

模板选择相关定义位于 `op_kernel/tiling_key_fast_gelu.h`。

该文件通过 `ASCENDC_TPL_ARGS_DECL` 和 `ASCENDC_TPL_SEL` 定义 `DT_X` 模板参数，使 Kernel 能够根据输入 `x` 的数据类型分别生成：

- `DT_X = float16`
- `DT_X = float32`

Host 侧通过 `ASCENDC_TPL_SEL_PARAM(context, DT_X)` 根据输入数据类型选择对应的 Kernel 模板实例。

## 8. 编译说明

项目根目录 `CMakeLists.txt` 中通过 `npu_op_package` 定义自定义算子包，并分别添加 Host 和 Kernel 子目录：

```cmake
add_subdirectory(op_host)
add_subdirectory(op_kernel)
```

通常可在支持 Ascend C 开发环境的机器上执行如下流程进行构建：

```bash
mkdir -p build
cd build
cmake ..
make
```

生成的算子包安装路径由根目录 `CMakeLists.txt` 中的配置决定：

```cmake
INSTALL_PATH ${CMAKE_BINARY_DIR}
```

## 9. 注意事项

- 输入数据类型必须为 `float16` 或 `float32`，否则 Host 侧 Tiling 会返回失败。
- 当前算子 workspace 大小为 `0`，计算过程中仅使用 UB 中的输入、输出和临时 buffer。
- 算子按一维连续内存处理输入张量，因此支持任意 shape 的 `ND` 张量。
- 对于小张量，Host 侧会限制最多使用 `8` 个 Core，以减少多核调度带来的额外开销。
- `tileLength` 会根据 UB 大小和 32 字节对齐要求动态计算。

## 10. 参考实现位置

- Host 侧实现：`op_host/fast_gelu.cpp`
- Kernel 侧实现：`op_kernel/fast_gelu.cpp`
- Tiling 数据结构：`op_kernel/fast_gelu_tiling.h`
- 模板数据类型选择：`op_kernel/tiling_key_fast_gelu.h`

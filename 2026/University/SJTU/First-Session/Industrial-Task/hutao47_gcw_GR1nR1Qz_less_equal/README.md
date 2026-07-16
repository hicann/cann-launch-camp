# LessEqual Ascend C 自定义算子核心源码

本目录保存了 `LessEqual` 自定义算子的 Host、Kernel 和 tiling 核心源码。算子运行目标为 Ascend 910B，用于逐元素判断 `x1 <= x2`，支持同形张量和符合 NumPy 规则的广播输入，输出布尔张量。

> 本目录是平铺的源码集合，不是可独立构建的完整工程。目录中没有 `CMakeLists.txt`，并且 Host 源码仍使用标准工程中的相对包含路径。使用前需要按本文的“工程集成”章节整理文件。

## 算子接口

### 计算公式

```text
y = x1 <= x2
```

### 输入与输出

| 名称 | 方向 | 支持的数据类型 | 格式 | 说明 |
| --- | --- | --- | --- | --- |
| `x1` | 输入 | `float16`、`float32`、`int32`、`int8` | `ND` | 左侧比较张量 |
| `x2` | 输入 | 与 `x1` 相同 | `ND` | 右侧比较张量 |
| `y` | 输出 | `bool` | `ND` | 每个位置的比较结果 |

约束：

- `x1` 和 `x2` 的数据类型必须一致。
- 从末维开始比较时，每一维必须相等，或者其中一个维度为 `1`。
- 广播后的张量秩不能超过 `16`，该上限由 `LE_MAX_DIM` 定义。
- 输出形状是两个输入广播后的形状。
- 空张量会直接产生空输出。

广播示例：

```text
x1 shape: [2, 1]
x2 shape: [1, 3]
y  shape: [2, 3]
```

## 文件说明

| 当前文件 | 实际职责 | 标准工程目标位置 |
| --- | --- | --- |
| `less_equal.cpp` | Host 侧算子注册、shape/type 推导和 tiling | `op_host/less_equal.cpp` |
| `less_equal (1).cpp` | Ascend C Kernel | `op_kernel/less_equal.cpp` |
| `less_equal_tiling.h` | Host 与 Kernel 共用的 tiling 数据结构 | `op_kernel/less_equal_tiling.h` |
| `tiling_key_less_equal.h` | 数据类型模板声明与选择 | `op_kernel/tiling_key_less_equal.h` |

`less_equal (1).cpp` 是复制时产生了重名后缀的 Kernel 文件。集成工程时应将其重命名为 `less_equal.cpp`，但必须放在 `op_kernel` 目录，不能覆盖 Host 侧文件。

## 实现原理

### Host 侧

Host 代码完成：

- 注册 `LessEqual` 算子及 Ascend 910B 配置。
- 校验两个输入的数据类型与广播兼容性。
- 推导广播输出形状，并将输出数据类型设置为 `bool`。
- 根据 AIV 核数、UB 容量和数据类型计算 `blockDim`、`perCore`、`tileLen`。
- 生成输出形状和输入 stride；stride 为 `0` 表示该维广播。
- 将任务分为无广播快速模式和广播模式。

### Kernel 侧

Kernel 根据 tiling 中的 `mode` 分派：

- `mode = 0`：同形快速路径，按核和 tile 连续搬运并计算。
- `mode = 1`：广播路径，按输出行计算输入基址，再加载连续向量或广播标量。

实现使用双缓冲队列、`DataCopyPad` 和尾块对齐处理。`float16` 与 `float32` 直接执行小于等于比较；`int8` 转换为 `half` 后比较；`int32` 通过 `Min(x1, x2) == x1` 实现 `x1 <= x2`。比较掩码最终转换为值为 `0` 或 `1` 的布尔输出。

## 工程集成

建议整理为以下结构：

```text
code/
|-- CMakeLists.txt
|-- op_host/
|   |-- CMakeLists.txt
|   `-- less_equal.cpp
`-- op_kernel/
    |-- CMakeLists.txt
    |-- less_equal.cpp
    |-- less_equal_tiling.h
    `-- tiling_key_less_equal.h
```

整理后，Host 文件中的以下相对包含路径才能正确解析：

```cpp
#include "../op_kernel/less_equal_tiling.h"
#include "../op_kernel/tiling_key_less_equal.h"
```

还需要由完整算子模板提供顶层、`op_host` 和 `op_kernel` 的 CMake 配置。本目录当前没有这些文件，因此不能直接在这里运行 CMake 完成构建。

## 环境要求

- Ascend 910B
- 支持 Ascend C 自定义算子开发的 CANN 环境

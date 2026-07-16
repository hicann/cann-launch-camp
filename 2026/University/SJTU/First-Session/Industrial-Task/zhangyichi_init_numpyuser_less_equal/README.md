# LessEqual 算子课题成果

## 1. 项目简介

本项目基于 Ascend C 实现 LessEqual 自定义算子，对两个输入张量 `x1` 和 `x2` 进行逐元素比较，并返回布尔类型结果：

```text
y = (x1 <= x2)
```

实现支持 `float16`、`float32`、`int32` 和 `int8` 四种输入数据类型，输出数据类型始终为 `bool`。算子兼容非 32 字节对齐的数据长度，并支持 NumPy/TensorFlow 风格的广播语义。

## 2. 算子接口

| 类型 | 名称 | 数据类型 | 数据格式 | 说明 |
| --- | --- | --- | --- | --- |
| 输入 | `x1` | `float16`、`float32`、`int32`、`int8` | ND | 第一个输入张量 |
| 输入 | `x2` | 与 `x1` 相同 | ND | 第二个输入张量 |
| 输出 | `y` | `bool` | ND | `x1 <= x2` 的逐元素比较结果 |

输入约束：

- `x1` 与 `x2` 的数据类型必须相同。
- 两个输入形状应满足 NumPy 广播规则：从最右侧维度开始比较，各维大小相等或其中一个为 `1` 时可以广播。
- 输出形状为两个输入广播后的形状。
- 当前实现支持的最大张量维数为 25。

## 3. 功能特性

- 支持 `float16`、`float32`、`int32`、`int8` 四种输入数据类型。
- 支持同形状张量逐元素比较。
- 支持标量与张量、向量与矩阵以及高维张量之间的广播。
- 支持维度长度及尾块非 32 字节对齐的场景。
- 支持零元素张量，并在 Kernel 中避免无效访存。
- 根据输入规模、数据类型、AI Core 数量和 UB 容量进行 Tiling 与多核切分。
- 使用 Ascend C 向量指令完成比较与布尔结果生成。

## 4. 目录结构

```text
zhangyichi_init_numpyuser_less_equal/
├── README.md
└── src/
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

各文件作用如下：

| 文件 | 作用 |
| --- | --- |
| `src/op_host/less_equal.cpp` | 定义算子接口、输出形状与类型推导、广播信息生成、Tiling 和多核切分逻辑 |
| `src/op_kernel/less_equal.cpp` | 实现 Kernel 侧的 CopyIn、Compute、CopyOut、广播寻址和分类型比较逻辑 |
| `src/op_kernel/less_equal_tiling.h` | 定义 Host 与 Kernel 共享的 Tiling 数据结构 |
| `src/op_kernel/tiling_key_less_equal.h` | 定义不同输入数据类型对应的 TilingKey 模板参数 |
| 各级 `CMakeLists.txt` | 配置 Host、Kernel、Tiling 及算子包的编译过程 |

## 5. 实现说明

### 5.1 Host 侧

Host 侧完成以下工作：

1. 检查两个输入的数据类型是否一致，并将输出类型设置为 `bool`。
2. 按照 NumPy 广播规则推导输出形状，同时生成输入张量的广播 stride。
3. 将可连续处理或可复用同一标量的后缀维度合并为 segment，减少 Kernel 侧逐元素坐标换算。
4. 根据数据类型和平台 UB 容量确定单次处理的 `tileLength`。
5. 根据输出长度和可用 AI Core 数量确定 `blockDim` 与每核处理长度。
6. 根据输入数据类型选择对应的 Kernel 模板实例。

### 5.2 Kernel 侧

Kernel 采用 `CopyIn -> Compute -> CopyOut` 流程：

- `CopyIn`：从 GM 搬运输入数据到 UB；非对齐尾块使用 `DataCopyPad`；广播场景根据输出坐标和 stride 计算输入偏移。
- `Compute`：使用向量指令生成比较 mask，再转换为 `bool` 输出。
- `CopyOut`：将结果从 UB 搬运回 GM；非对齐尾块采用扩展搬运接口。

不同数据类型的计算路径为：

- `float16`、`float32`：使用向量比较指令执行 `<=`；标量广播场景优先使用标量比较接口。
- `int32`：利用 `min(x1, x2) == x1` 与 `x1 <= x2` 等价的关系，在保持 `int32` 精度的前提下完成比较。
- `int8`：先精确转换为 `half`，再执行向量比较。

性能方面，代码通过连续数据快速路径、多核切分、广播 segment 合并、标量复用、UB 缓冲复用以及 `float32` 双缓冲，降低访存和坐标计算开销。

## 6. 编译方法

### 6.1 环境要求

- Linux 操作系统
- 支持 Ascend C 自定义算子开发的 CANN Toolkit
- Ascend 910B 计算单元
- CMake 3.16.0 或更高版本
- 训练营提供的算子编译环境及相关依赖

### 6.2 编译步骤

首先加载 CANN 环境变量。不同服务器上的实际安装路径可能不同，以下为常见路径：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

在个人提交目录下执行：

```bash
mkdir -p build
cd build
cmake ../src
cmake --build . -j
```

如果训练营服务器已经自动配置 CANN 环境，可省略环境变量加载步骤。编译生成的 `build/` 目录属于本地构建产物，无需提交到代码仓库。

## 7. 运行与精度验证

本项目为算子源码工程，可在训练营提供的评测工程中完成算子编译、部署和调用。运行时需传入两个数据类型相同且形状满足广播规则的张量，输出为广播后形状对应的布尔张量。

精度验证以 TensorFlow `tf.math.less_equal` 的输出作为参考结果：

```python
import tensorflow as tf

x1 = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
x2 = tf.constant([2, 3], dtype=tf.float32)
reference = tf.math.less_equal(x1, x2)
print(reference)
```

参考输出：

```text
[[ True  True]
 [False False]]
```

比较运算要求结果完全一致，不设置数值误差容限。当前提交版本已在 CANN 训练营服务器完成编译，并通过训练营评测平台的精度测试与性能测试。

## 8. 注意事项

- 不支持两个输入使用不同的数据类型。
- 不满足广播规则的输入形状会在形状推导或 Tiling 阶段返回失败。
- 请勿将 `build/`、编译日志、二进制文件或其他临时文件提交到代码仓库。
- 若在其他服务器上编译，请根据实际 CANN 安装位置调整环境变量脚本路径。

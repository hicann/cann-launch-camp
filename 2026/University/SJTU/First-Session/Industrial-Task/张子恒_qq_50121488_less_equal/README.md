# LessEqual Ascend C 自定义算子

本项目基于 Ascend C 实现 TensorFlow `tf.math.less_equal` 对应的 LessEqual 自定义算子，目标运行平台为 Atlas 训练/推理产品（`ascend910b`）。算子逐元素判断 `x1 <= x2`，输出 `bool` 张量。

## 功能特性

- 支持 `float16`、`float32`、`int32`、`int8` 输入。
- 两个输入的数据类型必须一致，输出固定为 `bool`。
- 支持 ND 格式、非 32 字节对齐形状和空张量。
- 支持 NumPy/TensorFlow 标准广播，包括标量广播、向量与矩阵广播及高维广播。
- 连续输入使用 DMA 搬运；常见广播按连续尾部 span 分段 DMA。
- 浮点比较使用 `Compare`，`int32` 使用 `Min + Compare(EQ)`，普通 `int8` 输入先无损转换为 `half` 再比较。
- 比较产生的压缩 bitmask 通过 `Select + Cast` 向量生成 bool 字节。

## 目录结构

```text
.
|-- README.md
|-- code
|   |-- CMakeLists.txt
|   |-- op_host
|   |   |-- CMakeLists.txt
|   |   `-- less_equal.cpp
|   `-- op_kernel
|       |-- CMakeLists.txt
|       |-- less_equal.cpp
|       |-- less_equal_tiling.h
|       `-- tiling_key_less_equal.h
`-- scripts
    |-- build.sh
    |-- clean.sh
    `-- check_submission.sh
```

## 实现架构

Kernel 保持标准 Ascend C 算子结构：

1. `Init`：解析 tiling，划分当前 AI Core 的输出区间，初始化 GM Tensor、UB 队列和计算缓冲区。
2. `Process`：按 tile 循环执行 `CopyIn -> Compute -> CopyOut`。
3. `CopyIn`：连续输入使用 `DataCopyPad`；广播输入优先按连续 span 分段搬运，完全离散场景使用正确性回退。
4. `Compute`：根据输入 dtype 选择对应的向量比较方案，并将压缩 mask 转换为 bool 字节。
5. `CopyOut`：使用 `DataCopyPad` 将有效结果写回 GM，兼容非对齐尾块。

Host 侧负责 Shape/DType 推导、广播 stride/span 计算、UB 容量估算、核数与 tile 大小配置。

## 构建环境

- CANN 8.5.0 或与题目环境兼容的版本
- Atlas `ascend910b`
- CMake 3.16 及以上
- Linux 构建环境
- 已安装并配置 Ascend Toolkit

构建前应加载 CANN 环境，例如：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

实际路径请以本机 CANN 安装位置为准，也可以通过 `ASCEND_HOME_PATH` 指定 Toolkit 目录。

## 构建方法

```bash
bash scripts/check_submission.sh
bash scripts/build.sh
```

默认构建目录为项目根目录下的 `build/`。可以通过环境变量调整：

```bash
BUILD_DIR=/tmp/less_equal_build BUILD_TYPE=Release JOBS=8 bash scripts/build.sh
```

清理构建产物：

```bash
bash scripts/clean.sh
```

## 使用说明

构建完成后，CMake 工程会生成自定义算子包、Host tiling 动态库、自动生成的 ACLNN 接口及 Kernel 动态库。具体安装和调用方式取决于当前 CANN 环境生成的包结构；评测平台可直接使用 `code/` 目录构建。

本仓库不包含独立的输入数据运行器。算子运行需要安装生成的自定义算子包，并通过框架或自动生成的 ACLNN 接口调用。

## 已知边界

- 完全离散且无法形成连续 span 的复杂广播仍保留逐元素收集路径。
- 当前 tiling 最多保存 64 维 Shape/Stride 信息。
- 张量总元素数使用 `uint32_t` 表示。
- `int8` 标量与张量广播因 AI Core 不支持运行时 `int8` 标量直接转换为 `half`，保留标量比较分支。


# GELU 自定义算子

本工程实现了 CANNJudge GELU 题目的自定义算子。

## 工程结构

```text
.
├── README.md
└── code
    ├── CMakeLists.txt
    ├── op_host
    │   ├── CMakeLists.txt
    │   └── gelu.cpp
    └── op_kernel
        ├── CMakeLists.txt
        ├── gelu.cpp
        ├── gelu_tiling.h
        └── tiling_key_gelu.h
```

## 源码说明

- `code/op_kernel/gelu.cpp`：Ascend C Kernel 侧实现。
- `code/op_host/gelu.cpp`：Host 侧 shape 推导、dtype 推导和 tiling 实现。
- `code/op_kernel/gelu_tiling.h`：Host 侧和 Kernel 侧共享的 tiling 数据结构。
- `code/op_kernel/tiling_key_gelu.h`：tiling key 相关头文件。

## 算子功能

本算子按元素计算 GELU：

```text
GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
```

支持的数据类型：

- `float16`
- `float32`

支持的数据格式：

- `ND`

输出 tensor 的 shape 和 dtype 与输入 tensor 保持一致。

## 实现说明

- `float16` 路径使用 tanh 形式的 GELU 近似：

```text
0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
```

- `float32` 路径使用 erf 形式 GELU 的多项式近似实现。
- `float32` 路径使用 degree-9 多项式，并移除了多项式路径中的正向 clamp。
- Host 侧 tiling 对小 shape 做了限核处理，避免过度并行带来的额外开销：

```text
float16: 每个核至少处理 512 个元素
float32: 每个核至少处理 256 个元素
```

## 构建方式

本工程需要在已配置 CANN 自定义算子开发环境的机器上构建。

典型构建命令如下：

```bash
cd code
mkdir -p build
cd build
cmake ..
make -j
```

工程的 CMake 配置使用：

```text
find_package(ASC REQUIRED)
```

目标硬件配置为：

```text
ascend910b
```

## 提交说明

提交到 CANNJudge 时，请按照平台要求提交工程源码。

本工程需要包含：

- `code/` 目录下的可运行源码
- `README.md` 说明文档
- `code/` 目录下的 CMake 构建文件

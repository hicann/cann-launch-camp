# GELU Ascend C Custom Operator

## 目录说明

本目录为 GELU 自定义算子提交目录，包含可运行源码、算子 host/kernel 实现以及构建配置。

```text
GELU/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── CMakeLists.txt
│   └── gelu.cpp
└── op_kernel/
    ├── CMakeLists.txt
    ├── gelu.cpp
    ├── gelu_tiling.h
    └── tiling_key_gelu.h
```

## 算子功能

实现 PyTorch 默认精度模式下的 GELU：

```text
GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
```

支持数据类型：

- float32
- float16

输出张量形状和数据类型与输入保持一致。

## 实现说明

- host 侧在 `op_host/gelu.cpp` 中完成 shape/type 推导和 tiling 参数计算。
- kernel 侧在 `op_kernel/gelu.cpp` 中完成逐元素 GELU 计算。
- 支持 32 字节对齐场景和非 32 字节对齐输入。
- float32 路径使用精确 erf 公式计算。
- float16 路径转换到 float 计算后再转换回 float16。

## 构建说明

使用目录中的 `CMakeLists.txt` 作为构建入口，并按平台提供的 Ascend C/CANN 算子工程构建流程编译。

## 提交说明

提交时请直接提交本 `GELU` 目录，目录中已包含：

- 可运行源码
- `op_host` 文件夹
- `op_kernel` 文件夹
- README 文档
- 必要的 CMake 构建文件

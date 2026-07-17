# LessEqual Ascend C 算子

## 功能说明

本项目基于 Ascend C 实现 `LessEqual` 算子，对两个输入张量逐元素执行
`x1 <= x2` 比较，并输出 `bool` 张量。

实现支持：

- `float16`、`float32`、`int32` 和 `int8` 输入；
- NumPy/TensorFlow 风格广播；
- 标量广播、连续张量和通用高维广播；
- 非 32 字节对齐的输入输出；
- 多核并行和向量化比较。

## 目录结构

```text
.
├── README.md
├── build.sh
└── code
    ├── CMakeLists.txt
    ├── op_host
    │   ├── CMakeLists.txt
    │   └── less_equal.cpp
    └── op_kernel
        ├── CMakeLists.txt
        ├── less_equal.cpp
        ├── less_equal_tiling.h
        └── tiling_key_less_equal.h
```

## 实现概要

Host 侧完成输出形状推导、输入类型检查、广播 stride 计算、执行模式选择和多核
任务划分。Kernel 侧针对同形状、单侧标量广播和通用广播分别处理，并使用 Ascend C
向量比较接口生成布尔结果。输入输出尾块通过非对齐搬运接口处理，避免越界访问。

## 构建环境

- CANN 8.5.0
- Ascend 910B
- CMake 3.16 或更高版本
- 支持 Bash 的 Linux 环境

## 构建方法

已安装 CANN Toolkit 且环境位于默认路径时执行：

```bash
chmod +x build.sh
./build.sh
```

如 CANN 环境脚本不在默认路径，可显式指定：

```bash
CANN_ENV=/path/to/ascend-toolkit/set_env.sh ./build.sh
```

可通过环境变量调整构建目录、构建类型和并行数：

```bash
BUILD_DIR=build-release BUILD_TYPE=Release JOBS=8 ./build.sh
```

构建日志保存在 `${BUILD_DIR}/build.log`。脚本会在构建失败或日志中出现内核编译错误时
返回非零状态。

## 验证结果

实现已通过 CANNJudge 的五个公开测试点，所有测试点精度均为 1。保留版本的多次性能
测试结果如下，单位为微秒：

```text
4.30  3.62  12.20  4.20  16.80
4.00  3.84  13.04  4.46  17.22
4.14  3.84  12.02  4.30  17.28
3.46  5.12  11.84  3.34  18.32
```

由于共享评测环境存在波动，性能数据以多次运行结果综合评估。

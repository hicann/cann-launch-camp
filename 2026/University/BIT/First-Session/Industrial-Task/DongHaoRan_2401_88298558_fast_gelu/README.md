# FastGelu 自定义算子提交说明

## 1. 项目简介

本项目实现了 Ascend C 自定义算子 `FastGelu`，用于计算 FastGELU 激活函数：

```text
FastGelu(x) = x / (1 + exp(-1.702 * x))
```

算子支持 `float16` 和 `float32` 输入，输出数据类型与输入保持一致，主要面向 `ascend910b` 平台。

## 2. 目录结构

```text
.
├── README.md
└── code
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

其中：

- `code/op_host/fast_gelu.cpp`：Host 侧算子注册、shape/type 推导与 tiling 计算。
- `code/op_kernel/fast_gelu.cpp`：Ascend C Kernel 侧 FastGelu 计算实现。
- `code/op_kernel/fast_gelu_tiling.h`：Host 与 Kernel 共享的 tiling 数据结构。
- `code/op_kernel/tiling_key_fast_gelu.h`：tiling key 相关定义。
- `code/CMakeLists.txt`：自定义算子工程构建入口。

## 3. 实现思路

本实现采用 Host 侧切分与 Kernel 侧分块计算的方式完成 FastGelu：

1. Host 侧根据输入长度、数据类型和 AIV 核数计算 `blockDim` 与 `blockLength`。
2. 对小张量限制有效核数，减少过度并行带来的调度开销。
3. `blockLength` 按 32B 对齐，保证各核起始地址尽量满足高效搬运要求。
4. Kernel 侧使用双缓冲队列进行 GM 与 UB 之间的数据搬运。
5. 对完整 tile 与 tail 数据分别处理，完整 tile 走普通 `DataCopy`，尾块根据长度选择对齐搬运或 `DataCopyPad`。
6. `float32` 路径按公式直接计算；`float16` 路径直接使用 half 向量计算，以减少 Cast 和临时 buffer 开销。

## 4. 算子规格

| 项目 | 内容 |
|---|---|
| 算子名称 | `FastGelu` |
| 输入 | `x` |
| 输出 | `y` |
| 支持数据类型 | `float16`、`float32` |
| 支持格式 | `ND` |
| 目标平台 | `ascend910b` |
| 计算公式 | `y = x / (1 + exp(-1.702 * x))` |

## 5. 编译方法

进入工程根目录后执行：

```bash
cd code
mkdir -p build
cd build
cmake ..
make -j
```

如果环境中未加载 CANN 工具链，需要先执行类似命令：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

具体路径以实际安装环境为准。

## 6. 提交说明

提交时请保证个人目录中至少包含以下内容：

```text
姓名_gitcode账号_fast_gelu
├── README.md
└── code
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

## 7. 注意事项

- 请不要提交 `build/`、`.o`、`.so`、缓存文件等编译中间产物。
- 请确保 `README.md`、Host 代码、Kernel 代码和 CMake 构建文件均位于提交目录内。
- 若平台提供统一测试脚本，应以平台测试结果为准。
- 本实现未使用 `FasterGelu`、`FasterGeluV2` 或 `Tanh` 替代计算，保持 FastGelu 公式实现。

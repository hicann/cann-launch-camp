# GELU 自定义算子

这是一个基于 Ascend C / CANN 的 GELU 自定义算子工程，包含 host 侧算子注册与 tiling 逻辑，以及 device 侧 AICore kernel 实现。

## 功能说明

本工程实现了一个名为 `Gelu` 的自定义算子，输入为 `input_x`，输出为 `output`。算子支持 `float16` 和 `float32` 两种数据类型，输出 shape 和数据类型与输入保持一致。

device 侧 kernel 会根据 host 侧计算得到的 tiling 参数，将输入数据按 core 和 tile 进行切分处理，并在 AICore 上完成 GELU 计算。

## 目录结构

```text
code/
├── CMakeLists.txt              # 工程主构建脚本
├── op_host/
│   ├── CMakeLists.txt          # host 侧构建脚本
│   └── gelu.cpp                # 算子注册、shape/type 推导和 tiling 计算
└── op_kernel/
    ├── CMakeLists.txt          # device 侧 kernel 构建脚本
    ├── gelu.cpp                # AICore kernel 实现
    ├── gelu_tiling.h           # host/device 共用的 tiling 数据结构
    └── tiling_key_gelu.h       # kernel 模板参数声明
```

## 核心文件说明

### `code/op_host/gelu.cpp`

host 侧主要完成以下工作：

- 注册 `Gelu` 算子的输入、输出、数据类型和格式。
- 设置 shape 推导函数，输出 shape 与输入 shape 保持一致。
- 设置数据类型推导函数，输出数据类型与输入数据类型保持一致。
- 根据输入规模、数据类型长度、AICore 数量和 UB 空间计算 tiling 参数。
- 将 tiling 参数写入 `GeluTilingData`，供 device 侧 kernel 使用。

### `code/op_kernel/gelu.cpp`

device 侧主要完成以下工作：

- 根据当前 block 索引确定每个 core 负责的数据范围。
- 按 tile 将数据从 GM 搬运到本地 buffer。
- 对每个 tile 执行 GELU 向量计算。
- 将计算结果从本地 buffer 写回 GM。

其中 `float16` 路径使用 sigmoid 形式的近似计算，`float32` 路径使用 erf 形式计算。

### `code/op_kernel/gelu_tiling.h`

定义 host 侧传递给 device 侧的 tiling 参数，包括：

- `totalDataNum`：总数据量。
- `smallCoreDataNum`：普通 core 处理的数据量。
- `bigCoreDataNum`：多分配一个 block 的 core 处理的数据量。
- `tileDataNum`：每个 tile 处理的数据量。
- `tailCoreNum`：需要处理更多数据的 core 数量。

### `code/op_kernel/tiling_key_gelu.h`

声明 `Gelu` 算子 kernel 支持的数据类型模板参数，目前支持：

- `float16`
- `float32`

## 构建说明

工程使用 CMake 和 CANN 自定义算子构建接口。构建前需要确保 Ascend CANN 环境已经正确安装，并且相关环境变量已经配置完成。

典型构建流程如下：

```bash
cd code
mkdir -p build
cd build
cmake ..
make
```

构建完成后，自定义算子包会输出到构建目录中，具体路径由 `CMakeLists.txt` 中的 `INSTALL_PATH ${CMAKE_BINARY_DIR}` 决定。

## 算子限制

- 当前算子配置面向 `ascend910b`。
- 输入格式为 `ND`。
- 支持数据类型为 `float16` 和 `float32`。
- workspace 大小为 0。

## 备注

本工程代码结构较简洁，主要用于演示或实现 GELU 自定义算子的 host 注册、tiling 计算和 AICore kernel 计算流程。后续如果需要扩展更多数据类型、更多 shape 场景或其他 Ascend 芯片配置，可以在 host 侧注册信息、tiling 逻辑和 kernel 模板参数中继续补充。

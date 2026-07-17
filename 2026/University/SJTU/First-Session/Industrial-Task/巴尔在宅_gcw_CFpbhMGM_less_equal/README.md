# LessEqual Ascend C 自定义算子

本工程实现了运行于 Ascend 910B 的 `LessEqual` 自定义算子。算子逐元素计算 `x1 <= x2`，支持形状相同的直接比较和符合 NumPy 规则的广播比较，输出布尔张量。

## 算子说明

### 计算公式

```text
y[i] = (x1[i] <= x2[i])
```

发生广播时，输入索引会按照输出形状和各输入的广播步长进行映射。

### 输入与输出

| 名称 | 类型 | 数据类型 | 数据格式 | 说明 |
| --- | --- | --- | --- | --- |
| `x1` | 输入 | `float16`、`float32`、`int32`、`int8` | `ND` | 第一个比较张量 |
| `x2` | 输入 | 与 `x1` 相同 | `ND` | 第二个比较张量 |
| `y` | 输出 | `bool` | `ND` | `x1 <= x2` 的逐元素结果 |

约束：

- `x1` 和 `x2` 的数据类型必须一致。
- 两个输入从末维向前比较时，每一维必须相等，或者其中一维为 `1`。
- 当前 tiling 数据最多记录 8 个维度，即输入广播后的秩不能超过 `8`。
- 空张量会生成空输出，不启动实际计算。

广播示例：

```text
x1 shape: [2, 1]
x2 shape: [1, 3]
y  shape: [2, 3]
```

## 实现概览

### Host 侧

`op_host/less_equal.cpp` 完成以下工作：

- 注册 `LessEqual` 算子、输入输出类型和 `ND` 格式。
- 推导广播后的输出形状，并将输出类型设置为 `bool`。
- 校验输入类型、广播兼容性、维度和元素数量。
- 根据平台 AIV 核数与 UB 大小计算 `blockDim`、`perCore` 和 `tileLen`。
- 为广播维生成步长；步长为 `0` 表示该维需要广播。
- 生成两种执行模式的 tiling 数据。

### Kernel 侧

`op_kernel/less_equal.cpp` 根据 tiling 的 `mode` 选择执行路径：

- `mode = 0`：无广播快速路径。数据按核、按 tile 切分后直接进行向量计算。
- `mode = 1`：广播路径。按输出行计算输入基址，并依据广播步长加载标量或连续向量。

Kernel 使用双缓冲队列搬运输入和输出，处理非对齐尾块，并将比较产生的掩码转换为值为 `0` 或 `1` 的布尔输出。针对 `float16`、`float32`、`int32` 和 `int8` 使用对应的模板实例与计算缓冲区。

## 目录结构

```text
code/
|-- CMakeLists.txt
|-- README.md
|-- op_host/
|   |-- CMakeLists.txt
|   `-- less_equal.cpp
`-- op_kernel/
    |-- CMakeLists.txt
    |-- less_equal.cpp
    |-- less_equal_tiling.h
    `-- tiling_key_less_equal.h
```

主要文件：

- `op_host/less_equal.cpp`：算子注册、shape/type 推导和 tiling 计算。
- `op_kernel/less_equal.cpp`：Ascend C Kernel 与广播/非广播计算逻辑。
- `op_kernel/less_equal_tiling.h`：Host 与 Kernel 共用的 tiling 数据结构。
- `op_kernel/tiling_key_less_equal.h`：四种输入数据类型的 tiling 模板声明与选择。

## 环境要求

- Ascend 910B
- 支持 Ascend C 自定义算子开发的 CANN 环境
- CMake 3.16 或更高版本
- 已正确设置 CANN 工具链环境变量，并能通过 `find_package(ASC REQUIRED)` 找到 ASC CMake 包

例如，在 CANN 安装目录对应的环境脚本存在时先执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

实际路径请以当前开发环境中的 CANN 安装位置为准。

## 构建

在本目录执行：

```bash
cmake -S . -B build
cmake --build build -j
```

顶层 `CMakeLists.txt` 会构建并打包：

- Host 侧 tiling 动态库 `cust_optiling`
- ACLNN 调用动态库 `cust_opapi`
- Kernel 动态库 `ascendc_kernels`
- 名为 `custom` 的自定义算子包

构建产物安装位置由工程配置为 `build` 目录。具体文件名和子目录可能随 CANN 版本及构建环境变化，请以 CMake 构建输出为准。

## 使用注意事项

- 本目录只包含算子实现与构建配置，不包含独立测试程序或测试数据。
- 修改 `LessEqualTilingData` 时，必须同步检查 Host 侧赋值与 Kernel 侧读取，保持字段布局一致。
- 修改支持的数据类型时，需要同时更新算子注册、tiling 模板选择和 Kernel 模板实现。
- Windows 路径下的源码需要复制或上传到配置好 CANN/Ascend 910B 的 Linux 环境后再进行编译和设备验证。

## 当前验证范围

README 根据本目录当前源码与 CMake 配置生成。最终可用性仍应在目标 CANN 环境中完成编译，并在 Ascend 910B 设备上进行精度验证。

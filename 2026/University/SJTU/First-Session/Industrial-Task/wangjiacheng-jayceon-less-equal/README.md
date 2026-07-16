# LessEqual Ascend C 自定义算子

本目录实现了面向 **Ascend 910B** 的 `LessEqual` 自定义算子。算子对两个输入张量执行逐元素“小于等于”比较，并输出布尔张量：

```text
y[i] = (x1[i] <= x2[i])
```

除相同形状的逐元素比较外，算子还支持符合 NumPy 规则的广播。例如：

```text
x1.shape = [2, 3]
x2.shape = [3]
y.shape  = [2, 3]
```

> 本仓库只提交算子源码与构建配置，不提交 `build/` 目录。`build/` 中的动态库、目标文件、自动生成代码及 CMake 缓存均应在本地重新生成。

## 1. 功能特性

- 算子名称：`LessEqual`
- 目标芯片：`Ascend 910B`
- 执行单元：Ascend AI Core
- 输入数量：2
- 输出数量：1
- 输入、输出格式：`ND`
- 最大支持维度：16 维
- 支持动态形状
- 支持标量广播、行广播和通用多维广播
- 两个输入的数据类型必须一致
- 输出数据类型固定为 `bool`

### 支持的数据类型

| 输入 `x1` | 输入 `x2` | 输出 `y` |
|---|---|---|
| `float16` | `float16` | `bool` |
| `float32` | `float32` | `bool` |
| `int32` | `int32` | `bool` |
| `int8` | `int8` | `bool` |

## 2. 仓库目录结构

```text
wangjiacheng_@Jayceon_less_equal/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── CMakeLists.txt
│   └── less_equal.cpp
└── op_kernel/
    ├── CMakeLists.txt
    ├── less_equal.cpp
    ├── less_equal.cpp.orig
    ├── less_equal_tiling.h
    └── tiling_key_less_equal.h
```

各文件作用如下：

| 文件 | 说明 |
|---|---|
| `CMakeLists.txt` | 工程构建入口，指定 `ascend910b` 并组织 Host 与 Kernel 编译 |
| `op_host/less_equal.cpp` | 算子注册、输出形状推导、数据类型推导、广播分析和 Tiling 配置 |
| `op_kernel/less_equal.cpp` | 在 AI Core 上执行数据搬运、向量比较和结果回写 |
| `op_kernel/less_equal_tiling.h` | 定义 Host 与 Kernel 共享的 Tiling 数据结构和执行模式 |
| `op_kernel/tiling_key_less_equal.h` | 定义 `float16`、`float32`、`int32` 和 `int8` 的模板实例 |
| `op_kernel/less_equal.cpp.orig` | Kernel 源码的历史备份，不参与当前核心逻辑说明 |

编译后会自动生成 `build/`，但该目录不属于源码仓库。

## 3. 算子执行流程

整个算子的执行过程分为 Host 侧准备和 Kernel 侧计算两部分：

```text
输入 x1、x2
    │
    ▼
Host 侧检查数据类型并推导广播后的输出形状
    │
    ▼
Host 侧计算广播步长、输出元素数、执行模式和 AI Core 数量
    │
    ▼
生成 TilingData 并下发至 AI Core
    │
    ▼
Kernel 将输入数据从 Global Memory 搬入 Local Memory
    │
    ▼
执行 x1 <= x2 的向量化比较
    │
    ▼
将比较结果转换为 bool 并写回 Global Memory
    │
    ▼
输出 y
```

### 3.1 Host 侧

`op_host/less_equal.cpp` 主要负责：

1. 从最后一维开始对齐两个输入形状，推导广播后的输出形状；
2. 检查对应维度是否满足“相等或其中一维为 1”的广播规则；
3. 检查两个输入的数据类型是否一致，并将输出类型设置为 `bool`；
4. 为广播维度设置步长 0，使 Kernel 重复读取同一输入位置；
5. 根据输入形状选择合适的 Tiling 模式；
6. 根据输出元素数量和硬件可用 AIV Core 数量确定实际核数；
7. 将形状、步长、核数、Tile 大小和执行模式写入 `LessEqualTilingData`。

### 3.2 Kernel 侧

`op_kernel/less_equal.cpp` 使用 Ascend C 流水线完成计算：

```text
CopyIn → Compute → CopyOut
```

- `CopyIn`：将输入数据从 Global Memory 搬运到片上 Local Memory；
- `Compute`：执行向量化小于等于比较；
- `CopyOut`：将 0/1 形式的布尔结果写回 Global Memory；
- 单个 Tile 最多处理 `8192` 个元素；
- 多核任务按照约 `256` 个元素的粒度对齐切分。

不同数据类型采用不同实现：

- `float16`、`float32`：直接调用 `Compare(..., CMPMODE::LE)`；
- `int8`：先转换为 `half`，再执行小于等于比较；
- `int32`：先计算 `min(x1, x2)`，再判断结果是否等于 `x1`，从而实现 `x1 <= x2`。

## 4. 广播与 Tiling 模式

工程针对不同输入形状定义了 8 种执行模式：

| 模式 | 说明 |
|---|---|
| `FLAT` | 两个输入与输出元素数量一致，直接逐元素比较 |
| `X1_SCALAR` | `x1` 为单元素标量，对整个 `x2` 广播 |
| `X2_SCALAR` | `x2` 为单元素标量，对整个 `x1` 广播 |
| `GENERAL` | 通用多维广播，通过输出坐标计算两个输入偏移 |
| `X1_ROW` | `x1` 为末维行向量，采用快速行广播 |
| `X2_ROW` | `x2` 为末维行向量，采用快速行广播 |
| `X1_REPEAT` | `x1` 按末维重复，但不满足快速行广播条件 |
| `X2_REPEAT` | `x2` 按末维重复，但不满足快速行广播条件 |

常见的相同形状、标量广播和行向量广播会进入专用快速路径；其他合法形状由通用广播路径处理。

## 5. 环境要求

编译和运行本工程需要：

- Ascend 910B 硬件环境；
- 已安装支持 Ascend C 自定义算子开发的 CANN Toolkit；
- CMake 3.16 或更高版本；
- C/C++ 编译环境；
- 已正确加载 Ascend Toolkit 环境变量；
- 当前用户具有访问 NPU 设备的权限。

CANN 的安装目录与版本可能因机器而异，应以本机实际环境为准。

## 6. 从源码编译

### 6.1 加载 CANN 环境

常见安装方式下可执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

如 CANN 安装在其他路径，请改为本机对应的 `set_env.sh`。

可以检查关键环境变量：

```bash
echo "$ASCEND_HOME_PATH"
echo "$ASCEND_CANN_PACKAGE_PATH"
```

若环境脚本没有设置 `ASCEND_CANN_PACKAGE_PATH`，可以手动指定：

```bash
export ASCEND_CANN_PACKAGE_PATH=/path/to/Ascend/ascend-toolkit/latest
```

### 6.2 配置工程

进入当前算子目录后执行：

```bash
cmake -S . -B build \
  -DASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH}"
```

若本机 CANN 环境已经能够被 CMake 自动识别，也可以直接执行：

```bash
cmake -S . -B build
```

### 6.3 编译算子

```bash
cmake --build build -j"$(nproc)"
```

只构建自定义算子主目标时可使用：

```bash
cmake --build build --target custom -j"$(nproc)"
```

### 6.4 主要编译产物

成功编译后，相关文件通常位于：

```text
build/libcust_opapi.so
build/lib/libcust_opapi.so
build/include/aclnn_less_equal.h
build/op_kernel/ascendc_kernels/
```

这些文件均由构建过程生成，不应提交到 Git 仓库。

### 6.5 清理构建结果

需要完全重新配置工程时，直接删除构建目录：

```bash
rm -rf build
```

随后重新执行 CMake 配置和编译命令即可。

## 7. ACLNN 接口

编译过程中会生成 ACLNN 接口头文件：

```text
build/include/aclnn_less_equal.h
```

典型调用分为两步：

```cpp
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;

aclnnLessEqualGetWorkspaceSize(
    x1Tensor,
    x2Tensor,
    outputTensor,
    &workspaceSize,
    &executor);

aclnnLessEqual(
    workspace,
    workspaceSize,
    executor,
    stream);
```

实际调用时还需要完成 ACL 初始化、设备与 Stream 创建、输入输出内存申请、数据拷贝以及资源释放。

## 8. Git 提交建议

`build/` 包含本机路径、CMake 缓存、自动生成代码、动态库和目标文件，不适合进入版本控制。建议在仓库的 `.gitignore` 中加入：

```gitignore
# CMake and local build outputs
build/
cmake-build-*/
CMakeFiles/
CMakeCache.txt
cmake_install.cmake
Makefile

# Compiled files
*.o
*.a
*.so
*.log

# Editor and system files
.vscode/
.idea/
.DS_Store
```

提交前可检查：

```bash
git status
```

若 `build/` 之前已经被 Git 跟踪，仅添加 `.gitignore` 不会自动取消跟踪，需要执行：

```bash
git rm -r --cached build
```

然后再提交变更。

## 9. 关键参数

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `LESS_EQUAL_MAX_DIMS` | 16 | 最大支持维度 |
| `TILE_LENGTH` | 8192 | 单个 Tile 最大处理元素数 |
| `MIN_ELEMENTS_PER_CORE` | 32768 | Host 侧估算有效核数的参考粒度 |
| `coreAlignment` | 256 | Kernel 多核切分时的元素对齐粒度 |

## 10. 使用限制

- 两个输入的数据类型必须完全一致；
- 当前仅注册 `float16`、`float32`、`int32` 和 `int8`；
- 当前目标芯片配置为 `ascend910b`；
- 输入形状必须满足广播规则；
- 最大输出维度为 16；
- 输出 `bool` 在 Kernel 中以 `uint8_t` 形式存储，每个元素使用一个字节表示 0 或 1。

合法广播示例：

```text
[2, 3] 与 [3]       → [2, 3]
[4, 1, 8] 与 [1, 5, 8] → [4, 5, 8]
[] 与 [2, 3]        → [2, 3]
```

非法广播示例：

```text
[2, 3] 与 [2, 2]
```

## 11. 常见问题

### 11.1 CMake 找不到 `ASC`

确认已经加载 CANN 环境，并检查：

```bash
echo "$ASCEND_HOME_PATH"
echo "$ASCEND_CANN_PACKAGE_PATH"
```

必要时通过 `-DASCEND_CANN_PACKAGE_PATH=...` 显式指定 CANN 路径。

### 11.2 运行时找不到 `libcust_opapi.so`

将编译输出目录加入动态库搜索路径：

```bash
export LD_LIBRARY_PATH="$PWD/build/lib:$PWD/build:${LD_LIBRARY_PATH}"
```

### 11.3 为什么仓库中没有 `build/`

`build/` 是本地生成目录，其中包含与操作系统、CPU 架构、CANN 安装路径和当前源码绝对路径相关的文件。其他开发者克隆仓库后应在自己的环境中重新生成该目录。

### 11.4 修改源码后如何重新编译

一般可直接执行：

```bash
cmake --build build -j"$(nproc)"
```

如果修改了 CMake 配置、切换了 CANN 环境或出现缓存路径问题，建议删除 `build/` 后重新配置。

## 12. 推荐阅读顺序

对于不熟悉 C++ 或 Ascend C 的读者，建议按以下顺序理解工程：

1. 阅读本 README 中的“算子执行流程”；
2. 查看 `op_host/less_equal.cpp` 中的 `class LessEqual`，了解输入、输出和芯片注册；
3. 查看 `InferShape` 与 `InferDataType`，理解形状和类型推导；
4. 查看 `TilingFunc`，理解广播模式选择和多核配置；
5. 查看 `op_kernel/less_equal.cpp` 中的 `Process()`，理解不同执行模式的分流；
6. 查看 `ComputeVector()`，理解各数据类型的比较实现；
7. 最后查看 `CopyIn`、`CopyOut` 和广播偏移计算细节。

---

本工程通过 Host 侧广播推导与 Tiling 策略，将不同形状和数据类型的 `LessEqual` 操作映射到 Ascend 910B AI Core 上执行，并针对常见广播场景设计了专用计算路径。

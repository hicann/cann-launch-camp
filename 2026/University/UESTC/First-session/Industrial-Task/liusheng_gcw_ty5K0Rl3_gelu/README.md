# Custom Gelu AscendC Operator

本项目实现了一个面向 Ascend 910B 的自定义 `Gelu` 算子，包含 Host 侧算子注册与 tiling 逻辑、Device 侧 AscendC kernel，以及 CMake 构建配置。

## 项目结构

```text
.
├── CMakeLists.txt
├── op_host/
│   ├── CMakeLists.txt
│   └── gelu.cpp
├── op_kernel/
│   ├── CMakeLists.txt
│   ├── gelu.cpp
│   ├── gelu_tiling.h
│   └── tiling_key_gelu.h
└── scripts/
    ├── build.sh
    └── run_smoke.sh
```

## 功能说明

- 算子名称：`Gelu`
- 支持数据类型：`float16`、`float32`
- 支持格式：`ND`
- 输出形状：与输入 `self` 保持一致
- 目标硬件：`ascend910b`

Device 侧实现中，`float32` 路径使用基于 `erf` 的 GELU 公式，`float16` 路径使用近似计算公式以提升执行效率。Host 侧根据输入规模、数据类型、UB 大小和核心数计算 tiling 参数，并选择对应模板实例。

## 环境要求

构建前请确认已经安装并初始化 CANN/AscendC 开发环境，并且 CMake 可以找到 `ASC` 包。

常见环境初始化方式如下，实际路径请以本机安装位置为准：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

依赖工具：

- CMake 3.16 或更高版本
- CANN/Ascend Toolkit
- 支持 AscendC 自定义算子构建的编译环境

## 构建

推荐使用脚本构建：

```bash
./scripts/build.sh
```

默认构建目录为 `build`。也可以指定构建目录：

```bash
./scripts/build.sh out/build
```

脚本执行的核心流程等价于：

```bash
cmake -S . -B build
cmake --build build
```

构建成功后，自定义算子包会输出到 CMake 构建目录下，具体产物名称和目录由 CANN 的 `npu_op_package` 规则生成。

## 基础检查

可以运行 smoke 脚本检查项目文件和构建产物是否齐备：

```bash
./scripts/run_smoke.sh
```

如果使用了自定义构建目录：

```bash
./scripts/run_smoke.sh out/build
```

该脚本不会修改源码，也不会替代真实的 NPU 精度/性能测试；它主要用于 PR 前确认源码、CMake 配置、说明文档、脚本和本地构建目录处于可提交状态。

## PR 提交清单

- [x] 可运行源码
- [x] 配套说明文档
- [x] 构建脚本
- [x] 基础运行/产物检查脚本

提交 PR 前建议至少完成一次：

```bash
./scripts/build.sh
./scripts/run_smoke.sh
```

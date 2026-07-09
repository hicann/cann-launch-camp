# FastGelu Ascend C 算子工程说明

## 1. 课题说明

本项目实现了一个基于 Ascend C 的 FastGelu 激活算子，目标是在昇腾 NPU 上通过自定义算子方式完成逐元素的 FastGelu 计算。

FastGelu 的核心公式为：

$$

y = x \times \frac{1}{1 + e^{-1.702 \times x}}
$$

项目中采用了等价变换形式，减少了绝对值运算开销，并将计算逻辑封装在 Kernel 侧，配合 Host 侧的算子注册、Shape/Type 推导以及 Tiling 参数生成完成完整的算子流程。

### 研究目标
- 实现一个可在 Ascend AI Core 上运行的 FastGelu 自定义算子；
- 支持 float16 和 float32 两种输入类型；
- 通过 Tiling 与多核调度提升大规模输入场景下的执行效率；
- 形成可复用的 Ascend C 自定义算子工程模板。

## 2. 项目结构

```text
.
├── CMakeLists.txt
├── op_host/
│   ├── CMakeLists.txt
│   └── fast_gelu.cpp
└── op_kernel/
    ├── CMakeLists.txt
    ├── fast_gelu.cpp
    ├── fast_gelu_tiling.h
    └── tiling_key_fast_gelu.h
```

### 目录说明
- op_host：Host 侧算子注册、Tiling 配置、Shape/Type 推导。
- op_kernel：Kernel 侧 Ascend C 实现，包含计算核心与数据搬运逻辑。
- CMakeLists.txt：工程总入口，负责组织整个算子包构建。

## 3. 关键实现点

### Host 侧
- 注册自定义算子 `FastGelu`；
- 根据输入张量 shape 计算 Tiling 参数；
- 根据输入类型选择对应的 Kernel 模板实例；
- 为大规模输入自动配置多核分块策略。

### Kernel 侧
- 使用 Ascend C 的 `DataCopy` 和 `Compute` 流程完成数据搬运与计算；
- 利用等价变换减少重复计算；
- 将输入数据按块处理，避免单次处理过大数据导致资源压力过高。

## 4. 环境要求

建议在已安装昇腾 CANN/Ascend Toolkit 的开发环境中编译运行，需满足：
- CMake 3.16 或以上；
- Ascend C / CANN 开发环境；
- 支持 Ascend 910B 的编译与运行环境。

## 5. 运行指南

### 5.1 配置开发环境

在 Ascend 开发环境中加载工具链环境变量。例如：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.sh
```

如果你的安装路径不同，请将上述命令中的路径替换为实际路径。

### 5.2 创建构建目录

```bash
mkdir -p build
cd build
```

### 5.3 配置工程

```bash
cmake ..
```

### 5.4 编译工程

```bash
cmake --build . -j
```

编译完成后，工程将生成相应的算子库和算子包文件，具体产物位置取决于当前构建目录和 CMake 配置结果。

### 5.5 使用算子

编译成功后，可将生成的算子包用于后续的推理或算子验证流程，例如：
- 与 ACL/ACLNN 相关调用流程配合使用；
- 在训练推理框架中注册并调用该自定义算子；
- 编写自己的测试脚本对输入张量进行前向验证。

## 6. 建议的后续扩展

- 增加单元测试或端到端验证脚本；
- 支持更多数据类型，例如 bfloat16；
- 针对不同形状与不同块大小做性能对比分析；
- 将算子封装为更便于调用的 Python 或 C++ 测试例程。

## 7. 说明

本项目可作为“基于 Ascend C 实现自定义激活算子”的入门示例，适合用于课程作业、科研课题或算子开发训练。

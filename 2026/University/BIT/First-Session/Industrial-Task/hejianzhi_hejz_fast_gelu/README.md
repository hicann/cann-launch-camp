# FastGELU AscendC 实现说明

## 课题简介

本项目基于昇腾 CANN 平台，使用 AscendC 实现一个 FastGELU 激活函数算子。
FastGELU 是一种近似 GELU 的高效激活函数，常用于 Transformer 类模型中。
本实现重点是将 FastGELU 的计算逻辑封装为自定义算子，并通过 AscendC 的算子开发流程完成 host 侧注册、kernel 侧计算以及 tiling 配置。

## 项目结构

- `op_host/`：算子 host 侧注册与 tiling 配置
- `op_kernel/`：AscendC kernel 实现
  - `fast_gelu.cpp`：核心计算逻辑
  - `fast_gelu_tiling.h`：tiling 数据结构
  - `tiling_key_fast_gelu.h`：模板参数定义
- `CMakeLists.txt`：项目构建入口

## 实现目标

- 实现 FastGELU 的基本数学计算
- 支持 float16 和 float 数据类型
- 通过自定义 tiling 进行块划分，提升执行效率
- 保持与原有算子接口兼容，便于后续集成测试

## 运行与构建步骤

### 1. 环境准备

请确保已安装以下环境：

- CANN / Ascend 运行环境
- CMake
- 编译器与相关依赖

### 2. 配置构建

在项目根目录下执行：

```bash
cmake -S . -B build
```

### 3. 编译项目

```bash
cmake --build build -j
```

### 4. 运行测试或验证

根据实际实验环境，可通过自定义测试程序或算子验证脚本调用此算子进行结果检查。

## 代码说明

- `op_host/fast_gelu.cpp`
  - 完成算子注册
  - 定义输入输出信息
  - 生成 tiling 数据，设置 block dimension

- `op_kernel/fast_gelu.cpp`
  - 使用 AscendC 的全局内存、局部张量和向量计算接口
  - 通过双缓冲方式进行 DMA 与计算重叠
  - 完成 FastGELU 的逐块计算

## 注意事项

- 本项目为课程实验型实现，重点在于理解 AscendC 算子开发流程。
- 若需要实际部署到生产环境，还需要进一步进行性能调优、精度验证和算子兼容性测试。
- 请在实际运行前确认 Ascend 设备和驱动环境正常。

## 参考思路

本实现采用了与常见 FastGELU 近似公式一致的计算方式：

$$
\text{FastGELU}(x) = \frac{x}{1 + e^{-1.702x}}
$$

该公式用于在保持较好近似效果的同时，降低计算复杂度。

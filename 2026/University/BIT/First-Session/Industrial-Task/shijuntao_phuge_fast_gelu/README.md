# FastGelu 昇腾自定义算子 README

## 1. 工程简介

基于昇腾 CANN AscendC 实现 FastGelu 激活函数自定义算子，采用 Host 侧 Tiling 切分与 Device 核函数协同计算架构，支持 ACLNN 标准接口调用，适配昇腾 NPU 加速推理。

## 2. 目录结构

```text
├── CMakeLists.txt
├── op_host/        # Tiling + 算子注册 + Shape/DType 推导
│   └── fast_gelu.cpp
└── op_kernel/      # 核函数 + tiling 结构体 + 模板特化
    ├── fast_gelu.cpp
    ├── fast_gelu_tiling.h
    └── tiling_key_fast_gelu.h
```

## 3. 文件说明

**根目录**
- `CMakeLists.txt`：加载 CANN 编译工具链，串联编译 Host 与 Kernel 模块，打包输出算子动态库。

**op_host（主机侧）**
- `fast_gelu_tiling.h` / `tiling_key_fast_gelu.h`：定义 Tiling 切分参数结构体及数据类型泛型模板。
- `fast_gelu.cpp`：实现 Tiling 切分算法、形状/类型推导及算子注册，自动生成 `aclnnFastGelu` 调用接口。

**op_kernel（设备核）**
- `fast_gelu.cpp`：Ascend C 向量计算核函数，基于双缓冲流水线在 NPU 上执行 FastGelu 数学运算。

## 4. 编译步骤

```bash
# 1. 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 构建编译目录
mkdir build && cd build
cmake ..
make -j
编译产物输出至 build 目录，包含 Tiling 库、Kernel 核库及 ACLNN 接口库。

5. 使用方式
编译生成算子动态库并部署至设备；

通过 aclnnFastGelu 标准接口调用自定义算子；

支持推理框架直接调用，加速 GELU 计算。

6. 环境依赖
CANN 6.x 及以上版本

CMake ≥ 3.16

目标芯片：Ascend 910B（可通过 CMake 配置切换）
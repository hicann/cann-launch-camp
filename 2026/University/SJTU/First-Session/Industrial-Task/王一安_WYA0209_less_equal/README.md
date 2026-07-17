# LessEqual 算子实现

## 概述

本目录包含 LessEqual 算子的 Ascend C 实现。LessEqual 是一个逐元素比较算子，用于判断 `x1 <= x2`，返回布尔类型结果。

## 功能特性

- **支持数据类型**：float16、float32、int32、int8
- **输出类型**：bool（uint8）
- **广播支持**：完整支持 NumPy/TensorFlow 风格的广播语义
- **非对齐支持**：兼容非 32 字节对齐的维度

## 目录结构

```
王一安_WYA0209_less_equal/
├── CMakeLists.txt              # 顶层构建文件
── build.sh                    # 构建脚本
├── README.md                   # 本说明文档
── op_host/
│   ├── CMakeLists.txt          # Host 侧构建文件
│   └── less_equal.cpp          # 算子注册、Shape 推导、Tiling 函数
└── op_kernel/
    ├── CMakeLists.txt          # Kernel 侧构建文件
    ├── less_equal.cpp          # Kernel 核心计算逻辑
    ├── less_equal_tiling.h     # Tiling 数据结构定义
    └── tiling_key_less_equal.h # 模板分发宏定义
```

## 构建方法

### 前置条件

- CANN 开发环境已安装（建议 8.0.RC1 或更高版本）
- CMake 3.16+
- GCC 7.3+

### 构建步骤

```bash
# 1. 进入算子目录
cd 王一安_WYA0209_less_equal

# 2. 赋予构建脚本执行权限
chmod +x build.sh

# 3. 执行构建
./build.sh
```

构建成功后，输出文件位于 `build/` 目录下。

## 算子接口

### 输入

| 参数名 | 类型 | 说明 |
|--------|------|------|
| x1 | tensor | 第一个输入张量，支持 float16/float32/int32/int8 |
| x2 | tensor | 第二个输入张量，类型需与 x1 相同 |

### 输出

| 参数名 | 类型 | 说明 |
|--------|------|------|
| y | tensor | 比较结果张量，bool 类型，shape 与广播后的输入一致 |

### 计算规则

```
y[i] = (x1[i] <= x2[i]) ? True : False
```

## 广播示例

### 示例 1：相同形状

```
x1: [1.0, 2.0, 3.0, 4.0]  (float16, shape=[4])
x2: [2.0, 2.0, 2.0, 2.0]  (float16, shape=[4])
y:  [True, True, False, False]  (bool, shape=[4])
```

### 示例 2：广播

```
x1: [[1, 2], [3, 4]]  (float32, shape=[2,2])
x2: [2, 3]            (float32, shape=[2])
y:  [[True, True], [False, False]]  (bool, shape=[2,2])
```

x2 被广播为 `[[2,3],[2,3]]`，然后逐元素比较。

## 实现细节

### Host 侧 (op_host/less_equal.cpp)

- **算子注册**：定义输入输出类型、格式，注册 InferShape 和 InferDataType
- **InferShape**：实现 NumPy 广播规则，计算输出形状
- **InferDataType**：输出固定为 bool 类型
- **TilingFunc**：根据输入规模和 UB 大小计算最优分块策略，预计算广播 stride

### Kernel 侧 (op_kernel/less_equal.cpp)

- **Init**：初始化全局内存指针、管道缓冲区，复制 tiling 数据
- **Process**：分 tile 处理数据，每个 tile 执行 CopyIn -> Compute -> CopyOut
- **CopyIn**：根据是否需要广播，选择 DataCopy 或逐元素拷贝
- **Compute**：使用 Ascend C `Compare` 指令执行 `CMP_LE` 比较
- **CopyOut**：将 bool 结果写回全局内存

### 广播寻址

通过预计算的 stride 实现高效的广播寻址：
- 对于 flat output index，从内层到外层分解为多维坐标
- 使用 `inFlatIdx += coord * stride[d]` 计算输入索引
- 广播维度（大小为 1）的 stride 为 0，自动实现广播效果

## 作者信息

- **姓名**：王一安
- **GitCode 账号**：WYA0209
- **学校**：上海交通大学
- **课题**：LessEqual 算子开发

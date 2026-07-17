# LessEqual算子 - Ascend C原生实现

## 项目概述

本项目基于Ascend C编程语言，在昇腾NPU硬件上实现了高性能的LessEqual算子。该算子用于逐元素比较两个张量的大小关系（x1 <= x2），返回布尔类型的结果。

算子行为与TensorFlow原生`tf.math.less_equal`算子完全一致，支持NumPy标准的广播语义。

## 功能特性

- **支持的数据类型**：float16、float32、int32、int8
- **输出类型**：bool（int8表示，1=True，0=False）
- **广播支持**：完全支持NumPy/TensorFlow标准广播语义
- **非对齐兼容**：支持N、N2、N3、N4为非32整倍数的内存非对齐场景
- **硬件平台**：Ascend 910B

## 项目结构

```
zhangyiming_zYiming_less_equal/
├── README.md                          # 项目说明文档
├── project.txt                        # 项目描述文件
└── src/                               # 源码目录
    ├── CMakeLists.txt                 # 主CMake配置文件
    ├── op_host/                       # Host侧源码
    │   ├── CMakeLists.txt             # Host侧CMake配置
    │   └── less_equal.cpp             # Host侧实现（Tiling、形状推断、算子注册）
    └── op_kernel/                     # Kernel侧源码
        ├── CMakeLists.txt             # Kernel侧CMake配置
        ├── less_equal.cpp             # Kernel侧实现（AI Core计算逻辑）
        ├── less_equal_tiling.h        # Tiling数据结构定义
        └── tiling_key_less_equal.h    # 模板参数声明与选择
```

## 核心实现机制

### 处理模式

算子实现了两种处理模式：

1. **Fast模式（mode=0）**：当两个输入张量形状完全相同时使用
   - 直接按一维方式分块处理
   - 内存访问模式简单高效

2. **Broadcast模式（mode=1）**：当两个输入张量形状不同但满足广播条件时使用
   - 将张量按行划分（最后一维为每行长度）
   - 根据步长信息处理广播逻辑（stride=0表示标量广播）

### Tiling策略

Host侧根据UB容量和数据类型计算最优的Tile大小：

- **Tile大小计算**：根据数据类型所需的UB空间，计算可用UB能容纳的最大Tile大小
- **并行策略**：按核心数均分数据，每个核心处理一部分元素或行
- **内存对齐**：数据长度向上对齐到256的倍数（AI Core向量指令要求）

### 计算逻辑

不同数据类型采用不同的比较策略：

- **float16/float32**：直接使用`Compare`指令进行LE比较
- **int32**：使用`Min`+`EQ`组合实现LE比较（因为Compare不支持int32 LE）
- **int8**：先转换为half类型再进行比较

## 输入输出规范

### 输入参数

| 参数名 | 类型 | 数据类型 | 格式 | 说明 |
|--------|------|----------|------|------|
| x1 | tensor | float16/float32/int32/int8 | ND | 第一个输入张量 |
| x2 | tensor | float16/float32/int32/int8 | ND | 第二个输入张量 |

### 输出参数

| 参数名 | 类型 | 数据类型 | 格式 | 说明 |
|--------|------|----------|------|------|
| y | tensor | bool | ND | 比较结果张量 |

### 约束条件

- 两个输入张量数据类型必须相同
- 输入张量形状需满足广播规则
- 维度范围：N ∈ [1, 10000], N2 ∈ [1, 10000], N3 ∈ [1, 2000], N4 ∈ [1, 500]

## 构建说明

### 环境要求

- 操作系统：Linux（推荐Ubuntu 22.04）
- CANN版本：6.0.RC1及以上
- 编译器：Ascend C Compiler

### 构建步骤

```bash
# 创建构建目录
mkdir build && cd build

# 配置CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
make -j$(nproc)

# 安装（可选）
make install
```

### 构建产物

构建成功后会生成以下文件：

- `custom.so`：算子包动态库
- `cust_optiling.so`：Tiling动态库
- `cust_opapi.so`：ACLNN接口动态库

## 使用方法

### ACLNN接口调用

```cpp
#include "aclnn/aclnn.h"

// 创建输入张量
aclnnTensor* x1 = ...;  // float16/float32/int32/int8
aclnnTensor* x2 = ...;  // 与x1相同类型

// 创建输出张量描述
aclnnTensorDescriptor yDesc;
aclnnCreateTensorDescriptor(&yDesc, ACL_FORMAT_ND, ACL_BOOL, ...);

// 调用算子
aclnnLessEqual(x1, x2, yDesc, ...);
```

### 广播示例

```python
# 示例：矩阵与向量广播比较
x1 = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)  # shape=[2,2]
x2 = tf.constant([2, 3], dtype=tf.float32)            # shape=[2]

# x2广播为[[2,3],[2,3]]
y = tf.math.less_equal(x1, x2)  # shape=[2,2]
# 结果: [[True, True], [False, False]]
```

## 精度验证

算子精度与TensorFlow原生`tf.math.less_equal`完全对齐，比较运算要求完全准确，无误差容忍。

验证方法：

1. 生成随机测试数据
2. 在TensorFlow上执行`tf.math.less_equal`
3. 在Ascend NPU上执行本算子
4. 对比输出结果，要求完全一致

## 性能优化

### 已实现的优化

1. **向量化计算**：利用AI Core向量指令批量处理数据
2. **UB缓存优化**：合理分配UB缓冲区，最大化数据复用
3. **双缓冲技术**：输入输出队列采用双缓冲，隐藏数据传输延迟
4. **连续维度合并**：在Tiling阶段合并连续维度，减少Kernel侧维度处理开销

### 性能指标

- **吞吐量**：充分利用AI Core并行计算能力
- **内存带宽**：优化内存访问模式，提高数据传输效率

## 技术文档

- [LessEqual算子任务说明](../LessEqual.md)
- [PR提交指南](../../PR-submission-guide.md)

## 联系方式

- 作者：zhangYiming
- GitCode账号：zYimng
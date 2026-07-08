# **Ascend C 自定义算子：FastGelu 高性能实现**

## **一、 赛题与项目背景**

FastGelu 是一种快速近似的高斯误差线性单元（GELU）激活函数，相比标准 GELU 计算效率更高，广泛应用于深度学习模型（特别是 Transformer 架构和大语言模型）的激活层中。  
本项目基于 MindSpore 原生 ops.fast\_gelu 算子的核心业务逻辑，采用 **Ascend C** 编程语言进行算子原生开发。旨在昇腾 **Ascend 910B NPU** 硬件上实现一款兼顾高精度（双万分之一）与超高性能（微秒级极速响应）的自定义算子。

## **二、 数学原理与指令映射**

算子的核心计算公式是对标准 GELU 函数的高效近似：

$$y \= \\frac{x \\cdot e^{0.851 \\cdot (x \- |x|)}}{1 \+ e^{-1.702 \\cdot |x|}}$$  
为了充分利用 NPU 的 Vector 计算单元，我们在 Kernel 侧将上述连续数学公式拆解为以下 Ascend C 离散张量指令，确保计算的数值稳定并避免溢出：

1. **计算绝对值**：Abs $\\rightarrow |x|$  
2. **计算分母**：Muls, Exp, Adds $\\rightarrow 1 \+ e^{-1.702 \\cdot |x|}$  
3. **计算分子指数部分**：Sub, Muls, Exp $\\rightarrow e^{0.851 \\cdot (x \- |x|)}$  
4. **计算分子**：Mul $\\rightarrow x \\cdot e^{0.851 \\cdot (x \- |x|)}$  
5. **最终输出**：Div $\\rightarrow$ 分子 / 分母

## **三、 核心规格与约束支持**

本算子严格满足赛题定义的所有边界与常规场景：

* **数据类型支持**：全面支持 float16 与 float32 两种标准浮点类型。  
* **维度与形状**：支持任意多维张量（ND 格式）。最终维度可拆解为 (..., N4, N3, N2, N)。  
  * N $\\in$ \[1, 10000\]  
  * N2 $\\in$ \[1, 10000\]  
  * N3 $\\in$ \[1, 2000\]  
  * N4 $\\in$ \[1, 500\]  
* **精度达标**：  
  * float32：相对误差 \< 1e-4，绝对误差 \< 1e-4（双万分之一精度）。  
  * float16：相对误差 \< 1e-3，绝对误差 \< 1e-3（双千分之一精度）。  
* **非对齐场景**：原生兼容 N, N2, N3, N4 均不为 32 的整倍数的**内存非 32 字节对齐场景**。

## **四、 核心性能优化方案 (High-Performance Tuning)**

为了在 OJ 评测系统中取得顶尖的运行耗时表现，本工程实施了以下三项核心调优：

### **1\. 内存极度安全与 32 Bytes 严格对齐 (Tiling 侧)**

在处理非 32 字节整倍数（如尾部维度为奇数）的张量时，常规计算极易发生 Global Memory (GM) 或 Unified Buffer (UB) 的越界访问，导致 Undefined Behavior (UB)。

* **解决方案**：在 CPU Host 侧的 Tiling 算法中，动态获取当前数据类型字长，强制计算 align\_num。将总数据量和每个 AI Core 分配的 block\_len 均**向上取整到 32 Bytes 的整数倍**。配合 Kernel 侧的安全偏移量检查，彻底消灭了尾块越界问题。

### **2\. 异步流水线：Double Buffering 双缓冲机制 (Kernel 侧)**

早期的串行逻辑会导致计算单元（Vector）等待搬运单元（DMA），反之亦然。

* **解决方案**：将 BUFFER\_NUM 设定为 2，把 Local Tensor 内存划分为 Ping 和 Pong 两个交替使用的缓冲区。这使得**数据搬入 (CopyIn)**、**向量计算 (Compute)** 和**数据写回 (CopyOut)** 三个阶段实现了流水线式的物理并发，相互掩盖了执行延迟，整体耗时缩减约 40%。

### **3\. 指令调度开销极小化：Tile 块激进放大**

* **解决方案**：摒弃了每次只搬运极少量数据进 UB 的低效做法。通过精准计算 NPU 910B 的 Unified Buffer 容量上限，为 calcBuf 预留足够寄存器空间后，尽可能放大了单次循环处理的 tile\_len 数据量。这大幅减少了 AI Core 内部的 for 循环次数和指令发射开销，将用时稳定压制在 10 微秒量级内。

## **五、 工程目录结构**

zhoulihangyu\_liudehua777\_fast\_gelu/  
├── CMakeLists.txt                 \# 主工程构建配置，串联 Tiling 库与 Kernel 库  
├── README.md                      \# 项目说明文档  
├── op\_host/                       \# Host侧实现 (运行于 CPU)  
│   ├── fast\_gelu.cpp              \# 包含算子定义、Shape/Type推导以及 Tiling 切分算法  
│   └── CMakeLists.txt             \# Host侧编译配置  
└── op\_kernel/                     \# Kernel侧实现 (运行于 AI Core)  
    ├── fast\_gelu.cpp              \# Ascend C 原生 API 计算逻辑与双缓冲流水线  
    ├── fast\_gelu\_tiling.h         \# Tiling 通信结构体定义 (含对齐参数与分块参数)  
    ├── tiling\_key\_fast\_gelu.h     \# 泛型模板参数注册 (支持 fp16/fp32 动态分发)  
    └── CMakeLists.txt             \# Kernel侧编译配置


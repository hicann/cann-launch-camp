# FastGelu 算子实现

## 1. 概述

本项目基于 Ascend C 在昇腾 910B NPU 上实现了 MindSpore 的 fast_gelu 激活函数。

算子严格遵循数值稳定公式：

\[ \text{output} = \frac{x \cdot \exp\big(0.851 \cdot (x - |x|)\big)}{1 + \exp\big(-1.702 \cdot |x|\big)} \]

支持 float16 和 float32 两种数据类型，输入张量可以是任意维度，并兼容非 32 字节对齐的尾部数据。

## 2. 文件结构

```
├── CMakeLists.txt                    # 顶层 CMake 配置
├── op_host/
│   ├── CMakeLists.txt
│   └── fast_gelu.cpp                 # Host 侧 Tiling 逻辑与算子注册
└── op_kernel/
    ├── CMakeLists.txt
    ├── fast_gelu.cpp                 # Kernel 侧核函数实现
    ├── fast_gelu_tiling.h            # Tiling 数据结构定义
    └── tiling_key_fast_gelu.h        # 数据类型模板选择
```

## 3. 性能优化要点

### 3.1 自适应多核并行

当总数据量 ≤ 64 KB 时，自动使用单核执行，避免多核调度与同步开销。

当数据量较大时，按照每核至少处理 2 个 32B 块的粒度分配核数，充分利用硬件并行能力。

### 3.2 均衡负载与尾块处理

输入数据按 32 字节对齐后均分给各核，前 tailBlockNum 个核仅多处理一个 32B 块，实现各核执行时间基本相同，无核空闲等待。

通过 tileDataNum 和 tailDataNum 精确控制每个分块大小，最后一个 tile 自动适配实际剩余元素，支持任意非对齐场景。

### 3.3 双缓冲流水线

使用四个独立队列（输入、临时1、临时2、输出），每个队列开启双缓冲（BUFFER_NUM = 2）。

在双缓冲机制下，数据搬运（DataCopy）与向量计算（Abs / Muls / Exp / Add / Sub / Mul / Div）自动重叠，计算单元几乎无空闲。

### 3.4 高精度数值计算

完全按照原始公式分步计算，不使用任何近似激活函数。

分母：abs(x) → -1.702 * abs(x) → exp → +1

分子：x - abs(x) → 0.851 * (x - abs(x)) → exp → × x

最终：分子 / 分母

精度指标：

- float32：相对误差 < 1e-4，绝对误差 < 1e-4
- float16：相对误差 < 1e-3，绝对误差 < 1e-3

### 3.5 内存访问优化

依据 Unified Buffer 实际容量动态计算最大 tile 大小，保证单次搬运数据量远大于 16 KB，最大化带宽利用率。

每核内通过 SetGlobalBuffer 精确定位自己的数据段，内存访问连续且无冲突。

## 4. 编译与运行

### 4.1 环境要求

- CANN 8.5.0 及以上版本
- CMake ≥ 3.16.0
- 支持昇腾 910B 的驱动与固件

### 4.2 编译

```bash
mkdir build && cd build
cmake ..
make
```

编译成功后将在构建目录下生成算子包。

### 4.3 单算子调用测试

可通过 ACLNN 接口或 kernel_sigmoid 类似的直调方式进行测试，输入支持任意 float16/float32 张量。

## 5. 验证结果

与 MindSpore CPU 参考结果逐元素对比，所有测试用例（小/中/大数据量，float16/float32）均精度通过。

小数据量耗时较初始版本显著缩短，大数据量性能保持最优，整体吞吐稳定。

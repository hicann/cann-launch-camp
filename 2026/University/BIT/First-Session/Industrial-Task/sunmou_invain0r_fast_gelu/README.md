# FastGelu 算子 (Ascend C)

基于 Ascend C 编程语言在昇腾 NPU 上实现的高性能 FastGelu 激活函数算子。

## 数学原理

FastGelu 是对标准 GELU 函数的快速近似，经过恒等变换可化简为：

$$y = x \cdot \text{sigmoid}(1.702 \cdot x)$$

从 9 次运算（Abs→Sub→Muls→Exp→Mul→Muls→Exp→Adds→Div）降为 3 次向量化运算（Muls→Sigmoid→Mul），大幅提升计算效率。

## 项目结构

```
src/
├── CMakeLists.txt               # 工程构建配置
├── op_host/
│   └── fast_gelu.cpp            # Host 侧 Tiling 实现
├── op_kernel/
│   ├── fast_gelu.cpp            # Kernel 侧核函数
│   ├── fast_gelu_tiling.h       # Tiling 结构体定义
│   └── tiling_key_fast_gelu.h   # Tiling Key 模板定义
└── README.md
```

## 设计概要

### Host 侧 (`op_host/fast_gelu.cpp`)

| 设计 | 说明 |
|------|------|
| **Tiling 精简** | 仅计算 3 个通用参数（`length`, `blockNum`, `tileDataNum`） |
| **UB 适配** | 根据 UB 大小和 4 slot 占用计算最大 tile 元素数 |
| **核数自适应** | 保证每核至少 4 个 tile，小数据自动合并到更少核上 |

### Tiling 结构体 (`fast_gelu_tiling.h`)

```cpp
struct FastGeluTilingData {
    uint32_t length;       // 总元素个数
    uint32_t blockNum;     // 使用的 AI Core 数量
    uint32_t tileDataNum;  // 每个完整 tile 的元素数
};
```

### Kernel 侧 (`op_kernel/fast_gelu.cpp`)

| 设计 | 说明 |
|------|------|
| **无核类型分支** | 所有核使用统一公式按 32B 块粒度计算数据范围 |
| **直通/流水双路径** | 单 tile 时跳过队列同步直接计算；多 tile 时启用双缓冲流水 |
| **UB slot 复用** | 复用 `yLocal` 作为 Muls/Sigmoid 临时空间，峰值 4 slot |
| **快速数学化简** | 3 次向量指令完成 FastGelu 计算 |

## 环境依赖

- **硬件**：昇腾 910B (ascend910b)
- **软件**：CANN 商用版（含 AscendC 开发套件）、CMake >= 3.16.0

## 构建

```bash
mkdir build && cd build
cmake ..
make
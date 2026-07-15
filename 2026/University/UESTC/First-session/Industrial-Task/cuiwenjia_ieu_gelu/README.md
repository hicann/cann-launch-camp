# Gelu — Ascend C 算子赛题

## 方案概述

使用 **Padé 有理分式 + Sigmoid** 近似替代标准 Erf 计算 GELU。

核心技巧：利用恒等式 $\frac12+\frac12\tanh(z)=\operatorname{sigmoid}(2z)$，
将 Tanh 替换为 Exp（910B 上 Exp 比 Tanh 快得多），同时保持完全相同的精度。

### 公式

$$z = \frac{x \cdot (C_0 + C_2 \cdot x^2)}{1 + D_2 \cdot x^2}$$

$$\operatorname{GELU}(x) = x \cdot \operatorname{sigmoid}(2z) = \frac{x}{1 + \exp(-2z)}$$

| 系数 | 值 |
|------|-----|
| $C_0$ | 0.7974154410732703 |
| $C_2$ | 0.0456776776271884 |
| $D_2$ | 0.0107061942549241 |
| **理论最大误差** | $\mathbf{3.22 \times 10^{-5}}$ |

### 精度对比

| 方法 | 最大误差 | half (1e-3) | fp32 (1e-4) |
|------|---------|-------------|-------------|
| Exact $\operatorname{erf}$ | 0 | ✅ | ✅ |
| Hendrycks $\tanh\bigl(\sqrt{2/\pi}(x+0.0447x^3)\bigr)$ | $4.73\times10^{-4}$ | ✅ | ❌ |
| $\operatorname{Sigmoid}(1.702x)$ | $2.03\times10^{-2}$ | ❌ | ❌ |
| **Padé rational-sigmoid（本方案）** | $\mathbf{3.22\times10^{-5}}$ | **✅** | **✅** |

详见 `fit_gelu.py`。

## 最终性能

| Case | time | best |
|------|------|------|
| 1 | **3.21ms** | 3.21ms |
| 2 | 7.03ms | 6.93ms |
| 3 | 13.25ms | 11.89ms |
| 4 | 4.46ms | 3.86ms |
| 5 | 10.89ms | 9.64ms |
| 6 | 51.76ms | 48.75ms |

Score: **85.65**, Rank: **#3**（截止该文件编写时）

## 优化历程

| 尝试 | 结果 |
|------|------|
| 手写 Erf | 精度通过，但 Erf 高延迟 |
| 内置 Gelu\<T\> API | half 通过，fp32 精度 0.948x ❌ |
| Gelu\<T, true\> (高精度) | fp32 精度仍 0.948x ❌ |
| FasterGelu / FasterGeluV2 | 精度 0.19 ❌ |
| Sigmoid 近似 (x·σ(1.702x)) | 误差 2e-2 ❌ |
| 标准 tanh (Hendrycks) | 误差 4.7e-4，fp32 ❌ |
| Padé rational-tanh | 3.22e-5 ✅，但 Tanh 较慢 |
| **Padé rational-sigmoid** | **3.22e-5 ✅，Exp 替代 Tanh，性能 +27%** |
| 单 tile TBuf 快路径 | 小张量改善 |
| 自适应核数（512/1024/768 分档） | 中张量改善 |
| 512B GM 对齐 | DMA 改善 |
| UB bank padding | 无收益 |

## 关键代码

- `op_kernel/gelu.cpp` — Kernel 侧实现，含 Padé 计算 + 单/多 tile 双路径
- `op_host/gelu.cpp` — Host 侧 Tiling，自适应核数和 tile 大小
- `op_kernel/gelu_tiling.h` — TilingData 结构体
- `op_kernel/tiling_key_gelu.h` — TilingKey 模板声明
- `fit_gelu.py` — 系数拟合脚本

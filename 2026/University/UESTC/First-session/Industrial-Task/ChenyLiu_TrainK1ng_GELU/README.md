# GELU 算子 (Ascend C)

基于华为昇腾 Ascend C 编程框架实现的高性能 GELU（Gaussian Error Linear Unit）激活函数算子。

## 特性

| 项目 | 说明 |
|------|------|
| **支持平台** | Ascend 910B |
| **支持数据类型** | FP16 (FLOAT16)、FP32 (FLOAT) |
| **数据格式** | ND |
| **计算模式** | AIV Only（纯向量计算） |
| **优化策略** | 双缓冲 (Double Buffer)、32B 地址对齐、负载均衡分核 |

---

## 目录结构

```
.
├── gelu.cpp              # Host 侧：算子注册、Tiling 函数、Shape/数据类型推导
├── gelu_tiling.h         # Tiling 数据结构定义
├── tiling_key_gelu.h     # TilingKey 模板定义（支持 FP16/FP32 模板选择）
└── gelu_kernel.cpp       # Device 侧：AICore Kernel 实现
```

---

## 算法原理

本算子针对不同精度采用不同的计算策略，在精度与性能之间取得平衡：

### 1. FP16 快速近似（Tanh 近似）

利用 GELU 的 tanh 近似公式，减少高成本指令调用：

```
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

实现细节：
- 先对输入做 **clip 到 -6.0**，防止极端值导致数值不稳定
- 使用 `Mul` → `Muls` → `Adds` → `Mul` → `Muls` → `Tanh` → `Adds` → `Muls` → `Mul` 的流水线指令序列
- 全程使用 `half` 类型向量指令，最大化吞吐

### 2. FP32 精确实现（Erf）

使用误差函数 `erf` 的标准定义：

```
GELU(x) = 0.5 * x * (1 + erf(x / √2))
```

实现细节：
- 调用 `Erf`  intrinsic 指令
- 保持 FP32 全精度计算链

---

## 设计与优化

### Tiling 策略

Host 侧 `TilingFunc` 根据输入长度自动计算最优核数与分块策略：

1. **核数选择**：
   - 优先使用全部 AIV 核，但确保每个核至少处理 **512** 个元素
   - 若数据量较小，自动减少核数，避免空转

2. **32B 对齐**：
   - FP32 场景：要求 `blockLength` 为 **8** 的倍数（8 × 4B = 32B）
   - FP16 场景：要求 `blockLength` 为 **16** 的倍数（16 × 2B = 32B）
   - 通过逐步递减 `blockDim` 的方式寻找最优对齐核数，保证 `DataCopy` 高效执行

3. **TilingKey 模板选择**：
   - 根据输入数据类型（`DT_FLOAT16` / `DT_FLOAT`）在编译期选择对应 Kernel 模板
   - 避免运行时分支判断，减少指令开销

### Kernel 侧优化

- **双缓冲 (Double Buffer)**：`BUFFER_NUM = 2`，`TILE_LENGTH = 4096`
  - 拷贝与计算流水线并行，隐藏内存延迟
- **非对齐安全处理**：
  - 对于不能被 32B 整除的尾部数据，使用 `DataCopyPadExt` / `DataCopyPad` 安全拷贝
  - 避免越界或非法内存访问
- **负载均衡**：
  - 采用 "余数优先分配" 策略：将 `length % coreNum` 的剩余元素优先分配给前 `rem` 个核
  - 保证各核处理量差不超过 1 个元素

---

## 编译与运行

### 环境要求

- CANN 开发套件（建议版本 ≥ 8.0）
- Ascend 910B 硬件环境
- `aarch64` 或 `x86_64` 编译宿主机

### 编译步骤

```bash
# 1. 配置 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 进入算子工程目录
cd <your_op_project>

# 3. 编译（以 msopgen 生成的 CMake 工程为例）
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### 运行测试

将编译生成的 `.om` 或 `.json` 文件通过 ACL 接口 / ATC 工具加载到 Ascend 910B 上执行。

---

## 接口说明

### 输入

| 名称 | 类型 | 格式 | 说明 |
|------|------|------|------|
| `input_x` | FP16 / FP32 | ND | 输入张量 |

### 输出

| 名称 | 类型 | 格式 | 说明 |
|------|------|------|------|
| `output` | FP16 / FP32 | ND | GELU 激活后的输出张量，shape 与输入一致 |

---

## 注意事项

1. **数据类型一致性**：输入与输出的数据类型必须相同，算子内部不做类型转换。
2. **空 Tensor 保护**：若 `length == 0`，Tiling 函数返回 `GRAPH_FAILED`，防止下发空任务。
3. **不支持的数据类型**：当前仅支持 FP16 和 FP32，其余类型（如 INT8、BF16）会在 Tiling 阶段返回失败。
4. **邮箱配置**：提交代码前请确保 Git 配置与 CLA 签署邮箱一致：
   ```bash
   git config --global user.email "你的CLA邮箱"
   git config --global user.name "你的姓名"
   ```

---

## 作者

CANN 社区贡献者

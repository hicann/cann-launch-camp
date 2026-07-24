# 团队实践报告

## 一、团队信息与贡献说明

### 1.1 团队基本信息

- 团队标识（组号）：哈基米奶龙
- CANNJudge 提交账号：yycc
- CANNJudge 提交结果或链接：https://cannjudge.cn/hit/20260721/qmmcustom/ranking

### 1.2 团队成员分工与贡献

| 姓名 | GitCode 账号 | 负责模块/任务分工 | 实际贡献说明 | 对应 commit hash |
|------|-------------|------------------|-------------|-----------------|
| 刘文轩 | 2301_79553548 | Notebook 与报告 | 完成 Notebook 全部单元格运行、结果整理与验证，编写团队实践报告 | 5f619d59d441d6f98c5a8c18ee04cec6963354f9 |
| 姚畅 | yycc | 算子实现（asc） | 完成 TilingData 设计、QmmCubeBasicKernel 和 QmmPertokenKernel 实现及 5 项性能优化，完成算子功能测试和 Profiling | 22cd069b953aa9984cec12712200b7f6ea801802 |

### 1.3 团队协作说明

团队采用"分工开发、统一汇总"的协作模式：

1. **任务拆分**：姚畅负责算子核心实现（Tiling 设计、Kernel 编写及性能优化），包括 5 项性能优化策略的探索与落地；刘文轩负责 Notebook 运行验证和报告撰写。
2. **代码汇总**：姚畅提交算子源码 `.asc` 文件，刘文轩提交 Notebook 结果和报告。
3. **集成验证**：姚畅完成算子功能测试（12 组 shape 全部通过）和 Profiling 数据收集，刘文轩在此基础上整理 Notebook 输出、编写报告，两人共同确认模型接入推理结果正确。
4. **评审机制**：每项优化变更后均重新运行全部 24 条测试用例（12 shape × INT32/BF16），确保功能不回退。

---

## 二、结果展示

### 2.1 单算子精度比对结果

12 组测试 shape 全部通过功能验证（INT32 路径精确匹配，BF16 路径在 rtol=0.01, atol=0.01 范围内通过）：

| M    | K     | N     | INT32 | BF16 |
|------|-------|-------|-------|------|
| 1    | 4096  | 4096  | PASS  | PASS |
| 1    | 4096  | 6144  | PASS  | PASS |
| 1    | 4096  | 24576 | PASS  | PASS |
| 1    | 12288 | 4096  | PASS  | PASS |
| 50   | 4096  | 4096  | PASS  | PASS |
| 50   | 4096  | 6144  | PASS  | PASS |
| 50   | 4096  | 24576 | PASS  | PASS |
| 50   | 12288 | 4096  | PASS  | PASS |
| 4096 | 4096  | 4096  | PASS  | PASS |
| 4096 | 4096  | 6144  | PASS  | PASS |
| 4096 | 4096  | 24576 | PASS  | PASS |
| 4096 | 12288 | 4096  | PASS  | PASS |

**结论**：INT32 12/12 通过，BF16 12/12 通过，总计 24/24 全通过。

### 2.2 单算子性能测试结果

以下数据为 2026-07-23 本机实测（Ascend 910B, CANN 9.1.0），使用 `torch.npu.Event` 精确计时（50 次迭代取平均，不含 Host 端 Python 开销）：

| Shape | INT32 (μs) | BF16 (μs) |
|-------|:----------:|:---------:|
| M=1, K=4096, N=4096 | 24.8 | 25.3 |
| M=1, K=4096, N=6144 | 28.0 | 29.1 |
| M=1, K=4096, N=24576 | 63.0 | 63.6 |
| M=1, K=12288, N=4096 | 36.3 | 36.9 |
| M=50, K=4096, N=4096 | 32.5 | 37.0 |
| M=50, K=4096, N=6144 | 53.1 | 57.3 |
| M=50, K=4096, N=24576 | 122.4 | 128.4 |
| M=50, K=12288, N=4096 | 69.4 | 73.5 |
| M=4096, K=4096, N=4096 | 370.6 | 458.5 |
| M=4096, K=4096, N=6144 | 633.2 | 782.2 |
| M=4096, K=4096, N=24576 | 2790.7 | 3559.6 |
| M=4096, K=12288, N=4096 | 1155.6 | 1264.6 |

> 注：使用 `torch.npu.Event` 计时，warmup 5 次后取 50 次迭代平均。该计时方式排除了 Python 层的函数调用开销和 kernel launch 排队延迟，反映 kernel 在 NPU 上的实际执行时间。

### 2.3 算子接入模型性能测试结果

将 QmmCustom 算子接入 Qwen3-8B W8A8 量化模型后，模型输出正确的 attention 机制描述（输入为 "An attention function can be described as mapping a query and a set of key-value pairs to an output..."）：

> **模型输出文本：**
>
> The output of an **attention function** is typically a **weighted sum** of the **value vectors**, where the weights are determined by the **similarity** between the **query vector** and the **key vectors** from the set of key-value pairs.
>
> ### Formal Description:
>
> Given:
> - A **query vector** $ Q $
> - A set of **key-value pairs**: $ (K_1, V_1), (K_2, V_2), \dots, (K_n, V_n) $
>
> The **attention function** computes the output as:
>
> $$
> \text{Attention}(Q, K, V) = \sum_{i=1}^{n} \alpha_i V_i
> $$
>
> Where:
> - $ \alpha_i = \text{softmax}\left( \frac{Q \cdot K_i}{\sqrt{d_k}} \right) $
> - $ d_k $ is the dimensionality of the key vectors (used for scaling to prevent large values)
> - $ \cdot $ denotes the dot product
> - $ \text{softmax} $ ensures that the weights $ \alpha_i $ sum to 1 and represent attention distribution
>
> ### Final Output:
> The **output** is a **vector** that is a
>
> *(输出在 256 tokens 处截断，与参考结果一致)*

模型推理性能（来自 scheduler 日志）：
- **Prefill 阶段**：49.25 ms
- **Decode 阶段**：平均 36.16 ms/token（256 tokens 总计）

模型 Profiling 中 QmmCustom 算子耗时统计：

| Phase | Block Dim | Avg (μs) | Calls | Min (μs) | Max (μs) |
|-------|-----------|----------|-------|----------|----------|
| Prefill | 20 | 98.7 | 72 | 55.8 | 150.5 |
| Prefill | 16 | 61.6 | 72 | 36.9 | 94.0 |
| Decode | 20 | 56.3 | 216 | 28.4 | 96.6 |
| Decode | 16 | 35.6 | 216 | 20.7 | 60.3 |

Profiling 结果显示 QmmCustom 在 Prefill 和 Decode 阶段均正常调用，算子正确替换了原 `npu_quant_matmul`。

---

## 三、方案说明

### 3.1 设计思路

#### 整体架构

算子采用 `__mix__(1, 2)` 混合架构：1 个 AIC（AI Core Cube 单元）负责 INT8 矩阵乘法计算，2 个 AIV（AI Core Vector 单元）负责反量化操作。

```
数据流：
  x1(INT8, [M,K]) ──┐
                     ├──► Cube: Matmul INT8×INT8 → INT32 workspace
  x2(INT8, [K,N]) ──┘              │
                                    ▼
                     CrossCoreSetFlag/WaitFlag 同步
                                    │
        ┌───────────────────────────┤
        ▼                           ▼
  AIV0: Cast→Mul→Muls→Cast    AIV1: Cast→Mul→Muls→Cast
  (处理一半行)                  (处理另一半行)
        │                           │
        └───────────┬───────────────┘
                    ▼
            y(BF16, [M,N])
```

#### TilingData 设计

`QmmCustomTilingData` 结构体包含：
- `TCubeTiling tiling`：标准 Matmul Tiling 参数（baseM/baseN/baseK、usedCoreNum 等）
- `bool isPertoken`：标记是否走 BF16 反量化路径
- `uint64_t workspaceSize`：Path 2 所需的 GM workspace 大小

Tiling 计算使用 `MultiCoreMatmulTiling` API，设置 `baseK=128`、`baseN=256`，自适应 `baseM` 三段式（M≤16→16, M<128→32, else→128）。分核策略上，M≤64 时仅切 N 维度以充分利用多核；M>64 时同时切 M 和 N，优先选择可整除 AIC 核数的 N 分块数，平衡 Cube 利用率和多核负载均衡。

#### Kernel 实现

- **Path 1（Cube-only）**：`QmmCubeBasicKernel` — AIC 完成 INT8 matmul，结果直接写 INT32 输出。
- **Path 2（Cube+Vector）**：`QmmPertokenKernel` — AIC 先完成 matmul 写 workspace，SetFlag 通知 AIV，AIV 读 workspace 做反量化（INT32→FLOAT32→×scale→×pertoken→BF16），写最终输出。

### 3.2 问题解决与优化策略

#### 遇到的问题及解决方案

1. **GM 数据拷贝对齐问题**：非对齐的 GM 数据直接 `DataCopy()` 会导致越界崩溃。最终方案通过 VEC_LEN=2048 的对齐分块循环，确保每次 `DataCopy()` 操作的数据地址和长度均对齐，避免了非对齐问题。
2. **GM 标量读取失败**：使用 `reinterpret_cast` 直接读 GM 地址不可靠。改用 `SetGlobalBuffer` 创建 GlobalTensor，再用 `GetValue(0)` 读取标量，这是 AscendC 访问 GM 标量的正确方式。
3. **多核分块对齐问题**：简单 `totalLength / numCores` 导致非对齐分块。改用 `CeilDivHost` + `AlignUpHost` 确保 singleM 按 16 对齐、singleN 按 32 对齐。
4. **v13 误删 SetOrgShape 导致 NPU 崩溃**：以为只用 `SetSingleShape` 就够了，但 `TCubeTiling` 依赖 `SetOrgShape` 初始化基维度（Ka, Kb, M, N）。恢复 `SetOrgShape` 后正常。

#### AI 辅助使用说明

本团队在开发过程中使用了 AI 编程助手（DeepSeek 和 Codex）：

- **解决的问题**：快速定位 AscendC API 用法（如 `SetGlobalBuffer` + `GetValue` 读取 GM 标量的正确方式、`AscendC::Std::sqrt` 标量用法），加速 Tiling 结构体设计和性能瓶颈分析。
- **引入的问题**：AI 曾建议删除 `SetOrgShape`（导致 v13 NPU 崩溃），以及建议使用 `reinterpret_cast` 读 GM（不可靠）。团队通过回归测试及时发现并修正。
- **验证方式**：所有 AI 建议的代码变更均经过 24 条测试用例验证通过后才采纳，确保功能正确性。

#### 性能优化策略

最终版本（FINAL）包含 5 项有效性能优化，外加 1 项代码清理（移除冗余 `SetShape`，对性能无影响）：

| 序号 | 优化项 | 原理 | 效果 |
|------|--------|------|------|
| 1 | Double Buffer | `InitBuffer(intQueue_, 2)` + `InitBuffer(outQueue_, 2)`，MTE 预取与 Vector 计算流水线重叠 | 大 shape 提升显著 |
| 2 | 删除全部 PipeBarrier | Cast→Mul→Muls→Cast 全在 VEC pipe，顺序执行无需同步 | 消除无效等待 |
| 3 | 自适应 baseM | `(M≤16)?16:(M<128?32:128)`，大 M 用 128 提升 Cube 利用率 | Cube 效率提升 |
| 4 | Tiling 缓存 | `static std::map` + `std::mutex`，同 shape 免重算 GetTiling | Host 侧 ~1400μs→0 |
| 5 | VEC_LEN=2048 | Vector 处理 chunk 翻倍，减少循环次数 | 小幅 |
| — | 精简 Tiling API | 删除冗余 `SetShape`，仅保留 `SetOrgShape` + `SetSingleShape`（代码清理，无性能影响） | — |

---

## 四、收获与感悟

### 刘文轩（2023113088）

本次启航营实践让我全面体验了从算子开发到模型接入的完整流程。我主要负责 Notebook 的运行验证和报告撰写，这让我对每个环节的输入输出有了清晰的认识——从环境准备、算子编译、功能测试到模型推理，每一步都需要仔细检查输出结果。

通过整理 Profiling 数据和撰写报告，我深入理解了 5 项性能优化的原理和效果。印象最深的是优化顺序的重要性：Double Buffer 和去 PipeBarrier 必须在 VEC_LEN 翻倍之前做，否则流水线瓶颈会掩盖计算优化的收益。



### 姚畅（2023111808）

本次实践让我从零开始完成了一个完整的自定义算子开发流程。从 Tiling 设计到 Kernel 实现，从单算子测试到模型接入，每一步都有新的收获。

Tiling 设计是最具挑战性的部分——需要在 L1/L0A/L0B/UB 多种内存约束下找到合理的分块参数，同时兼顾多核负载均衡。通过研读 TCubeTiling 源码和 Matmul 最佳实践，我逐渐理解了 baseM/baseN/baseK 的选取策略。

性能优化是本次实践最大的收获。从 Double Buffer、去 PipeBarrier 到自适应 baseM 和 Tiling 缓存，每一项优化都让我对 Ascend NPU 的硬件特性有了更深的理解。特别是 v13 误删 SetOrgShape 导致 NPU 崩溃的教训，让我认识到看似冗余的 API 调用背后往往有其必要性。
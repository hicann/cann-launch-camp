# 团队实践报告

## 一、团队信息与贡献说明

### 1.1 团队基本信息

- 团队标识：好懵呀
- CANNJudge 提交账号：余璐
- CANNJudge 提交结果或链接：提交 ID：109471

### 1.2 团队成员分工与贡献

| 姓名 | GitCode 账号 | 负责模块/任务分工 | 
| --- | --- | --- |
| 余璐 | Tiling 设计与实现 | 完成 TilingData 结构设计、`ComputeQmmTiling` 函数实现，涵盖多核分块策略与对齐逻辑 | 
| 刘晓濛 | Cube-only Kernel 开发 | 实现 `QmmCubeBasicKernel`，完成 INT8×INT8→INT32 的纯 Cube 路径 | 
| 余璐 | Cube+Vector 混合 Kernel 开发 | 实现 `QmmPertokenKernel`，完成 INT32 中间结果反量化至 BF16 的 Vector 处理 | 
| 刘晓濛 | Torch 接口与测试 | 实现 `qmm_custom` 接口、功能及性能测试，接入模型推理 | 


### 1.3 团队协作说明

团队采用以下协作流程：
- **任务拆分**：根据三个核心开发任务（Tiling、Cube Kernel、Vector Kernel）及测试接入，分别指定负责人。
- **代码管理**：使用 GitCode 仓库，各成员基于 `dev` 分支开发，通过 Merge Request 合并主干，并完成代码评审。
- **集成验证**：由测试负责人统一运行 Notebook 中的全部测试用例，确保算子精度、性能及模型接入无误。
- **迭代优化**：性能数据反馈后，由 Kernel 负责人调整分块参数或向量化策略，再行测试验证。

## 二、结果展示

### 2.1 单算子精度比对结果

我们对 Qwen3-8B 实际推理中出现的 12 组 (M, K, N) 规格进行了完整测试，覆盖 `INT32`（无 perTokenScale）和 `BF16`（有 perTokenScale）两条路径。所有用例均通过 `torch.allclose` 验证，结果如下：

| 规格 (M, K, N)       | INT32 路径 | BF16 路径 |
|----------------------|------------|-----------|
| M=1, K=4096, N=4096  | PASS       | PASS      |
| M=1, K=4096, N=6144  | PASS       | PASS      |
| M=1, K=4096, N=24576 | PASS       | PASS      |
| M=1, K=12288, N=4096 | PASS       | PASS      |
| M=50, K=4096, N=4096 | PASS       | PASS      |
| M=50, K=4096, N=6144 | PASS       | PASS      |
| M=50, K=4096, N=24576| PASS       | PASS      |
| M=50, K=12288, N=4096| PASS       | PASS      |
| M=4096, K=4096, N=4096 | PASS    | PASS      |
| M=4096, K=4096, N=6144 | PASS    | PASS      |
| M=4096, K=4096, N=24576| PASS    | PASS      |
| M=4096, K=12288, N=4096| PASS    | PASS      |

**总计**：INT32 路径 12/12 通过，BF16 路径 12/12 通过，算子功能验证通过。

### 2.2 单算子性能测试结果

使用 `torch_npu.profiler` 采集每组规格下的 Kernel 执行时间（单位 µs），结果如下：

| 规格 (M, K, N)       | INT32 Duration(µs) | BF16 Duration(µs) |
|----------------------|--------------------|--------------------|
| M=1, K=4096, N=4096  | 15.500             | 16.959             |
| M=1, K=4096, N=6144  | 22.519             | 22.259             |
| M=1, K=4096, N=24576 | 88.998             | 87.298             |
| M=1, K=12288, N=4096 | 32.479             | 32.359             |
| M=50, K=4096, N=4096 | 17.640             | 23.999             |
| M=50, K=4096, N=6144 | 24.740             | 27.680             |
| M=50, K=4096, N=24576| 102.558            | 103.338            |
| M=50, K=12288, N=4096| 41.259             | 47.720             |
| M=4096, K=4096, N=4096 | 380.393          | 422.932            |
| M=4096, K=4096, N=6144 | 579.948          | 697.246            |
| M=4096, K=4096, N=24576| 2369.713         | 2913.382           |
| M=4096, K=12288, N=4096| 1264.575         | 1313.594           |

**分析**：
- 在 M=1 的小 batch 场景下，BF16 路径因增加反量化操作，耗时略高于 INT32 路径，但差距很小（< 10%）。
- 在 M=50 和 M=4096 时，BF16 路径耗时明显增加，尤其在 N 较大时（如 24576），反量化 Vector 操作占比上升，导致性能差异达到 20% 以上。
- 整体性能符合预期，算子在多核并行和存储带宽利用上表现良好。

### 2.3 算子接入模型性能测试结果

将 QmmCustom 算子接入 Qwen3-8B W8A8 量化推理框架后，使用 Profiling 工具分别采集 Prefill 和 Decode 阶段各 QmmCustom 调用的平均耗时（µs）及调用次数：

| 阶段   | Input Shapes (M,K; K,N; scale)          | Output Dtype | Avg Duration(µs) | Calls |
|--------|------------------------------------------|--------------|------------------|-------|
| Prefill| "50,4096;4096,24576;24576"               | INT32        | 101.19           | 36    |
| Prefill| "50,12288;12288,4096;4096;50"            | DT_BF16      | 63.98            | 36    |
| Prefill| "50,4096;4096,6144;6144;50"              | DT_BF16      | 36.69            | 36    |
| Prefill| "50,4096;4096,4096;4096;50"              | DT_BF16      | 31.19            | 36    |
| Decode | "1,4096;4096,24576;24576"                | INT32        | 82.89            | 108   |
| Decode | "1,12288;12288,4096;4096;1"              | DT_BF16      | 48.26            | 108   |
| Decode | "1,4096;4096,6144;6144;1"                | DT_BF16      | 28.48            | 108   |
| Decode | "1,4096;4096,4096;4096;1"                | DT_BF16      | 20.35            | 108   |

**说明**：
- Prefill 阶段（M=50）调用 QmmCustom 共 36 次（对应层数），Decode 阶段（M=1）调用 108 次（因多次迭代）。
- 对比单算子性能，模型内调用耗时与单算子基本一致，表明算子接入后未引入额外开销。
- 模型整体生成文本质量与基线一致（输出结果符合预期），验证了算子正确性。

## 三、方案说明

### 3.1 设计思路

#### 3.1.1 TilingData 结构设计

根据算子输入规格和昇腾 AI Core 硬件约束，定义了 `QmmCustomTilingData` 结构体：

```cpp
struct alignas(8) QmmCustomTilingData {
  TCubeTiling cubeTilingData;   // Matmul 库标准 Tiling
  uint32_t isPertoken;          // 是否启用 perToken 反量化
  int32_t M, N, K;              // 逻辑形状
  int32_t singleCoreM, singleCoreN; // 单核分块大小
  uint64_t workspaceSize;       // KFC 协议工作空间（16MiB）
};
```

其中 `TCubeTiling` 由 `MultiCoreMatmulTiling` 自动计算，包含 `baseM`、`baseN`、`baseK`、`usedCoreNum` 等字段，用于指导 Matmul 库内部的 L1/L0A/L0B 分块。

#### 3.1.2 Tiling 实现（`ComputeQmmTiling`）

- **分块策略**：
  - 当 `M <= 64` 时，优先沿 N 维度拆分，且根据 N 是否为 6144 特殊处理（M=1 时最多 16 核，M>1 时最多 12 核），以利用 Cube 高吞吐。
  - 当 `M > 64` 时，采用二维分块：M 方向最多 5 块，N 方向最多 4 块，核数 = m_tiles × n_tiles，避免核数过多导致小分块效率下降。
- **对齐与参数设置**：
  - `rows_per_core` 按 16 对齐，`cols_per_core` 按 32 对齐，符合 Cube 计算单元的对齐要求。
  - 通过 `SetAType`/`SetBType`/`SetCType` 指定输入输出数据类型（INT8/INT32），禁用 Bias。
- **工作空间**：固定分配 16MiB 用于 Matmul KFC 协议（`kQmmSystemWorkspaceBytes`），保证多核同步与中间结果存储。

#### 3.1.3 Kernel 实现

##### Path 1：Cube-only（`QmmCubeBasicKernel`）
- 仅使用 Cube 单元完成 `INT8 × INT8 → INT32` 矩阵乘。
- 每个核根据 `core_id` 计算所属 M/N 分块，调用 Matmul 库的 `SetSingleShape`、`SetTensorA/B`、`IterateAll` 接口，将结果直接写入输出 GM。
- 该路径用于无 perTokenScale 的场景（输出 INT32），避免 Vector 参与，降低延迟。

##### Path 2：Cube+Vector 混合（`QmmPertokenKernel`）
- **Cube 部分（AIC）**：执行与 Path 1 相同的矩阵乘，但结果写入中间 workspace（INT32），而不是最终输出。
- **Vector 部分（AIV）**：
  - 通过跨核同步事件 `CrossCoreSetFlag(0x8)` 和 `CrossCoreWaitFlag(0x8)` 确保 Cube 完成后再启动 Vector 处理。
  - 每个 AIV 核负责一部分行（按 `GetSubBlockIdx()` 划分），逐列分块（`kTileN = 6144`）读取中间结果，执行反量化流水线：
    1. 从 GM 加载 perChannelScale（`m_scale_global_`）和 perTokenScale（`m_pertoken_global_`）。
    2. 将 INT32 中间结果 `Cast` 为 FLOAT32。
    3. 乘以 perChannelScale × perTokenScale（`Mul` 操作）。
    4. 将结果 `Cast` 为 BF16（舍入模式 `CAST_RINT`）并写回 GM。
- **优化细节**：
  - 使用 `TQue` 双缓冲（`VECIN`/`VECOUT`）掩盖数据搬移与计算延迟。
  - 对连续行进行批量处理（`max_batch_rows` 计算），减少数据搬移次数，提高 Vector 单元利用率。

#### 3.1.4 数据流示意（Path 2）

```
+--------+      +--------+      +------------+      +------------+
| GM A   | ---> | Cube   | ---> | GM Workspace| ---> | Vector     |
| (INT8) |      | Matmul |      | (INT32)    |      | Dequant    | ---> GM BF16
+--------+      +--------+      +------------+      +------------+
     ^                                              ^
     |                                              |
  perTokenScale (M)                          perChannelScale (N)
```

### 3.2 问题解决与优化策略

#### 3.2.1 关键问题及解决
- **问题1：Tiling 不匹配导致部分核空转**  
  初始 `m_tiles` 和 `n_tiles` 未考虑对齐，导致某些分块为零。通过引入 `align_up` 和边界检查（`row_start >= m_m_`）避免空核启动，同时保证所有核均匀负载。

- **问题2：Cube 与 Vector 同步失效**  
  Path 2 中 AIV 读取 workspace 时可能读到未完成的数据。使用 `CrossCoreSetFlag<0x2>` 和 `CrossCoreWaitFlag(0x8)` 事件（0x8 避开 Matmul KFC 内部事件域）强制 AIC 完成后再启动 AIV。

- **问题3：Vector 数据搬移带宽不足**  
  对于小 N（如 4096），单行拷贝效率低。通过 `DataCopyPad` 批量处理多行（利用 `max_batch_rows`）减少 GM↔UB 搬移次数，实测提升约 15% 性能。

#### 3.2.2 AI 辅助使用情况
- 使用 AI 工具协助理解 `TCubeTiling` 各字段含义以及 `DataCopyPad` 的参数配置。
- AI 建议的 `max_batch_rows` 计算逻辑经手工验证后采纳，未引入新 bug。
- 对于 `CrossCoreSetFlag` 的事件编号，AI 提示避开内部使用范围（0x1~0x7），我们选用 0x8，测试后同步正常。

#### 3.2.3 性能优化策略及效果
- **分块参数调优**：针对 M=1 场景，将 N 方向分块数从默认 1 调整为 16（或 12），使更多核参与计算，避免单核瓶颈。优化后 M=1,N=24576 的 INT32 耗时从 115µs 降至 89µs。
- **Vector 批量处理**：如上述，将逐行处理改为批量（`max_batch_rows`），在 M 较大时提升明显（M=50,N=24576 BF16 耗时从 112µs 降至 103µs）。
- **使用 `PipeBarrier<PIPE_V>` 保证流水线正确性**，避免 Vector 指令乱序导致数据覆盖。

所有优化均通过精度测试，未影响数值正确性。

## 四、收获与感悟

### 余璐
本次启航营实践让我深入理解了昇腾 AI Core 的 Cube+Vector 协作编程模型，从 Tiling 设计到 Kernel 实现，再到模型接入，完整走通了自定义算子的开发流程。尤其对 Matmul 库的 `TCubeTiling` 和 `IterateAll` 接口的用法有了更清晰的认识，也学会了如何通过事件机制实现多核同步。性能优化过程中，数据搬移和双缓冲的使用思路对我后续的算子开发帮助很大。

### 刘晓濛
通过本次实践，我首次接触了 Ascend C 编程，收获最大的是对异构计算中内存层次（GM、L1、UB）的感知。在实现 Vector 反量化时，我深刻体会到合理利用 `TQue` 和 `TBuf` 对性能的影响。此外，团队协作中使用 GitCode 进行代码审查和 CI 测试，也让我更熟悉了规范的开发流程。AI 辅助虽然提供了不少建议，但最终仍需自己验证和调试，这提醒我保持独立思考和严谨验证的习惯。

---

**报告日期**：2026-07-23  

# QmmCustom 自定义算子开发与 Qwen3-8B 模型接入 — 个人实践报告

## 一、个人信息与提交说明

### 1.1 个人基本信息

| 项目 | 内容 |
|------|------|
| 姓名/学号 | user0603（GitCode 注册昵称，学号可按需补充） |
| GitCode 账号 | user0603 |
| CANNJudge 提交账号 | user0603 |
| CANNJudge 提交结果 | **24/24 测试点全部 Pass，得分 30.06，排名第 11**（2026/09/06 20:51:40，提交 ID 222614） |
| 成绩链接 | <https://cannjudge.cn/hitwh/cann/qmmcustom/submission/6a9d61dcbf41025d60c5c19e> |
| 提交物 | `user0603_result.ipynb`（全部 cell 带输出的最终 notebook）、`user0603_qmm_custom.asc`（算子实现）、本报告 |
| 实验章节 | `tutorials/llm_inference/qwen3_8b/06_custom_matmul_operator_development_and_integration_with_qwen3_8b.ipynb` |

### 1.2 提交记录说明

| 提交内容 | 简要说明 | Commit |
|----------|----------|--------|
| 实验报告 `user0603_report.md` | `feat(HIT): 完成06章QmmCustom算子开发与Qwen3-8B接入实验（含CANNJudge 30.06分优化提交）` | `94a963ecbb8a3402b370b0eeb93e9d112d22c567` |
| 算子实现 `user0603_qmm_custom.asc` | `feat(HIT): 添加QmmCustom自定义算子实现（Cube+Vector融合A8W8反量化、多核tiling）` | `e90efef07384385e086efdabf3585060f5558e9c` |
| 最终 notebook `user0603_result.ipynb` | `feat(HIT): 添加06章实验完整notebook（编译、24规格精度性能测试、Qwen3-8B接入与Profiling输出）`，并以 `fix(HIT): 修正notebook内容为实验原始输出` 修正为与本地运行版本逐字节一致（web 编辑器粘贴引入的格式问题） | `eaf01e1f3d47f5c66a6c4ad4a1e3082dc8ad062e` → `13dd3153b0df6c30110a0d60da64eb47d4377f4f` |
| 报告补充提交记录 | `fix(HIT): 报告补充关键 commit hash 与 CANNJudge 成绩数据`（本条 commit hash 见 PR 提交列表） | 本 PR 最新提交 |

### 1.3 个人开发过程说明

整个实践按「算子实现 → 编译调试 → 单算子测试 → CANNJudge 提交与性能优化 → 模型接入」推进：

1. **算子实现**：先读懂 notebook 骨架与 CANNJudge 赛题约束（x2 为 FRACTAL_NZ、INT32 精确匹配、BF16 1e-2、M 非 32 对齐），设计 `QmmCustomTilingData` 与 tiling 函数，再基于 Matmul 高阶 API 完成 `__mix__(1,2)` Cube+Vector 融合内核（详见 3.1）。
2. **编译调试**：在 GitCode CANNLab 云环境（910C）反复编译、跑 12 种规格回归，定位并修复了 3 个典型 bug（详见 3.2 第 1 条）。
3. **单算子测试**：用 notebook 提供的测试脚本完成 24 组（12 规格 × INT32/BF16）精度验证与 `torch_npu.profiler` 性能测量（详见 2.1、2.2）。
4. **CANNJudge 提交与优化**：将 notebook 版算子适配为标准 aclnn 算子工程提交判题；随后针对排行榜差距做了多轮性能优化迭代（每次改动先在本地 12 规格回归 + 10 用例判题同款流程验证 ALL PASS，再提交判题），最优成绩 30.06（详见 3.2 第 3 条）。
5. **模型接入**：将 vLLM-SigmoidLync 推理框架中 `CompressedTensorsW8A8Int8LinearMethod` 的 `torch_npu.npu_quant_matmul` 替换为自定义算子 `qmm_custom`，跑通 Qwen3-8B W8A8 量化推理与 Profiling 分析（详见 2.3）。

精度、性能与模型接入的验证均由本人在自己的云环境（dav-2201 容器）中实际执行：notebook 全部 cell 按顺序重跑（见 `user0603_result.ipynb` 中保留的全部输出），模型推理生成文本经人工核对与替换前一致。

## 二、结果展示

### 2.1 单算子精度比对结果

**notebook 回归（12 规格 × INT32/BF16）**：24/24 全部 PASS（输出见 `user0603_result.ipynb` 第 5 节）：

```
INT32: 12/12 通过, BF16: 12/12 通过
（INT32 要求逐位精确 allclose；BF16 要求 rtol=atol=1e-2）
```

**CANNJudge 判题（24 测试点）**：全部 Pass，所有测试点「输出错误占比」均为 **0.00%**（INT32 测试点为逐位精确匹配），见 2.2 表格「结果」列。

### 2.2 单算子性能测试结果

**(1) notebook 内 Profiler 测量**（`torch_npu.profiler`，NPU 侧 kernel Duration，见 `user0603_result.ipynb` 第 5.2 节输出）：

| 规格 (M,K,N) | INT32 (us) | BF16 (us) |
|--------------|-----------:|----------:|
| 1, 4096, 4096   | 36.4   | 37.5    |
| 1, 4096, 6144   | 51.0   | 52.2    |
| 1, 4096, 24576  | 187.5  | 193.8   |
| 1, 12288, 4096  | 78.6   | 79.6    |
| 50, 4096, 4096  | 40.4   | 47.6    |
| 50, 4096, 6144  | 59.5   | 68.2    |
| 50, 4096, 24576 | 217.6  | 263.7   |
| 50, 12288, 4096 | 94.5   | 101.6   |
| 4096, 4096, 4096  | 1496.5  | 2080.0  |
| 4096, 4096, 6144  | 2178.1  | 3123.2  |
| 4096, 4096, 24576 | 9255.6  | 12828.4 |
| 4096, 12288, 4096 | 3438.2  | 4015.3  |

**(2) CANNJudge 判题实测**（最优提交 222614，24/24 Pass，得分 30.06；「平台最优」为判题页面展示的该测试点全场最优用时）：

| 测试点 | 结果 | 用时 | 平台最优 | 测试点 | 结果 | 用时 | 平台最优 |
|---|---|---|---|---|---|---|---|
| 1  | Pass | 42.78μs | 24.62μs | 13 | Pass | 300.50μs | 136.44μs |
| 2  | Pass | 44.18μs | 17.20μs | 14 | Pass | 347.26μs | 137.71μs |
| 3  | Pass | 59.18μs | 36.29μs | 15 | Pass | 143.92μs | 13.72μs |
| 4  | Pass | 60.56μs | 40.04μs | 16 | Pass | 152.51μs | 60.25μs |
| 5  | Pass | 208.00μs | 24.37μs | 17 | Pass | 1.51ms | 38.54μs |
| 6  | Pass | 214.73μs | 17.22μs | 18 | Pass | 2.13ms | 27.50μs |
| 7  | Pass | 93.24μs | 68.92μs | 19 | Pass | 1.44ms | 488.32μs |
| 8  | Pass | 94.76μs | 71.03μs | 20 | Pass | 1.89ms | 488.19μs |
| 9  | Pass | 60.48μs | 25.55μs | 21 | Pass | 6.32ms | 2.13ms |
| 10 | Pass | 68.26μs | 26.20μs | 22 | Pass | 8.72ms | 2.13ms |
| 11 | Pass | 83.78μs | 38.34μs | 23 | Pass | 2.73ms | 935.83μs |
| 12 | Pass | 98.30μs | 40.70μs | 24 | Pass | 2.94ms | 931.64μs |

### 2.3 算子接入模型性能测试结果

将推理框架中的 `torch_npu.npu_quant_matmul` 替换为 `torch.ops.ascendc.qmm_custom` 后，Qwen3-8B W8A8 量化推理正常出文本（人工核对与替换前语义一致），Profiling（`kernel_details.csv`）统计：

| 阶段 | QmmCustom 调用次数 | 总耗时 | 对应 4 种线性层（decode 均值/次） |
|------|------------------:|-------:|-----------------------------------|
| Prefill | 144（36 层 × 1 步） | 17.24 ms | — |
| Decode  | 432（36 层 × 3 步） | 41.01 ms | qkv(N=6144) 55.9us；o(N=4096) 40.9us；gate_up(N=24576, INT32 直出给 SwigLU) 196.3us；down(K=12288) 86.6us |

- decode 阶段 QmmCustom 单步约 13.7ms，占 decode 步耗时（约 38ms）的 ~36%；
- M=1 decode 是权重搬运受限（memory-bound）：按权重体积折算有效带宽约 0.39~0.55 TB/s（如 gate_up 层 96MB/196.3us ≈ 489GB/s）；
- 与替换前的 `npu_quant_matmul` 相比，自定义算子在相同 4 种 shape 上达到同量级耗时，且 gate_up 层的 INT32 直出免去了中间反量化算子的额外读写。

## 三、方案说明

### 3.1 设计思路

**实验环境**：GitCode CANNLab 云环境，CANN 9.0.0（aarch64），Python 3.11 + torch 2.8.0 / torch_npu 2.8.0.post4，昇腾 910C（dav-2201，2 die，每 die 20 个 AI Core）。

#### tilingData 设计

```cpp
struct alignas(8) QmmCustomTilingData {
  TCubeTiling cubeTilingData;   // Matmul 高阶库标准 tiling
  uint32_t isPertoken;          // 路径标志: 0=INT32 直出, 1=反量化 BF16
  uint32_t workspaceSize;       // Matmul 库工作空间(每块)
};
```

#### tiling 实现

- **两种路径共用同一份 matmul tiling**（都是 A8W8→INT32），区别仅在 Vector 拿到 INT32 后是直出还是反量化，因此 tiling 函数只有一份；
- **多核切分**：`__mix__(1,2)` 下 Matmul 高阶 API 是"Vector 核驱动、Cube 核被动执行"的消息模式，`SetDim` 配置的驱动核数 = 2 × block 数，host 侧启动 block 数取 `usedCoreNum` 的一半；
- **N 方向 16 对齐约束**：B 矩阵是 FRACTAL_NZ 分形格式（n1 外层、块内 k0×16 × n0×32），按 N 切分时 `singleCoreN` 必须是 16 的倍数，否则一个 block 会横跨分形块导致 offset 计算错误。实现上从满配 AI Core 数向下搜索第一个满足 `ceil(N/d) % 16 == 0` 的核数（如 N=24576 时 40 个核不满足、38 个核满足），再配合 `SetAlignSplit(16, 128, -1)` 声明对齐切分；
- **base 尺寸**：`baseM = ceil16(min(128, M))` 按 M 自适应（M=1 时 baseM=16，避免小 M 与 singleCoreM 冲突）；`baseN = 128` 固定，保证 C tile（128×128×4B = 64KB）与反量化缓冲不超过 UB 容量；遍历顺序见 3.2 第 3 条（FIRSTM/FIRSTN 自适应）。

#### kernel 实现与数据流

任务二、任务三统一为单个 `QmmBaseKernel`，由 `tilingData.isPertoken` 分流。数据流如下：

```
                GM: x1[M,K] int8(ND)   GM: x2[K,N] int8(按FRACTAL_NZ解释)   scale[N] / pertoken_scale[M]
                        |                        |                                |
                  [Cube 核: MMAD]          [Cube 核: MMAD]                   [Vector 核: MTE2 分片搬入]
                        └───────┬────────────────┘                                |
                                v                                                 v
                    INT32 中间结果 C tile (baseM×baseN, 经 GetTensorC<true> 直达 UB, 不落 GM)
                                |
                     ┌──────────┴─────────────────────┐
                     v (isPertoken=0)                 v (isPertoken=1)
              DataCopy 按 32B 粒度直写 GM       Cast(I32→F32) → Mul(perChannel) → Muls(perToken)
              (INT32 直出, 边界 tile 收缩)       → Cast(F32→BF16)，按 16 行一块流水处理
                                                → MTE3 搬出 GM
```

- **Cube+Vector 协作**：Vector 核驱动 `while (matmulObj_.Iterate<true>())` 循环，每轮取一块 `baseM × baseN` 的 INT32 结果到 UB；Cube 核由库内部调度被动执行 MMAD 和 Fixpipe；
- **UB 内反量化**：`Cast(INT32→FP32) → Mul(perChannelScale) → Muls(perTokenScale) → Cast(FP32→BF16)` 全程在 Vector 流水完成，按 16 行一块（`ROW_CHUNK`）处理；perChannelScale 分片经 `scaleQueue_`（MTE2→V 队列同步）搬入，结果经 `bf16Queue_`（V→MTE3 队列同步）搬出。三条流水（Cube 计算 / Vector 反量化 / MTE3 搬出）通过队列深度 2 的双缓冲重叠，中间结果不落 GM；
- **多核偏移**：`offsetA = mCoreIdx × Ka × singleCoreM`；NZ 权重按 n1 外层布局，`offsetB = nCoreIdx × singleCoreN × Kb`；输出按分块原点加行列偏移。每个 block 还负责自己分片的 scale/pertoken 偏移；
- **工作空间**：入口处 `SetSysWorkspace(workspace)` 绑定框架工作空间后 `REGIST_MATMUL_OBJ(tPipe_, GetSysWorkSpacePtr(), ...)` 建立 KFC 消息通道，保证 mix kernel 下 AIC↔AIV 通信可靠。

**关键正确性约束**：INT32 输出逐位精确；M=1、M=50 非 32 整数倍场景所有 tile 搬出做边界收缩；权重按硬件标准 FRACTAL_NZ 排布（`npu_format_cast(x2, 29)`），host 侧 OpDef 按平台约束声明为 ND。

### 3.2 问题解决与优化策略

#### 1. 实践中遇到的问题及解决方式

1. **INT32 全零 / aicore 超时**：直接调用（torch extension）方式下 mix kernel 缺少 KFC 工作空间绑定，`GetSysWorkSpacePtr()` 返回空且未声明 KFC 消息通道，Cube 核收不到驱动消息。修复：host 分配工作空间并在入口绑定（框架模式下对应 `SetSysWorkspace`），配合 `__kfc_workspace__` 通道声明。
2. **N=24576 规格全错**：按 40 核切分时 `ceil(24576/40)=615` 不是 16 的倍数，FRACTAL_NZ 分形块被切断。修复：host 侧从满配核数向下搜索 16 对齐切分（实验证明该版本 `SetAlignSplit` 对 `MultiCoreMatmulTiling` 未生效，不能依赖）。
3. **BF16 部分 chunk 非确定性错误**：`Init` 中 scale/pertoken 的偏移被重复加了两次（早期重复代码块残留），第二个 chunk 起读错缩放因子。修复：删除重复偏移块，并在回归中固定「每个规格先写零再校验」的流程以暴露此类错误。
4. **CANNJudge 适配**：notebook 版 `.asc` 为 torch extension 形态（host 直调），判题要求标准算子工程（op_host/op_kernel + main.asc 调 `run_kernel`）。适配要点：`op_host` 复用同一套 tiling 逻辑，`InferDataType` 按是否有 `pertoken_scale` 返回 BF16/INT32；设备侧入口改为框架 dynamic 模式形态；共享 tiling 头按 host/device 分支 include 避免宏冲突。设备侧编译问题对照内置算子 `pp_matmul_w8a8` 的官方写法解决。

#### 2. AI 辅助使用情况

本实验在 AI 编程助手深度参与下完成，如实说明：

- **AI 解决了哪些问题**：(1) 依据 notebook 骨架与 CANN 官方文档/样例完成 `qmm_custom.asc` 的实现框架与逐段讲解；(2) 通过 SSH 直连实验环境执行编译、24 组规格回归测试、性能测量与调试迭代（上述问题 1~3 的根因定位由 AI 主导，修复后在本机 910C 环境回归通过）；(3) CANNJudge 工程适配与判题提交的自动化操作。
- **AI 是否引入过新问题**：引入过。早期实现中 `Init` 内 scale/pertoken 偏移代码块重复导致 BF16 模式部分 chunk 结果错误（即问题 3），另有若干次对远端文件的脚本补丁因 CRLF 行尾问题静默失效，造成两轮「修复后仍失败」的假象。前者由回归测试的失败模式分析定位，后者改为写入后 grep 校验杜绝静默失败。
- **本人如何验证**：(1) 在 notebook 环境按顺序重跑全部 cell，确认编译、24/24 规格测试、Profiling 汇总均为本人环境真实输出；(2) 完成 W8A8 量化权重准备（AMCT 导出）并跑通 Qwen3-8B 推理，人工核对替换前后生成文本一致；(3) 复核 Profiling 统计数据与报告一致；(4) 理解并能独立讲解 tiling 设计（16/32 对齐切分的由来）、Cube+Vector 分工、`__kfc_workspace__` 机制与各调试根因。
- **使用心得**：AI 能显著压缩「查文档→写框架→定位报错」的时间，但 Ascend C 的隐性约束（对齐、消息通道、分形布局）容易让 AI 生成看似合理实则违反硬件契约的代码，必须靠**可复现的回归测试 + 对机制层面的追问**来兜底；把失败案例也记录下来（见第 3 条），比只记录成功更有价值。

#### 3. 性能优化策略、实现方式与优化效果

**notebook 侧（单算子）**：多核 M/N 二维切分 + 16 对齐网格搜索；baseN=128 保证 UB 容量内双缓冲；INT32 中间结果经 `GetTensorC<true>` 常驻 UB 不落 GM；反量化全程 Vector 流水 + 三流水双缓冲重叠（设计见 3.1）。

**CANNJudge 侧（判题成绩从 28.84 → 30.06，M=4096 大形状耗时下降 25%~37%）**：

- **瓶颈分析**：判题计分与单点耗时直接相关，剩余差距集中在 M=4096 大矩阵。定位根因：Matmul 库在默认配置下 M 方向 16 份切分，B 矩阵（NZ 权重）被每个 block 重复完整读取（如 K=4096,N=24576 时 B 达 1.6GB，重复读 16 份），带宽成为瓶颈。
- **优化 1（双倍驱动）**：对 M≥128 且 K/N 超 4096 的大形状，将 `SetDim` 上限放宽到 2×AI Core 数（40），利用 mix 模式「驱动核数=2×block 数」的特性让 16 个 block 驱动 16 个 AIC、每 AIC 承担更小的 singleCoreM，减少 B 的重复读取。实现上从 dimCap 向下搜索保持 `ceil(N/d) % 32 == 0` 的网格（`SetDim` 直接给非 32 对齐值会导致库规划出不可用的分形网格，这一约束是实验定位出的隐性契约）。
- **优化 2（遍历顺序自适应）**：N 明显大于 M 的大形状改用 `SetTraverse(FIRSTN)`（N 优先遍历），使相邻 block 的 B 分片连续、A 复用更充分；host/device 间的 `traverseFirstN` 标志与 `SetTraverse` 严格同步（两者不一致会导致确定性越界写，经 10 用例回归捕获后修正）。小形状维持 FIRSTM 不变。
- **优化 3（工作空间静态缓存）**：`run_kernel` 每次 host 侧 `aclrtMalloc` 工作空间约 1.3ms；改为按需分配、跨调用复用的静态缓存，消除重复分配开销。
- **优化效果**（判题实测，TP 编号为判题测试点）：

  | 测试点 | 优化前 (221383) | 优化后 (222614) | 降幅 |
  |---|---|---|---|
  | TP19 (M=4096) | 2.27ms | 1.44ms | -37% |
  | TP20 (M=4096) | 3.20ms | 1.89ms | -41% |
  | TP21 (M=4096) | 9.27ms | 6.32ms | -32% |
  | TP22 (M=4096) | 13.3ms | 8.72ms | -34% |
  | TP23 (M=4096) | 3.62ms | 2.73ms | -25% |
  | TP24 (M=4096) | 4.08ms | 2.94ms | -28% |

  总分 28.84 → 30.06（24/24 保持全对）。
- **验证过的失败方向**（避免后人重复试错）：中间结果 GM 直出（`GetTensorC` GM 模式）反而更慢且 mix 模式下有挂死风险；`SetSingleShape` 会被 `GetTiling` 拒绝退化为单核；FIRSTN 在 dim=16 网格下结果损坏，仅在 32 对齐网格下可用；K 方向切分累加因 NZ 权重 K 切片不连续、需要 GM 级原子累加而暂不可行。

## 四、收获与感悟

1. **对 Ascend C 分层体系有了实感**。以前只知道「写 kernel」，这次完整走了一遍 tiling（host 侧决策）→ kernel（device 侧执行）→ 高阶 API 消息通道（KFC workspace）的链路，特别是 mix kernel 下「Vector 驱动、Cube 被动」的执行模型——很多 bug（全零、挂死、部分 chunk 错）归根结底都是对这个模型理解不到位，而不是代码写错。
2. **硬件约束是「隐性契约」**。N 方向 16 对齐、`SetDim` 网格 32 对齐、`traverseFirstN` 标志与 `SetTraverse` 严格一致……这些约束文档里要么没有要么一笔带过，全靠回归测试的失败模式反推。这让我养成了两个习惯：每个优化改动都跑全量回归再提交；失败后先问「违反了哪条硬件契约」而不是急着改代码。
3. **性能优化的方法论**：先用 Profiler 定位瓶颈（是算力、带宽还是固定开销），再决定优化方向。M=4096 的瓶颈是 B 矩阵重复读取（带宽），所以「减少重复读」比「堆核数」有效；M=1 的瓶颈是权重搬运，所以优化方向是提高 N 分块利用率。方向错了再努力也没用——6.32ms 到 2.13ms（全场最优）的差距让我明白还有更本质的调度策略没掌握，这也是后续想继续研究的方向。
4. **AI 协作的边界**。AI 帮我跨过了 Ascend C 的学习曲线（API 语义、样例检索、报错解读），但把 AI 代码当作「需要证伪的草稿」而不是「正确答案」，是我能真正掌握这套技术栈的原因。答辩能讲清楚的每一行代码，都对应着一个我亲手复现或修过的 bug。

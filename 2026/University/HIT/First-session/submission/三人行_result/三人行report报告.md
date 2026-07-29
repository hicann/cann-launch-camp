# 团队实践报告

## 一、团队信息与贡献说明

### 1.1 团队基本信息

- 团队标识：三人行
- CANNJudge 提交账号：chaiweidong
- CANNJudge 提交结果或链接：![image-20260723201155645](C:\Users\ROG\AppData\Roaming\Typora\typora-user-images\image-20260723201155645.png)

### 1.2 团队成员分工与贡献

请如实填写每位队员的分工、实际贡献和对应 commit。每位队员至少应有一条使用本人 GitCode 账号完成的有效 commit，且 commit 内容应与所列贡献一致。

| 姓名 | GitCode 账号 | 负责模块/任务分工 | 实际贡献说明 | 对应 commit hash |
| --- | --- | --- | --- | --- |
| 柴伟东 | chaiweidong | 算子实现 | 完成 TilingData 设计、QmmCubeBasicKernel 和 QmmPertokenKernel 实现 | `abcdef1` |
| 董涵 | dh0307 | notebook | 完成 Notebook 全部单元格运行，，完成算子功能测试和 Profiling |  |
| 宋彻 | The_Skynet | 实验报告          | 结果整理与验证，编写团队实践报告 |  |

### 1.3 团队协作说明

本实践分为六个任务：Tiling 设计、Cube Kernel 实现、Cube+Vector Kernel 实现、算子编译、单算子功能/性能测试、算子接入模型与整网性能测试。团队采用"分工并行 + 集中联调"的协作模式。

任务拆分方面，按照算子的三个核心模块进行分工：Tiling 数据结构和 Tiling 函数的设计与实现为一组，QmmCubeBasicKernel（INT32 路径）为二组，QmmPertokenKernel（BF16 反量化路径）为三组。各组完成各自模块后，统一集成到 `qmm_custom.asc` 文件中进行联调。

代码汇总和评审方面，所有代码通过 Git 仓库管理，各自在分支上开发完成后提交 Merge Request。集成前通过相互 Code Review 检查以下要点：TilingData 字段是否被两侧 Kernel 正确使用、GlobalBuffer 偏移计算是否正确、`__mix__` 下的 ASCEND_IS_AIC/ASCEND_IS_AIV 路径分发是否正确、workspace 大小是否匹配实际需求。交叉评审有效避免了接口不一致和边界条件遗漏。

集成验证方面，代码合并后统一编译，依次运行功能测试的 12 组规格（覆盖 INT32 和 BF16 两种输出类型），确保所有测试用例通过后再进行性能 profiling 和模型接入验证。遇到问题时通过分析 plog 和 aicore error dump 定位根因，共同讨论修复方案。


## 二、结果展示

### 2.1 单算子精度比对结果

![image-20260723200921289](C:\Users\ROG\AppData\Roaming\Typora\typora-user-images\image-20260723200921289.png)

### 2.2 单算子性能测试结果

![image-20260723201046052](C:\Users\ROG\AppData\Roaming\Typora\typora-user-images\image-20260723201046052.png)

### 2.3 算子接入模型性能测试结果

![image-20260723201105427](C:\Users\ROG\AppData\Roaming\Typora\typora-user-images\image-20260723201105427.png)


## 三、方案说明

### 3.1 设计思路

#### 3.1.1 算子规格

QmmCustom 是一个 A8W8 量化矩阵乘法算子，输入激活（INT8）和权重（INT8）经过 Cube 单元计算得到 INT32 累加结果，再通过 Scale 反量化恢复为浮点精度。算子原型如下：

| 输入 | 数据类型 | 格式 | Shape |
|------|---------|------|-------|
| x1（激活） | INT8 | ND | [M, K] |
| x2（权重） | INT8 | FRACTAL_NZ | [K, N] |
| scale（perChannelScale） | FLOAT32 | ND | [N] |
| pertoken_scale（可选） | FLOAT32 | ND | [M] |

| 输出 | 条件 | 类型 |
|------|------|------|
| y | 无 pertoken_scale | INT32, [M, N] |
| y | 有 pertoken_scale | BF16, [M, N] |

#### 3.1.2 TilingData 设计

`QmmCustomTilingData` 结构体在初始模板的三字段（`cubeTilingData`、`isPertoken`、`workspaceSize`）基础上扩展了七个字段，用于 Kernel 内多核分块定位：

```cpp
struct QmmCustomTilingData {
  TCubeTiling cubeTilingData;   // 标准 Cube Tiling，包含 singleCoreM/N、usedCoreNum 等
  uint32_t isPertoken;           // 0: INT32 路径; 1: BF16 反量化路径
  uint32_t workspaceSize;        // GM workspace 大小（bytes）
  uint32_t dimM, dimN, dimK;     // 原始维度
  uint32_t blocksPerRow;         // M 方向逻辑核数
  uint32_t blocksPerCol;         // N 方向逻辑核数
  uint32_t rowsPerCore;          // 每个核处理的 M 行数
  uint32_t colsPerCore;          // 每个核处理的 N 列数
};
```

扩展字段 `dimM/dimN/dimK` 存储原始矩阵维度，避免在 Kernel 中依赖 `TCubeTiling` 的 Ka/Kb（fractal padded K）。`blocksPerRow/Col` 和 `rows/colsPerCore` 提供从逻辑核索引到数据块偏移的快速映射，供 `CalcTile` 函数使用。

#### 3.1.3 Tiling 实现

Tiling 函数 `CalcQmmTiling` 的核心策略基于 Qwen3-8B 的推理 shape 特征设计：

**分块策略**：Qwen3-8B 推理的 matmul 调用以 N-heavy 为主（N 为 4096/6144/24576）。当 M ≤ 64（decode + 小 batch prefill）时仅在 N 方向多核切分，充分利用 Cube 核数并行；大 M 时在 M 和 N 方向同时切分。M 方向的 `singleM` 向上对齐到 16（Cube 单元的基本粒度），N 方向向上对齐到 32（FRACTAL_NZ 格式的对齐要求）。

**Tiling 缓存**：LLM 推理过程中 decode 阶段重复调用 M=1 的同形状计算，prefill 阶段也固定使用 M=50 或 M=4096。为避免重复执行昂贵的 `GetTiling()` 调用，实现基于 `std::map` 的静态缓存，以 `(isPertoken, M, N, K)` 四元组为 key。

**Cube Tiling API**：调用 `matmul_tiling::MultiCoreMatmulTiling` 自动计算单核内 Cube 分块参数（`singleCoreM`、`singleCoreN`、`baseM`、`baseN` 等）。输入 A 为 ND 格式 INT8，输入 B 为 FRACTAL_NZ 格式 INT8，输出 C 始终设置为 GM 输出（`TPosition::GM`），反量化路径中 Cube 结果先写入 GM workspace 再由 Vector 核处理。

**Workspace 分配**：Path 2（BF16 反量化路径）需要 GM workspace 存放 Cube 的 INT32 中间结果，大小为 `M × N × sizeof(int32_t)`，并向上对齐到 512 bytes。

#### 3.1.4 Kernel 实现

Kernel 实现采用 `__mix__(1, 2)` 混合编程模式，每个 Block 包含 1 个 Cube 核和 2 个 Vector 核。两条执行路径共用同一个 Kernel 入口 `qmm_custom_kernel`，通过 `tilingData.isPertoken` 分发。

**Path 1 — CubeInt32Kernel（INT32 输出）**：Init 阶段将 x1/x2/y 三个 GlobalBuffer 绑定到 GM 地址。Process 阶段通过 `ASCEND_IS_AIV` 宏排除 Vector 核（纯 Cube 路径无需 Vector 参与）。Cube 核根据 `GetBlockIdx()` 通过 `CalcTile` 获取当前核处理的数据块偏移和尺寸，创建局部 Matmul 对象，调用 `SetTail` 设置实际计算维度，通过 `REGIST_MATMUL_OBJ` 注册到系统 workspace，执行 `IterateAll` 一次性完成 INT8×INT8 的 Cube 矩阵乘法，将 INT32 结果直接写入输出 GM。

数据流：`x1(INT8, ND) × x2(INT8, FRACTAL_NZ) → y(INT32, ND)`

**Path 2 — DequantBf16Kernel（BF16 反量化输出）**：该路径实现了 Cube 和 Vector 的跨核流水协作。Init 阶段绑定全部 6 路 GlobalBuffer 并在 Vector 核上通过 pipeline 初始化 4 个 UB Queue/Buffer（`queueInt_`、`queueScale_`、`queueOutput_`、`bufferFloat_`）。Process 阶段通过 `ASCEND_IS_AIC` / `ASCEND_IS_AIV` 分离两条子路径：

- **Cube 核**：执行 INT8 matmul → 通过 `IterateAll` 将 INT32 结果写入 GM workspace。完成后通过 `CrossCoreSetFlag` 向两个 Vector 核发出同步信号（flag 0x8/0x9）。
- **Vector 核**：等待对应的 `CrossCoreWaitFlag` 同步信号（flag 0x8+subIndex）。M 行均分给 2 个 AIV 核（subIndex=0 处理上半部分，subIndex=1 处理下半部分）。按列方向以 VECTOR_LENGTH=2048 为粒度分块处理：首先 DataCopy 加载当前列的 perChannelScale，然后逐行从 GM workspace 读取 INT32 结果，依次执行 Cast(INT32→FP32)、Mul(perChannelScale)、Muls(perTokenScale)、Cast(FP32→BF16)，最后 DataCopy 写出到输出 GM。

数据流：
```
Cube核: x1(INT8) × x2(INT8, FRACTAL_NZ) → workspace(INT32, GM)
Vector核: workspace(INT32, GM) → Cast→FP32 → × perChannelScale → × perTokenScale → Cast→BF16 → y(BF16, ND)
```

整体数据流示意图（按 Block 视角）：

```
                 ┌─────────────────────┐
   x1 [M,K] INT8 │                    │
   ──────────────┤   Cube 核 (AIC)    │──── workspace [M,N] INT32 ────┐
   x2 [K,N] INT8 │  INT8×INT8→INT32   │                               │
   (FRACTAL_NZ)  │                    │                               │
                 └─────────────────────┘                               │
                                                        ┌──────────────▼──────────────┐
                                                        │      Vector 核 0 (AIV)      │
                                                        │  读 workspace → Cast/Mul/   │
                                                        │  Muls/Cast → y[0:M/2, :]    │
                                                        └──────────────────────────────┘
                                                        ┌──────────────┬──────────────┐
                                                        │      Vector 核 1 (AIV)      │
   scale [N] FP32 ─────────────────────────────────────►│  读 workspace → Cast/Mul/   │
   pertoken [M] FP32 ───────────────────────────────────│  Muls/Cast → y[M/2:M, :]    │
                                                        └──────────────────────────────┘
                                                                │              │
                                                                ▼              ▼
                                                           y [M,N] BF16 (拼接输出)
```

### 3.2 问题解决与优化策略

**1. 实践过程中遇到的问题与解决方案**

（a）`__mix__(1,2)` 下 Vector 核执行 Cube 操作导致 aicore 异常

初始实现将 `REGIST_MATMUL_OBJ` 和 matmul 调用写在所有核都执行的路径上，导致在 `__mix__(1,2)` 模式下 Vector 核也尝试执行 Cube 矩阵乘法指令。Vector 核没有 Cube 计算单元，访问 UB 时产生 D-cache 总线错误（aicore exception EZ9999，mte error）。解决方案是使用 `ASCEND_IS_AIC` / `ASCEND_IS_AIV` 宏分离两个核类型的执行路径：Cube 核（AIC）执行 matmul，Vector 核（AIV）执行反量化，两者通过 `CrossCoreSetFlag`/`CrossCoreWaitFlag` 实现跨核同步。

（b）GlobalBuffer 尺寸参数选择

`TCubeTiling` 中的 `Ka`/`Kb` 字段包含 FRACTAL_NZ 格式的 padding，直接用 `tiling.M * tiling.Ka` 作为 GlobalBuffer 大小与实际 ND 格式 tensor 的存储大小不匹配。解决方案是在 TilingData 中存储原始维度 `dimM/dimN/dimK`，GlobalBuffer 统一使用原始维度而非 fractal padded 值。

（c）Tiling 分块策略的调优

初始 Tiling 使用了基于 aivNum 的动态 baseN 选择（小 M 时 baseN=128），导致 FRACTAL_NZ 格式下后续 N 分块的权重地址出现非 32 对齐，产生错误结果。修改为 baseN 固定 256，N 方向对齐到 32，确保所有 NZ 分块正确读取。

**2. AI 辅助工具使用情况**

本次实践中使用了 AI 辅助工具（Claude Code）完成以下工作：
- 从 qmm_custom_project（注册形式自定义算子工程）中提取 Tiling 和 Kernel 实现逻辑，适配到直调形式的 notebook 框架中；
- 辅助分析 `__mix__` 下的 aicore exception 根因，定位到 Vector 核执行 Cube 指令的问题；
- 辅助完成 Tiling 参数命名重构和代码注释的中文化；
- 辅助完成本报告的编写。

AI 辅助在以下方面显著提升了效率：快速对照 project 工程和 notebook 模板的差异点；根据错误码和 dump 信息定位硬件异常根因；批量重构变量名。同时需要注意 AI 可能引入的问题：AI 初始生成的适配代码遗漏了 `__mix__` 下的核类型分发逻辑；`REGIST_MATMUL_OBJ` 的调用位置从 Init 移到 kernel entry 后仍需配合正确的 system workspace 指针。验证方式为逐项校验 AI 建议与参考实现的一致性，通过编译和运行测试确认正确性。

**3. 性能优化策略**

（a）Tiling 缓存：LLM 推理中 decode 阶段的形状高度重复（M=1），prefill 阶段也只有少数几个固定 M 值（50/4096）。引入静态 tiling 缓存后，除首次调用外后续同形状调用直接命中缓存，避免了重复的 `GetTiling()` 开销。

（b）Host 侧函数拆分：将 `qmm_custom` 按输出类型拆分为 `launchInt32` 和 `launchBf16` 两个内部函数，使得两类调用的 profiler tensor 信息构造各自独立，减少分支判断和 optional tensor 处理开销。

（c）Cube/Vector 跨核流水：Path 2 中 Cube 核完成 matmul 后立即通过 `CrossCoreSetFlag` 通知 Vector 核开始反量化，Cube 核随即可以处理下一轮调用，实现 Cube 和 Vector 的 overlap 执行。M 行均分给 2 个 AIV 核进一步提升了反量化阶段的并行度。


## 四、收获与感悟

**（此处由各队员分别填写，并注明姓名）**

本次启航营通过完整的"量化 matmul 算子开发 + 接入 Qwen3-8B 模型"实践，在以下方面获得了深刻的学习收获：

**柴伟东**：从算子原型定义、Tiling 设计、Cube/Vector Kernel 实现、编译、到 Torch 接口封装，完整体验了昇腾平台自定义算子开发的每个环节。特别是 `__mix__(1,2)` 混合编程模型下 Cube 与 Vector 的跨核协作与同步机制（`CrossCoreSetFlag`/`CrossCoreWaitFlag`），是单核 AI Core 编程无法学到的内容。

**董涵**：深入理解了 A8W8 量化方案中 INT8 矩阵乘和 Scale 反量化的数学原理，以及 perChannelScale（沿输出通道广播）和 perTokenScale（沿输入行广播）在 LLM 推理中的作用。了解了 FRACTAL_NZ 格式作为 Cube 单元高效输入的对齐要求和存储布局。

**宋彻**：学习了如何根据硬件资源（AIC/AIV 核数、UB 大小、L0A/L0B 缓存）和模型 shape 特征设计多核分块策略，以及 Tiling 参数如何影响多核利用率和单核计算效率。实践了通过 Tiling 缓存优化重复 shape 的推理场景。



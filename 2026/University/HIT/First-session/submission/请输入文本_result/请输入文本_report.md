# 团队实践报告

## 一、团队信息与贡献说明

### 1.1 团队基本信息

- 团队标识（组号）：请输入文本
- CANNJudge 提交账号：yuri-zwc
- CANNJudge 提交结果或链接：![ca3149e608ca3c08b21e38c5d93baa28.png](https://raw.gitcode.com/user-images/assets/10440592/87d7d546-8142-46e3-b805-6331cc90ad32/ca3149e608ca3c08b21e38c5d93baa28.png 'ca3149e608ca3c08b21e38c5d93baa28.png')

### 1.2 团队成员分工与贡献

| 姓名 | GitCode 账号 | 负责模块/任务分工 | 实际贡献说明 | 对应 commit hash |
| --- | --- | --- | --- | --- |
| 贾艺鹏 | Kirb114514 | Tiling 设计与 Kernel 实现、编译接入与测试 | 完成算子基础功能 | 2629b8b 、1c63efa 、dbf4d62 、5771525 |
| 张文程 | yuri-zwc | 算子性能优化 | 对 Kernel 实现进行性能分析与优化，降低推理耗时 | 23388a6 |

### 1.3 团队协作说明

本次实践围绕 QmmCustom 自定义量化 matmul 算子的完整开发流程展开，团队采用分工协作、迭代优化的协作模式。

任务拆分：根据实践大纲的六个任务，团队将工作划分为两个阶段：（1）贾艺鹏 负责 Tiling 数据结构设计、Kernel 双路径实现（INT32 输出 + BF16 反量化输出）、算子编译、Torch 接口接入以及全流程测试验证；（2）张文程 在算子功能跑通后负责性能分析与优化迭代。

代码汇总与评审：所有代码统一提交至团队仓库的同一分支 `请输入文本-Kirb114514`，通过 Git 分支合并进行代码集成。张文程 的优化代码通过 Merge Request 合入主分支，经功能回归测试验证精度无退化后完成合并。

集成与验证：完成编译后，统一运行单算子精度测试（12 组 shape）、单算子性能测试（Profiling 采集）以及模型接入后的端到端推理验证，确保各模块协同工作正常。

协作亮点：两位成员采用"先跑通、再优化"的策略——贾艺鹏 首先保证所有功能正确（12/12 精度 PASS），张文程 在此基础上进行性能调优，避免在优化过程中引入精度问题。最终模型 Decode 阶段平均推理时间从优化前 >100ms 降低到约 41.75 ms。

## 二、结果展示

### 2.1 单算子精度比对结果

为验证 QmmCustom 算子的计算正确性，采用官方提供的测试脚本对算子进行了功能验证。测试覆盖了 Qwen3-8B 推理过程中常见的 12 组矩阵规格（M ∈ {1, 50, 4096}，N ∈ {4096, 6144, 24576}，K ∈ {4096, 12288}），并分别对 INT32 输出模式（未提供 perTokenScale）和 BF16 输出模式（提供 perTokenScale）进行了精度测试。实验过程中，将自定义算子的输出结果与 PyTorch Reference 实现进行对比，采用 torch.allclose 判断计算结果是否满足精度要求。

经过 12 组数据测试，自定义算子能够正确完成 INT8 矩阵乘法计算，在两种输出模式下均能够获得与参考实现一致的计算结果，满足官方测试要求。

```
============================================================
测试汇总
============================================================
                      规格 |  INT32   |   BF16
----------------------------------------------
       M=1,K=4096,N=4096 |   PASS   |   PASS
       M=1,K=4096,N=6144 |   PASS   |   PASS
      M=1,K=4096,N=24576 |   PASS   |   PASS
      M=1,K=12288,N=4096 |   PASS   |   PASS
      M=50,K=4096,N=4096 |   PASS   |   PASS
      M=50,K=4096,N=6144 |   PASS   |   PASS
     M=50,K=4096,N=24576 |   PASS   |   PASS
     M=50,K=12288,N=4096 |   PASS   |   PASS
    M=4096,K=4096,N=4096 |   PASS   |   PASS
    M=4096,K=4096,N=6144 |   PASS   |   PASS
   M=4096,K=4096,N=24576 |   PASS   |   PASS
   M=4096,K=12288,N=4096 |   PASS   |   PASS
INT32: 12/12 通过, BF16: 12/12 通过
============================================================
```


### 2.2 单算子性能测试结果

完成精度验证后，对 QmmCustom 算子进行了性能测试。实验采用官方提供的 Profiling 测试脚本，在 Ascend 910B 平台上统计不同测试规格下的执行时间。

本次测试覆盖了 12 个不同规模的矩阵乘法规格，包括 Decode 场景（M=1）、混合场景（M=50）和 Prefill 场景（M=4096），能够较好地反映算子在实际模型推理中的性能表现。从测试结果可以看出，自定义算子能够稳定完成不同规模矩阵乘法计算，在各测试规格下均能够正常运行。

```
======================================================================
Profiler 数据汇总 (QmmCustom Duration)
======================================================================
              规格 (M,K,N) | INT32 Duration(us) |  BF16 Duration(us)
----------------------------------------------------------------------
       M=1,K=4096,N=4096 |            277.926 |            338.807
       M=1,K=4096,N=6144 |            413.188 |            507.010
      M=1,K=4096,N=24576 |           1789.356 |           2117.942
      M=1,K=12288,N=4096 |            811.976 |            897.238
      M=50,K=4096,N=4096 |            293.206 |            465.009
      M=50,K=4096,N=6144 |            433.469 |            698.014
     M=50,K=4096,N=24576 |           1908.158 |           2949.019
     M=50,K=12288,N=4096 |           1107.062 |           1268.785
    M=4096,K=4096,N=4096 |          13168.203 |          36804.476
    M=4096,K=4096,N=6144 |          20087.562 |          55349.266
   M=4096,K=4096,N=24576 |          83659.131 |         226065.477
   M=4096,K=12288,N=4096 |          41546.790 |          83198.882
======================================================================
```


### 2.3 算子接入模型性能测试结果

完成单算子测试后，将 QmmCustom 算子接入 Qwen3-8B 推理框架，并替换模型中的矩阵乘法算子，对模型整体推理流程进行了验证。

**模型推理结果（无 Profiler）：**

Qwen3-8B 成功加载模型并完成推理，自定义算子正确替代原有矩阵乘法算子参与计算，模型输出语义连贯。Prefill 阶段耗时约 196.53 ms，Decode 阶段平均推理时间 41.75 ms。

**模型推理结果（开启 Profiler）：**

为分析 QmmCustom 算子在模型中的实际耗时分布，开启 Profiler 进行端到端推理。Prefill 阶段耗时约 198.07 ms，Decode 阶段平均推理时间 41.23 ms。Profiler 采集到的 QmmCustom 各 Shape 耗时如下：

| Phase | Input Shapes | Output Dtype | Avg Duration(us) | Calls |
| --- | --- | --- | --- | --- |
| Prefill | 50,4096;4096,24576;24576 | INT32 | 2087.65 | 36 |
| Prefill | 50,12288;12288,4096;4096;50 | DT_BF16 | 1367.08 | 36 |
| Prefill | 50,4096;4096,6144;6144;50 | DT_BF16 | 755.17 | 36 |
| Prefill | 50,4096;4096,4096;4096;50 | DT_BF16 | 507.41 | 36 |
| Decode | 1,4096;4096,24576;24576 | INT32 | 1877.39 | 108 |
| Decode | 1,12288;12288,4096;4096;1 | DT_BF16 | 1023.81 | 108 |
| Decode | 1,4096;4096,6144;6144;1 | DT_BF16 | 557.71 | 108 |
| Decode | 1,4096;4096,4096;4096;1 | DT_BF16 | 374.97 | 108 |

模型推理输出示例：
> The output of an **attention function** is a **vector** that represents a weighted combination of the **value vectors**, where the weights are determined by the **similarity** between the **query vector** and the **key vectors** from the set of key-value pairs.

模型能够正常完成初始化和推理，自定义算子可以正确替代原有矩阵乘法算子参与计算，说明算子具有良好的兼容性和可用性。

## 三、方案说明

### 3.1 设计思路

**3.1.1 整体架构**

QmmCustom 算子采用 Host 侧 Tiling + Device 侧 Kernel 的两层架构：

- Host 侧：CalcQmmTiling 函数根据输入维度 (M, N, K) 和硬件资源计算分块参数，填充 QmmCustomTilingData 结构体，并通过 qmm_custom 函数启动 Kernel。
- Device 侧：qmm_custom_kernel 根据 isPertoken 标志分流至 QmmCubeBasicKernel（Path 1: INT32 输出）或 QmmPertokenKernel（Path 2: BF16 反量化输出）。

**3.1.2 TilingData 结构体设计**

QmmCustomTilingData 结构体采用 `#pragma pack(push, 8)` 保证 8 字节对齐，包含以下核心字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| cubeTilingData | TCubeTiling | CANN Matmul 库标准 Tiling 结构，含 baseM/baseN/baseK 等 |
| isPertoken | uint32_t | 标记当前路径：0 = INT32 输出，1 = BF16 反量化输出 |
| workspaceSize | uint32_t | Lib API 所需 workspace 大小 |
| M / N / K | uint32_t | 原始矩阵维度 |
| useNBlockSplit | uint32_t | 是否使用 N 维度分块（Decode 场景 M=1 时启用） |
| nTileCount | uint32_t | N 维度分块数 |
| usedCoreNum | uint32_t | 实际使用的 AI Core 数量 |

设计考量：

- TCubeTiling 复用 CANN Matmul 库的标准 Tiling 结构，通过 GetMatmulTiling API 自动计算，避免重复造轮子；
- isPertoken 作为 Kernel 内部分支选择标志，避免在 Host 侧维护两套独立的 Tiling 逻辑；
- useNBlockSplit 针对 Decode 场景（M=1）设计，由于 M 维度无法再拆分，改为按 N 维度切分多核并行。

**3.1.3 Tiling 函数实现**

CalcQmmTiling 函数通过 platform_ascendc::PlatformAscendCManager 获取硬件信息，核心逻辑：

1. 判断是否为 Decode 场景（M == 1），决定是否启用 NBlockSplit 多核并行策略；
2. 调用 GetMatmulTiling API 根据 (M, N, K) 自动计算 TCubeTiling，设置输入输出类型和分块参数；
3. Decode 场景下使用 SetFixSplit(QMM_BASE_M, QMM_BASE_N, -1)，将每个 N 维度 tile 分配到不同核上并行计算；
4. Prefill 场景且需要反量化时，使用 SetFixSplit(SelectBaseM(M), QMM_BASE_N, -1) 按 M 维度分块。

**3.1.4 Kernel 数据流与实现**

**Path 1: Cube-only（INT32 输出）—— QmmCubeBasicKernel**

数据流：GM (x1 INT8, x2 INT8) → L0A/L0B → Cube 单元 → INT32 结果 → GM (y INT32)

QmmCubeBasicKernel 仅使用 AI Core 的 Cube 计算单元，完成 INT8×INT8 矩阵乘法后直接将 INT32 结果写出。Decode 场景通过 ProcessNBlockSplit 实现多核 N 维度并行。

**Path 2: Cube+Vector（BF16 反量化输出）—— QmmPertokenKernel**

数据流：GM (x1 INT8, x2 INT8) → L0A/L0B → Cube 单元 → INT32 中间结果（VECOUT）→ Vector 单元：Cast INT32→FP32 → Mul perChannelScale → Mul perTokenScale → Cast FP32→BF16 → GM (y BF16)

QmmPertokenKernel 采用 Cube 计算 + Vector 反量化的协同流水线：

1. Cube 单元完成 INT8 matmul，将 INT32 结果输出到 VECOUT 队列；
2. Vector 单元接收 INT32 中间结果，依次执行类型转换、perChannelScale 乘法和 perTokenScale 乘法；
3. 最终 BF16 结果通过 DataCopyPad 写入输出 GM。

Decode 场景下，由于 M=1，perTokenScale 只需加载一次并驻留在 UB 中，Vector 核对每个 N 维度 tile 复用同一 scale 值完成反量化，有效减少了数据搬运开销。

### 3.2 问题解决与优化策略

**3.2.1 遇到的问题与解决方案**

**问题 1：编译环境配置与头文件引入**

编译自定义算子时，需要正确引入 CANN 的头文件和链接库。初始阶段遇到 kernel_operator.h 找不到、matmul_intf.h 路径不正确等问题。

解决方案：通过 ASCEND_TOOLKIT_HOME 环境变量定位 CANN 安装路径，在编译选项中正确设置 include 路径和链接库路径，使用 `torch.utils.cpp_extension.load` 完成算子编译。

**问题 2：TilingData 结构体对齐要求**

CANN Matmul 库要求 TilingData 结构体必须满足 8 字节对齐，否则会导致 Kernel 启动时读取 Tiling 参数异常。

解决方案：使用 `#pragma pack(push, 8)` 和 `alignas(8)` 保证结构体对齐，并将 TCubeTiling 放在结构体首字段位置。

**问题 3：Decode 场景多核并行策略选择**

初始实现中，所有场景均使用单核计算，Decode 阶段（M=1）耗时较高（平均 >100ms）。

解决方案：识别到 Decode 场景 M=1 无法按 M 维度拆分，改为按 N 维度拆分（useNBlockSplit），将 weight 矩阵的不同输出通道分配到不同 AI Core 上并行计算。单核负责一个或多个 N 维度 tile，通过 grid-stride loop 实现负载均衡。

**问题 4：perTokenScale 重复加载的性能开销**

在 QmmPertokenKernel 中，初始实现每次 Iterate 循环都重新从 GM 加载 perTokenScale，增加了数据搬运开销。

解决方案：在 Decode 场景（M=1）中，将 perTokenScale 一次加载到 UB 并驻留，后续所有 N 维度 tile 复用同一 scale 值。优化后 Decode 阶段 BF16 路径单算子耗时明显下降。

**3.2.2 AI 辅助使用情况**

使用方式：团队在以下环节使用了 AI 辅助工具：

1. TilingData 结构体设计：使用 AI 帮助梳理需要包含的字段及其数据类型，参考了 CANN Matmul 库的 TCubeTiling 文档。
2. Kernel 代码框架生成：基于 QmmCubeBasicKernel 和 QmmPertokenKernel 的类声明，AI 辅助生成了 Init 和 Process 方法的初步实现框架。
3. 编译错误排查：将编译错误信息输入 AI，快速定位到命名空间问题和 API 调用方式问题。
4. 测试脚本理解：AI 帮助解析了单算子测试和 Profiling 测试的代码逻辑。
5. Decode 多核并行策略设计：AI 辅助分析了 M=1 时的分块策略选择依据。

遇到的问题与验证：

- AI 生成的 Kernel 代码中，对 TCubeTiling 字段的访问方式与实际 API 存在差异，团队通过查阅 CANN 官方文档进行了修正；
- AI 建议的 NBlockSplit 策略经实际验证有效，Decode 阶段推理时间从 >100ms 降低到约 41.75 ms；
- 所有 AI 生成或建议的代码均经过人工 Review 和实际运行验证，确保精度测试全部 PASS 后才合并。

心得与体会：

- AI 工具在处理标准化任务（如结构体定义、API 调用模板）时效率较高，但在涉及硬件特定细节（如 L1/L0A/L0B 缓存大小、Cube 与 Vector 的协同调度）时仍需要开发者具备领域知识进行判断；
- 将编译错误信息直接输入 AI 进行诊断是一种高效的调试方式，但 AI 给出的修复建议需要结合实际代码上下文进行验证；
- AI 辅助可以加速"从 0 到 1"的原型搭建，但"从 1 到 N"的性能优化仍依赖开发者对硬件架构的深入理解。

**3.2.3 性能优化策略**

**优化 1：Decode 场景 N 维度多核并行（NBlockSplit）**

核心优化。识别到 Decode 场景 M=1 时无法按 M 维度并行，改为按 N 维度切分。通过 `useNBlockSplit` 标志，将 weight 矩阵按 QMM_BASE_N（256）为单位切分为多个 tile，分配至不同 AI Core 并行计算。优化前 Decode 平均推理时间 >100ms，优化后降至约 41.75 ms。

**优化 2：perTokenScale 驻留复用**

在 Decode 场景的 QmmPertokenKernel 中，将 perTokenScale 一次加载到 UB 并驻留，避免了每次 N 维度 tile 切换时重复加载。由于 M=1 时 perTokenScale 仅有一个元素，驻留开销极低，但节省了多次 GM→UB 的数据搬运。

**优化 3：Path 分支统一入口**

将两条路径统一到同一个 Kernel 入口 qmm_custom_kernel，通过 isPertoken 标志在运行时分流，避免了 Host 侧维护两套 Kernel 启动逻辑，减少了代码冗余。

优化效果：优化后，单算子精度测试 12/12 全部 PASS；模型端到端推理中，Decode 阶段平均推理时间从优化前 >100ms 降低到约 41.75 ms，QmmCustom 算子在 Prefill 和 Decode 阶段的耗时分布合理，验证了 Tiling 和 Kernel 实现的正确性与有效性。

## 四、收获与感悟

**团队成员 ：贾艺鹏**

这次启航营实践让我完整经历了一次"从零开发自定义算子并接入大模型"的全流程。在此之前，我对昇腾 NPU 的认知停留在"调用 torch_npu 的 API"，而这次从 TilingData 结构体设计到 Cube+Vector 双路径 Kernel 实现，再到编译、测试和模型接入，让我真正理解了昇腾 AI Core 的编程模型。

印象最深的是 Decode 场景的多核并行优化。一开始所有场景跑单核，Decode 平均推理时间超过 100ms，远不满足要求。仔细分析后发现 M=1 时传统按 M 维度分块不可行，于是借鉴了 NBlockSplit 的思路——按 N 维度切分，不同核计算不同的输出通道。这个改动让推理时间直接降到约 41ms，让我深刻体会到"理解硬件架构→设计正确并行策略"的重要性。

单算子精度测试 12/12 全部 PASS 的那一刻非常有成就感，但我也清楚地认识到这仅仅是开始。通过 Profiling 数据分析算子在模型各阶段的耗时分布，让我对性能调优有了更具体的认知——不仅仅让代码跑对，还要跑得高效。

**团队成员 ：张文程**

本次启航营实践让我对昇腾 AI 处理器的算子性能优化有了从理论到实践的完整认知。在贾艺鹏完成算子基本功能之后，我专注于性能优化工作。

通过分析单算子 Profiling 数据和模型推理日志，我识别到 Decode 阶段是性能瓶颈（初始 >100ms），主要问题在于 M=1 场景下的单核计算效率低下。在与贾艺鹏讨论后，我们确定了 N 维度多核并行的优化方向。优化过程中，我重点关注了分块大小、核数分配和负载均衡，最终将 Decode 平均推理时间降到约 41.75 ms。

这次实践让我深刻体会到：写好一个算子不仅仅是让它跑通，更重要的是理解硬件架构和计算瓶颈，在正确的方向上做优化。同时，团队协作中的"先跑通、再优化"策略也非常有效——在保证精度正确的前提下进行性能调优，避免了因优化引入精度问题后的排查困难。

此外，团队协作中使用 AI 工具辅助编码和调试的经历也让我受益匪浅。AI 可以帮我们快速生成代码框架和定位编译错误，但最终的方案决策和性能优化仍然需要人的判断力。这种"人机协作"的模式，或许是未来软件开发的新常态。



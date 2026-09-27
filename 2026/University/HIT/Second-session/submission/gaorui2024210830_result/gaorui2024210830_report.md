# 个人实践报告

## 一、个人信息与提交说明

### 1.1 个人基本信息

- 姓名：高睿
- 个人标识（学号或姓名拼音）：gaorui2024210830
- GitCode 账号：alpine_winter
- CANNJudge 提交账号：alpine_winter
- CANNJudge 提交结果或链接：https://cannjudge.cn/hitwh/cann/qmmcustom/submission/6a9e887cbf41025d601017ba

### 1.2 提交记录说明

> 说明：本次实践成果（算子代码、Tiling、测试、性能优化、Notebook 与报告）通过一次提交统一收录，commit hash 为 `8e5dd9d`（提交信息：`feat(HIT): 提交 gaorui2024210830 A8W8 量化 matmul 算子实践成果`）。

| 提交内容 | 简要说明 | 对应 commit hash |
| --- | --- | --- |
| 算子 Tiling 设计与实现 | 设计 `QmmCustomTilingData` 结构体与 `CalcQmmTiling` 函数，完成多核分块与 NZ 对齐 | `8e5dd9d` |
| Kernel 实现 | 实现 `QmmCubeBasicKernel`（Cube-only）与 `QmmPertokenKernel`（Cube+Vector 反量化）两条路径 | `8e5dd9d` |
| 单算子功能与性能测试 | 12 规格 × INT32/BF16 共 24 条用例全部 PASS，采集 Duration 性能数据 | `8e5dd9d` |
| 算子接入模型与验证 | 替换 `npu_quant_matmul` 为 `qmm_custom`，验证生成文本与 decode 平均耗时 | `8e5dd9d` |

> 有效贡献可以包括算子代码、Tiling、测试、性能优化、Notebook 或报告内容。纯空提交或仅修改格式不计为有效贡献。

### 1.3 个人开发过程说明

本次实践围绕「A8W8 量化 matmul 算子开发并接入 Qwen3-8B」这一主线展开，整体过程分为以下阶段：

1. **需求与规格分析**：从 Qwen3-8B 量化模型的 Profiling 结果中归纳出 `QmmCustom` 算子的输入输出规格——输入 x1（INT8, ND, [M,K]）、x2（INT8, FRACTAL_NZ, [K,N]）、scale（FLOAT32, [N]）及可选的 pertoken_scale（FLOAT32, [M]）；无 perTokenScale 时输出 INT32，有 perTokenScale 时输出 BF16。

2. **Tiling 设计**：根据算子原型与 Kernel 需求，自行设计 `QmmCustomTilingData` 结构体，并实现 `CalcQmmTiling` 函数。借助 `platform_ascendc::PlatformAscendCManager` 获取平台硬件信息，通过 `MultiCoreMatmulTiling` 自动计算 M/N/K 分块，并针对 NZ 格式做了 `singleCoreN` 的 32 对齐处理。

3. **Kernel 实现**：完成 `QmmCubeBasicKernel`（Cube-only，INT8×INT8→INT32）与 `QmmPertokenKernel`（Cube+Vector，INT8×INT8→INT32 再经 perChannelScale 与 perTokenScale 反量化输出 BF16）两条路径，入口函数用 `__mix__(1,2)` 声明实现 Cube 与 Vector 协同。

4. **编译**：在算子目录下用 cmake（`-DCMAKE_ASC_ARCHITECTURES=dav-2201`）+ make 编译，成功生成 `libascendc_ops.so`。

5. **单算子功能验证**：编写覆盖 12 种 (M,K,N) 规格的测试，分别验证 INT32（无 perTokenScale）与 BF16（有 perTokenScale）两种路径，24 条用例全部 PASS。

6. **单算子性能测试**：基于 `torch_npu.profiler` 采集各规格下 INT32 与 BF16 的 Duration 数据。

7. **模型接入验证**：将 `CompressedTensorsW8A8Int8LinearMethod` 中的 `torch_npu.npu_quant_matmul` 替换为 `torch.ops.ascendc_ops.qmm_custom`，运行量化推理，生成文本符合 attention 加权求和描述，decode 平均耗时 39.93 ms（低于 100ms 满分线）。

8. **模型 Profiling**：开启 Profiler 运行推理，统计 `QmmCustom` 在 Prefill/Decode 各 Shape 下的平均耗时。

在遇到 Tiling 分块对齐、Kernel 报错等问题时，通过逐步定位与辅助工具给出的建议（详见 3.2 节）逐一解决，最终完成精度、性能与模型接入的独立验证。

## 二、结果展示

### 2.1 单算子精度比对结果

功能测试覆盖 12 种 (M, K, N) 规格，每种规格分别验证 INT32 输出（无 perTokenScale）与 BF16 输出（有 perTokenScale）两种路径，共 24 条用例。INT32 路径采用精确匹配（rtol=0, atol=0），BF16 路径采用 rtol=0.01, atol=0.01 的容差比对。最终 **INT32 12/12 通过，BF16 12/12 通过，全部 PASS**。

| 规格 (M, K, N) | INT32 | BF16 |
| --- | --- | --- |
| M=1, K=4096, N=4096 | PASS | PASS |
| M=1, K=4096, N=6144 | PASS | PASS |
| M=1, K=4096, N=24576 | PASS | PASS |
| M=1, K=12288, N=4096 | PASS | PASS |
| M=50, K=4096, N=4096 | PASS | PASS |
| M=50, K=4096, N=6144 | PASS | PASS |
| M=50, K=4096, N=24576 | PASS | PASS |
| M=50, K=12288, N=4096 | PASS | PASS |
| M=4096, K=4096, N=4096 | PASS | PASS |
| M=4096, K=4096, N=6144 | PASS | PASS |
| M=4096, K=4096, N=24576 | PASS | PASS |
| M=4096, K=12288, N=4096 | PASS | PASS |

### 2.2 单算子性能测试结果

基于 `torch_npu.profiler` 采集各规格下 `QmmCustom` 的 Duration(us)，结果如下：

| 规格 (M, K, N) | INT32 Duration(us) | BF16 Duration(us) |
| --- | --- | --- |
| M=1, K=4096, N=4096 | 15.140 | 19.700 |
| M=1, K=4096, N=6144 | 23.100 | 28.460 |
| M=1, K=4096, N=24576 | 96.880 | 92.180 |
| M=1, K=12288, N=4096 | 31.820 | 36.100 |
| M=50, K=4096, N=4096 | 15.880 | 28.000 |
| M=50, K=4096, N=6144 | 46.120 | 57.680 |
| M=50, K=4096, N=24576 | 100.640 | 106.800 |
| M=50, K=12288, N=4096 | 43.820 | 48.480 |
| M=4096, K=4096, N=4096 | 382.660 | 472.080 |
| M=4096, K=4096, N=6144 | 575.180 | 724.880 |
| M=4096, K=4096, N=24576 | 2443.420 | 2991.280 |
| M=4096, K=12288, N=4096 | 1286.780 | 1364.120 |

### 2.3 算子接入模型性能测试结果

**生成文本**：将自定义算子接入 Qwen3-8B 量化模型后，生成的文本符合 attention 加权求和描述，与替换前的输出语义一致：

> The output of an **attention function** is a **vector** that is computed by **weighted summing** the **value vectors**, where the weights are determined by the **similarity** between the **query vector** and each **key vector** in the set of key-value pairs.
>
> ### Mathematically, the attention function can be described as:
>
> $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
>
> Where:
> - $ Q $: Query matrix (shape: $ [n \times d_k] $)
> - $ K $: Key matrix (shape: $ [m \times d_k] $)
> - $ V $: Value matrix (shape: $ [m \times d_v] $)
> - $ d_k $: Dimension of the key vectors (used for scaling)
> - $ \text{softmax} $: Applied along the attention heads or across the keys
>
> ### Output:
> - The **output** is a **vector** (or matrix, if multiple queries) of shape $ [n \times d_v] $, where each element is a **weighted combination** of the value vectors, based …

**decode 平均耗时**：`qwen3_8b decode average inference time cost is 39.93 ms`（低于 100ms 满分线）。

**模型 Profiling 结果**：开启 Profiler 后，`QmmCustom` 在 Prefill/Decode 各 Shape 下的平均耗时统计如下：

| Phase | Input Shapes | Output Dtype | Avg Duration(us) | Calls |
| --- | --- | --- | --- | --- |
| Prefill | "50,4096;4096,24576;24576" | INT32 | 98.853333 | 36 |
| Prefill | "50,12288;12288,4096;4096;50" | DT_BF16 | 65.052778 | 36 |
| Prefill | "50,4096;4096,6144;6144;50" | DT_BF16 | 53.283333 | 36 |
| Prefill | "50,4096;4096,4096;4096;50" | DT_BF16 | 34.381111 | 36 |
| Decode | "1,4096;4096,24576;24576" | INT32 | 80.812778 | 108 |
| Decode | "1,12288;12288,4096;4096;1" | DT_BF16 | 52.826852 | 108 |
| Decode | "1,4096;4096,6144;6144;1" | DT_BF16 | 34.490556 | 108 |
| Decode | "1,4096;4096,4096;4096;1" | DT_BF16 | 24.205000 | 108 |

## 三、方案说明

### 3.1 设计思路

本次采用 `<<<>>>` Kernel 直调方式开发 A8W8 量化 matmul 算子 `QmmCustom`，Host 侧通过 `kernel<<<numBlocks, dynUBufSize, stream>>>(args)` 直接启动 Kernel。

**Tiling 设计**

自定义 `QmmCustomTilingData` 结构体，包含：

- `TCubeTiling cubeTilingData`：CANN Matmul 库的标准 Tiling 结构，包含 M/N/K 的分块大小（singleCoreM/singleCoreN 等）、使用的核数（usedCoreNum）等字段；
- `uint32_t isPertoken`：标记是否走反量化路径（有 perTokenScale 为 1，否则为 0），用于 Kernel 内部分支选择；
- `uint32_t workspaceSize`：Cube 写入 INT32 中间结果所需的 GM workspace 大小；
- `uint32_t systemWorkspaceSize`：CANN 库 API 自身所需的 workspace 大小。

`CalcQmmTiling` 函数的实现要点：

1. 通过 `platform_ascendc::PlatformAscendCManager` 获取平台硬件信息；
2. 用 `matmul_tiling::MultiCoreMatmulTiling` 设置 A/B/C 的存储位置（GM）、格式（A/C 为 ND，B 为 NZ）与数据类型（A/B 为 INT8，C 为 INT32），再设置原始 shape；
3. 调用 `GetTiling` 得到多核分块结果，若 `singleCoreN` 不是 32 的整数倍，则按 32 对齐后重新计算（NZ 格式的 N 维度对齐约束）；
4. 根据 `isPertoken` 计算 workspace 大小：无 perTokenScale 时仅需 `systemWorkspaceSize`，有 perTokenScale 时需额外分配 `M*N*sizeof(int32_t)` 用于存放 Cube 的 INT32 中间结果。

**Kernel 实现**

算子包含两条执行路径，由 `isPertoken` 决定：

- **Path 1（Cube-only，`QmmCubeBasicKernel`）**：仅使用 AI Core 的 Cube 计算单元，完成 INT8×INT8 矩阵乘法后直接将 INT32 结果写入输出 GM，无 Vector 参与。数据流为 `x1(INT8, ND) @ x2(INT8, NZ) → y(INT32, ND)`。

- **Path 2（Cube+Vector，`QmmPertokenKernel`）**：Cube 单元先完成 INT8 matmul 得到 INT32 中间结果并写入 GM workspace；随后 Vector 单元从 GM 读取 INT32 结果，依次完成 `Cast`(INT32→FLOAT32)、乘以 `perChannelScale`、乘以 `perTokenScale`、`Cast`(FLOAT32→BF16) 等反量化操作，最终将 BF16 结果写入输出 GM。数据流为 `(x1 @ x2) * perChannelScale * perTokenScale → y(BF16, ND)`。

Kernel 入口 `qmm_custom_kernel` 用 `__mix__(1,2)` 属性声明，表示每个 Block 分配 1 个 Cube 核和 2 个 Vector 核，使 Cube 与 Vector 在同一 Kernel 内协同工作（Path 2 中 Cube 计算完成后通过 `CrossCoreSetFlag`/`CrossCoreWaitFlag` 同步）。两个 Kernel 复用相同的 Tiling 结构与 Kernel 启动入口，`CalcOffset` 依据 `blockIdx` 计算 M/N 维度的分块偏移，并对尾部块做 `SetTail` 处理。

### 3.2 问题解决与优化策略

**1. 实践过程中遇到的问题与解决**

- **Tiling 分块对齐问题**：NZ 格式对 N 维度有 32 对齐的约束，直接使用默认分块可能导致 `singleCoreN` 不是 32 的整数倍。通过在 `CalcQmmTiling` 中判断 `singleCoreN % 32 != 0` 时向上取整对齐到 32，并重新调用 `GetTiling`，解决了非对齐规格（如 N=6144 等）下的正确性问题。

- **Kernel 报错排查**：在实现 Cube+Vector 协同路径时，遇到 Cube 结果未就绪 Vector 就开始读取的问题，通过 `CrossCoreSetFlag`/`CrossCoreWaitFlag` 在 Cube 与 Vector 之间建立同步，保证反量化在 matmul 完成之后执行。

- **Vector 反量化链路验证**：反量化需依次完成 INT32→FLOAT32 Cast、逐通道乘法、逐 token 标量乘、FLOAT32→BF16 Cast，通过单算子 BF16 路径的 12 条用例比对，确认与参考结果在容差范围内一致。

**2. AI 辅助使用情况**

- 使用过的 AI 工具：CANNBot、ChatGPT、Claude。
- 解决的具体问题：Tiling 设计（多核分块与 NZ 对齐策略）、Kernel 报错排查（Cube 与 Vector 的同步、反量化实现细节）。
- 是否引入 bug：未引入 bug。
- 如何验证：多次验证，并针对 CANNJudge 要求内容的性能做了优化提升——AI 给出的 Tiling/Kernel 建议通过跑通 24 条单算子用例（INT32 12/12、BF16 12/12 全 PASS）、CANNJudge 提交等实测验证。在使用 AI 建议前先理解其原理，再结合代码审查与本地实测确认其正确性与性能收益，避免盲目照搬。

**3. 性能优化策略与效果**

- 使用 NZ 格式存放权重矩阵 x2，充分利用 Cube 单元的高效存储格式提升 matmul 计算性能。
- 采用 `MultiCoreMatmulTiling` 将 M/N 维度分配到多个核上并行计算，并对 `singleCoreN` 做 32 对齐，平衡缓存利用率与负载。
- Cube-only 路径直接输出 INT32，避免不必要的反量化开销；反量化路径仅在需要 BF16 输出时启用。
- 最终模型接入后 decode 平均耗时为 39.93 ms，低于 100ms 满分线。

## 四、收获与感悟

本次启航营的核心实践让我把此前分散学习的知识串联成了一次完整的自定义算子开发实战。从 A8W8 量化原理出发，通过需求分析归纳算子规格，再到 Tiling 设计、Kernel 实现、算子编译、单算子功能与性能测试，最终将自定义算子接入 Qwen3-8B 量化模型并完成验证，完整走通了「需求分析——算子开发——单算子测试——测试驱动优化——算子接入模型测试」的开发链路。

在实现过程中，我加深了对昇腾 AI Core 的 Cube 与 Vector 协作编程模型的理解：Cube-only 路径与 Cube+Vector 反量化路径的差异、`__mix__(1,2)` 的多核协同声明、以及 Cube 与 Vector 之间的跨核同步机制，这些都是在纸上难以真正体会的细节。Tiling 设计的 NZ 对齐、workspace 规划等看似琐碎的约束，恰恰是保证算子正确性与性能的关键。

此外，合理借助 AI 工具辅助 Tiling 设计与 Kernel 排错，再通过独立思考与实测验证，让我体会到「人机协作」的价值边界：AI 能快速给出思路与建议，但正确性判断、性能优化方向与最终验收仍需自己把关。这份经历不仅提升了我对 Ascend C 算子开发的理解，也锻炼了定位问题、独立验证与工程化交付的能力。

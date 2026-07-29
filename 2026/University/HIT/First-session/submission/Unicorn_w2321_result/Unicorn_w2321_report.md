# 团队实践报告

## 一、团队信息与贡献说明

### 1.1 团队基本信息

- 团队标识（组号）：Unicorn_w2321
- CANNJudge 提交账号：尚煊
- CANNJudge 提交结果或链接：https://cannjudge.cn/hit/20260721/qmmcustom/submission/6a61ba3b1336c465ba83f86a

### 1.2 团队成员分工与贡献

| 姓名 | GitCode 账号 | 负责模块/任务分工 | 实际贡献说明 | 对应 commit hash |
| --- | --- | --- | --- | --- |
| 尚煊 | @sx61699 | Kernel 实现、Host 接口与测试调优 | 完成 Cube-only 和 Cube+Vector 两条 Kernel 路径的代码编写，实现 Host 侧 Torch 接口，负责 Stream 同步方案设计与 CrossCore Flag 问题的调试修复，独立完成 12 组规格的精度验证与性能 Profiling 测试及数据解析，算子接入 Qwen3-8B 推理管线并验证输出一致性 | `477d0f9e9d9ec722e067d2c80a2c481bef5fb90d` |
| 罗星宇 | @2321Robin | Tiling 设计与报告撰写 | 完成 QmmCustomTilingData 结构体设计与 CalcQmmTiling 函数实现，参与 Dequant 向量化分块策略的设计与验证，撰写团队实践报告并整理测试数据与性能分析 | `98f75837867c5d84dea43c1bbdb2375d297849c7` |
|  |   |  |  |  |

> 有效贡献可以包括算子代码、Tiling、测试、性能优化、Notebook 或报告内容。纯合并、空提交或仅修改格式不计为有效贡献。禁止多人共用同一个 GitCode 账号提交。

### 1.3 团队协作说明

两人分工明确、协同推进。罗星宇负责 Tiling 层面的设计工作，包括 TilingData 结构体定义、CalcQmmTiling 函数实现以及 Dequant 分块策略的设计与验证；尚煊承担了主要的开发与调试工作，包括 Cube 和 Vector 两条 Kernel 路径的代码编写、Host 侧 Torch 接口与 Profiler 兼容层的实现、AIC/AIV 同步方案的设计与调试、全部精度与性能测试以及 Profiling 数据解析，最终完成算子接入 Qwen3-8B 推理管线并验证输出一致性。开发过程中两人保持密切沟通，罗星宇完成 Tiling 后交由尚煊进行 Kernel 集成与测试，测试中发现的问题再反馈回 Tiling 层面进行调整优化。最终由尚煊统一提交 Notebook 和算子文件，罗星宇完成报告撰写。

## 二、结果展示

### 2.1 单算子精度比对结果

全部 12 组规格测试通过，INT32 和 BF16 两条路径均全部 PASS：

| 规格 (M,K,N) | INT32 | BF16 |
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

INT32: 12/12 通过, BF16: 12/12 通过

### 2.2 单算子性能测试结果

| 规格 (M,K,N) | INT32 Duration(us) | BF16 Duration(us) |
| --- | --- | --- |
| M=1, K=4096, N=4096 | 147.240 | 146.140 |
| M=1, K=4096, N=6144 | 20.620 | 217.580 |
| M=1, K=4096, N=24576 | 216.960 | 29.040 |
| M=1, K=12288, N=4096 | 984.720 | 988.160 |
| M=50, K=4096, N=4096 | 114.240 | 554.860 |
| M=50, K=4096, N=6144 | 554.060 | 20.860 |
| M=50, K=4096, N=24576 | 6537.340 | 6566.360 |
| M=50, K=12288, N=4096 | 1597.680 | 9987.680 |
| M=4096, K=4096, N=4096 | 9977.200 | 2507.200 |
| M=4096, K=4096, N=6144 | 41378.040 | 40995.020 |
| M=4096, K=4096, N=24576 | 11863.620 | 20396.180 |
| M=4096, K=12288, N=4096 | 20285.720 | 1766.480 |

### 2.3 算子接入模型性能测试结果

- 模型：Qwen3-8B (W8A8 量化)
- Prefill 耗时：约 93.43 ms
- Decode 平均耗时：约 93.66 ms
- 推理生成文本验证通过，输出内容与原始算子一致

## 三、方案说明

### 3.1 设计思路

**TilingData 设计：**
- `QmmCustomTilingData` 结构体包含 `TCubeTiling`（标准 Cube Tiling 结构）、`isPertoken`（反量化路径标志）、输入输出维度 `M/N/K`、单核分块大小 `singleCoreM/singleCoreN`、`vectorCoreNum`（AIV 核数，为 Cube 核数的 2 倍）以及 `workspaceSize`（system workspace + user workspace 总和）。
- Tiling 函数 `CalcQmmTiling` 使用 `MatmulApiTiling` API 自动计算 Cube 分块参数，获取硬件平台信息，并根据 `isPertoken` 决定是否分配用户 workspace 空间。

**Kernel 实现：**
- **Path 1（Cube-only，INT32 输出）**：`QmmCubeBasicKernel` 仅由 AIC 执行，使用 `Matmul` 高阶 API 完成 INT8×INT8 矩阵乘法，结果直接写入输出 GM。
- **Path 2（Cube+Vector，BF16 输出）**：分为两个 Kernel 串行执行：
  - `QmmCubeWorkspaceKernel`（AIC 执行）：完成 INT8 matmul，将 INT32 中间结果写入用户 workspace。
  - `QmmDequantKernel`（AIV 执行）：按 2048 列固定块读取 INT32 结果，依次执行 Cast（INT32→FLOAT32）、乘以 perChannelScale、乘以 perTokenScale、Cast（FLOAT32→BF16）反量化操作。
- Kernel 使用 `__mix__(1, 2)` 属性，每个逻辑 Block 分配 1 个 AIC 和 2 个 AIV。两个 Kernel 在同一 Stream 上按提交顺序串行执行，避免 CrossCore Flag 同步问题。

### 3.2 问题解决与优化策略

1. **Stream 串行同步替代 CrossCore Flag**：BF16 路径最初尝试使用 `CrossCoreSetFlag/CrossCoreWaitFlag` 进行 AIC 与 AIV 同步，但与 Matmul 高阶 API 混用时在大规格下出现 flag 冲突和 AIV 提前读取问题。最终改为两个独立 Kernel 在同一 Stream 上串行提交，利用 Stream 天然的顺序执行特性，彻底规避了同步问题。

2. **AIV workspace 地址问题**：`GetUserWorkspace` API 不支持 Vector Core，因此在 Host 端显式计算用户 workspace 起始地址，作为参数直接传入 AIV Kernel，避免地址异常。

3. **Dequant 向量化分块策略**：采用按二维输出 tile 线性编号做跨核步进分配的策略，使 N=24576 时的 12 个列块均匀落到全部 AIV 上，同时避免单核内嵌套长循环。固定 2048 列块大小确保 N 维度可整除，不存在尾块非 32B 对齐问题。

4. **AI 辅助**：在开发过程中使用了 AI 辅助理解 Ascend C 编程模型（Cube/Vector 协作、TQue/TBuf 缓冲区管理）以及调试 Profiler 兼容性代码（运行时 dlsym 动态符号解析以兼容 CANN 8.5.2 不同补丁版本）。

## 四、收获与感悟

**尚煊：**

本次实践中我承担了主要的开发与调试工作。Kernel 实现方面，完成了 Cube-only 和 Cube+Vector 两条路径的代码编写，其中 CrossCore Flag 同步冲突的调试过程让我印象最深——大规格下 AIV 会提前读取未完成的中间结果，排查定位后改为同一 Stream 串行提交两个独立 Kernel 彻底解决。Host 接口方面，实现了 Profiler 兼容层以适配 CANN 8.5.2 不同补丁版本，通过运行时 dlsym 动态解析扩展接口符号，保证了 profiling 数据的正确采集。测试调优方面，独立完成 12 组规格的精度验证与 Profiling 性能测试，并编写了 kernel_details.csv 解析脚本自动回填 Shape/Dtype 元数据。最后完成了算子接入 Qwen3-8B 推理管线的工作，替换 npu_quant_matmul 并验证推理输出与原始算子一致。通过本次实践，我对昇腾 AI Core 编程模型从 Kernel 到 Host 的完整链路有了深入理解。

**罗星宇：**

本次实践中我主要负责 Tiling 设计与报告撰写。TilingData 结构体需要综合考虑 M/N/K 维度、Cube 核数、workspace 分配等多个因素，设计过程中反复与尚煊沟通 Kernel 端的实际需求来调整字段。Dequant 的向量化分块策略也是在测试反馈中逐步优化的——从最初的按行分配改为按二维 tile 线性编号跨核步进分配，解决了 N=24576 时负载不均的问题。撰写报告时系统梳理了整个开发流程，对 A8W8 量化原理、Ascend C 编程接口和算子全链路开发有了完整认知。这次经历让我体会到算子开发是 Tiling、Kernel、测试三者不断迭代配合的过程。

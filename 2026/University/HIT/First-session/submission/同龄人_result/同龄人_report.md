# 团队实践报告

## 一、团队信息与贡献说明

### 1.1 团队基本信息

- 团队标识（组号）：同龄人（5号）
- CANNJudge 提交账号：puavyy
- CANNJudge 提交结果或链接：https://cannjudge.cn/hit/20260721/qmmcustom/ranking

### 1.2 团队成员分工与贡献

| 姓名 | GitCode 账号 | 负责模块/任务分工 | 实际贡献说明 | 对应 commit hash |
| --- | --- | --- | --- | --- |
| 董国帅 | fysy_ | Tiling 设计与实现 | 完成 `QmmCustomTilingData` 结构体设计，包含 `TCubeTiling`、多核分块参数（`singleCoreM`/`singleCoreN`/`mBlockNum`/`nBlockNum`）等字段；实现 `CalcQmmTiling` 函数，针对 M≤64 小 M 场景和大 M 场景分别设计 N 维切分和二维分块策略；处理 M/K/N 对齐约束、核数选择及 workspace 规划，补充边界测试 | `8b32fa8` |
| 宫浩然 | theng | Cube-only Kernel 实现 | 实现 `QmmCubeBasicKernel` 类，完成 INT8×INT8→INT32 的 Cube 路径；处理多核 M 维行划分、尾块裁剪、GM Tensor 映射和 Matmul 高阶 API 调用，将 INT32 结果直接写入输出 GM；参与 INT32 路径精度验证 | `0ef6d9b` |
| 黄鑫 | Hash_hx | Cube+Vector Kernel 实现 | 实现 `QmmPertokenKernel` 类，完成 INT32→FP32→BF16 的 Vector 反量化路径；处理 per-channel scale 与 per-token scale 的双重缩放、跨核同步（`CrossCoreSetFlag`/`CrossCoreWaitFlag`）、尾块及数据搬运；参与 BF16 路径精度验证 | `c6be2e8` |
| 丁懿 | schordingerc | Torch 接口、模型接入、测试与报告撰写 | 完成 Torch 自定义算子注册与参数校验，将 `npu_quant_matmul` 替换为 `qmm_custom`；修复子进程动态库加载，完成单算子功能/性能测试、Qwen3-8B 推理与 Profiling 汇总；负责团队实践报告的整理与撰写 | `8e619e15` |

### 1.3 团队协作说明

团队按照“算子 Tiling—Kernel 实现—框架接入与验证”的链路拆分任务。董国帅负责 Tiling 数据结构、分块策略和 workspace 规划，为两条 Kernel 路径提供统一接口；宫浩然负责 Cube-only 路径的 INT8 矩阵乘实现；黄鑫负责 Cube+Vector 混合核路径的反量化实现；丁懿负责 Torch 接口、Notebook 6.2 模型替换、测试汇总和报告撰写。

各模块集成后，团队共同在 CANNLab NPU 环境中从头运行 Notebook，依次验证编译、12 组双输出类型精度、单算子 Profiling、Qwen3-8B 文本输出以及网络 Profiling。协作过程中，各成员使用本人 GitCode 账号 Fork 官方仓库并完成实际修改，队长通过 `git fetch` 获取成员分支，使用 `git diff` 审查修改内容后依次合并，确保代码、Notebook 输出、报告描述和 commit 记录一致。

## 二、结果展示

### 2.1 单算子精度比对结果

测试覆盖 12 组 `(M, K, N)` 规格，每组分别验证：

- 不传入 `perTokenScale`：INT8×INT8，输出 INT32，与 CPU 参考结果精确比较（`rtol=0, atol=0`）；
- 传入 `perTokenScale`：INT32 中间结果乘以 per-channel 与 per-token scale 后输出 BF16，以 `rtol=0.01, atol=0.01` 比较。

最终结果：

| M | K | N | INT32 结果 | BF16 结果 |
| ---: | ---: | ---: | --- | --- |
| 1 | 4096 | 4096 | PASS | PASS |
| 1 | 4096 | 6144 | PASS | PASS |
| 1 | 4096 | 24576 | PASS | PASS |
| 1 | 12288 | 4096 | PASS | PASS |
| 50 | 4096 | 4096 | PASS | PASS |
| 50 | 4096 | 6144 | PASS | PASS |
| 50 | 4096 | 24576 | PASS | PASS |
| 50 | 12288 | 4096 | PASS | PASS |
| 4096 | 4096 | 4096 | PASS | PASS |
| 4096 | 4096 | 6144 | PASS | PASS |
| 4096 | 4096 | 24576 | PASS | PASS |
| 4096 | 12288 | 4096 | PASS | PASS |

Notebook 中共完成 24 项精度验证，汇总输出为：`INT32: 12/12 通过, BF16: 12/12 通过`。

### 2.2 单算子性能测试结果

下表为 Profiling 采集的 `QmmCustom Duration`，单位为微秒（μs）：

| M | K | N | INT32 Duration(μs) | BF16 Duration(μs) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4096 | 4096 | 141.257 | 141.377 |
| 1 | 4096 | 6144 | 208.936 | 208.636 |
| 1 | 4096 | 24576 | 903.762 | 899.702 |
| 1 | 12288 | 4096 | 411.912 | 411.832 |
| 50 | 4096 | 4096 | 150.297 | 149.957 |
| 50 | 4096 | 6144 | 217.176 | 217.456 |
| 50 | 4096 | 24576 | 921.662 | 917.902 |
| 50 | 12288 | 4096 | 421.532 | 421.232 |
| 4096 | 4096 | 4096 | 413.912 | 414.512 |
| 4096 | 4096 | 6144 | 642.647 | 644.547 |
| 4096 | 4096 | 24576 | 3013.380 | 2994.460 |
| 4096 | 12288 | 4096 | 1443.891 | 1418.651 |

Profiler 共找到 24 条 QmmCustom 原始记录，覆盖上述 12 组规格的 INT32 与 BF16 两条路径。

### 2.3 算子接入模型性能测试结果

自定义算子被注册为 `torch.ops.ascendc_ops.qmm_custom`，并替换 Qwen3-8B W8A8 推理代码中的 `torch_npu.npu_quant_matmul`。模型能够针对 Attention 提示词输出符合逻辑的向量加权与缩放点积公式说明，网络功能正确。

最终结果：

| 项目 | 结果 |
| --- | --- |
| 推理 Decode 平均耗时 | 92.98ms |
| 课程网络性能满分阈值 | 低于 100 ms |
| 是否达到阈值 | 是 |

网络输入为：

```text
An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is
```

自定义算子接入后的输出以如下 attention 描述开头：

```text
The output of an attention function is a **weighted sum of the value vectors**, where the weights are determined by the similarity between the **query vector** and each **key vector** in the set of key-value pairs.
```

该结果符合课程要求的 attention 逻辑描述。

模型 Profiling 中的 QmmCustom 统计如下：

| Phase | Input Shapes | Output Dtype | Avg Duration(μs) | Calls |
| --- | --- | --- | ---: | ---: |
| Prefill | "50,4096;4096,24576;24576" | INT32 | 1152.458083 | 36 |
| Prefill | "50,12288;12288,4096;4096;50" | DT_BF16 | 591.008194 | 36 |
| Prefill | "50,4096;4096,6144;6144;50" | DT_BF16 | 294.604611 | 36 |
| Prefill | "50,4096;4096,4096;4096;50" | DT_BF16 | 198.028306 | 36 |
| Decode | "1,4096;4096,24576;24576" | INT32 | 1053.271204 | 108 |
| Decode | "1,12288;12288,4096;4096;1" | DT_BF16 | 545.870019 | 107 |
| Decode | "1,4096;4096,6144;6144;1" | DT_BF16 | 267.324130 | 108 |
| Decode | "1,4096;4096,4096;4096;1" | DT_BF16 | 178.116093 | 108 |


## 三、方案说明

### 3.1 设计思路

#### 3.1.1 TilingData 与分块策略

`QmmCustomTilingData` 保存 CANN Matmul 库返回的 `TCubeTiling`，以及 `isPertoken`、M/N/K、单核 M/N、二维网格块数、Vector tile 大小和 workspace 信息。Tiling 阶段通过 `PlatformAscendCManager` 获取 AIC 数量，再由分块策略选择逻辑核网格：

- Decode 或小 M（M≤64）场景只切 N，避免每核 M 过小；
- Prefill 或大 M 场景同时切分 M/N，优先使用更多 AIC；
- M 块按 16 对齐，INT8 FRACTAL_NZ 的 N/K 按 32 对齐；
- 同核数候选中倾向接近 `singleN ≈ 2 × singleM` 的分块；
- 调用 `GetTiling` 后使用其实际返回的 `singleCoreM/N` 重新计算块数和地址，避免平台为满足 L1/L0 约束调整分块后产生 GM 偏移错误。

对于 BF16 反量化路径，在系统 workspace 后按 512 字节对齐分配 `M×N×sizeof(int32_t)` 的用户 workspace；INT32 路径无需该中间结果区。

#### 3.1.2 Cube-only Kernel

`QmmCubeBasicKernel` 仅在 AIC 上执行。每个逻辑核根据二维网格确定 M/N 起点和尾块大小，A 输入采用 ND，B 权重采用 INT8 FRACTAL_NZ，C 输出采用 ND INT32。Kernel 设置全局原始 shape 与实际尾块后调用 Matmul 接口，将结果直接写入输出 GM。

#### 3.1.3 Cube+Vector Kernel

`QmmPertokenKernel` 使用 `__mix__(1, 2)`，每个逻辑 AIC 配置两个 AIV：

1. AIC 完成 INT8 Matmul，并把 INT32 中间结果写入用户 workspace；
2. 通过 `CrossCoreSetFlag`/`CrossCoreWaitFlag` 保证 GM 搬出完成后再启动同组 AIV；
3. 两个 AIV 分摊当前 M 块的行，以 1024/2048 个元素为 Vector tile 读取中间结果和 per-channel scale；
4. 依次执行 INT32→FP32 Cast、乘 per-channel scale、乘 per-token scale，再以 RINT 模式 Cast 为 BF16 并写回 GM。

### 3.2 问题解决与优化策略

#### 3.2.1 实践中遇到的问题及解决方法

**问题一：量化模型缺失**

组员环境中最初没有 Qwen3-8B-W8A8 权重。团队先完成第五章 AMCT W8A8 量化导出，确认 `config.json` 和权重目录存在后再运行第六章。

**问题二：自定义算子在推理子进程中未注册**

第一次仅替换 Python 调用，推理时报出 `ascendc_ops` 没有 `qmm_custom` 属性。原因是 Notebook 内核加载了动态库，但 `infer.sh` 创建的新进程没有加载。修复方式是在目标量化模块导入阶段写入 `torch.ops.load_library`，随后重新推理并确认返回码为 0。

**问题三：小 M 场景 Tiling 负载不均**

最初若固定较大的 baseM，M=1 时 `GetTiling` 可能失败或负载不均。团队改为让 Tiling API 根据实际 shape 自动选择基础分块，并使用 `min(AIC核数, M)` 限制有效核数，同时对小 M 场景采用 N 方向切分，有效降低每核空闲。

**问题四：混合核编程中的 AIC/AIV 指令隔离**

`__mix__(1, 2)` 混合核模式下，Cube 核和 Vector 核都会执行同一份 kernel 代码。Matmul 高阶 API 只在 Cube(AIC) 核上合法，若不加以区分，Vector(AIV) 核执行这些调用会导致非法地址访问。修复方式是在 Cube-only 路径中让 AIV 直接跳过；在 pertoken 路径中，`mm_.Init()` 仅在 `ASCEND_IS_AIC` 中调用，Vector 缓冲区分配仅在 `ASCEND_IS_AIV` 中调用。

**问题五：BF16 反量化精度误差**

原实现为 `(acc × per-channel) × per-token`，与内置算子的 `acc × (per-channel × per-token)` 在数学上等价但在 BF16 舍入上存在 1 ULP 差异。虽然单算子容差测试通过，但差异在网络中经多层传播后足以改变最大 logit。修复方式是将 Vector 路径改为先计算 `per-channel × per-token`，再乘 INT32 累加值，修复后四组算子输出与内置实现逐位一致，完整网络文本也恢复一致。

#### 3.2.2 AI 辅助使用说明

团队使用 AI 辅助分析环境路径、Notebook 执行日志、动态库注册位置和 Profiling CSV 字段，并协助整理测试结果及报告结构。AI 的第一次模型替换建议只关注了函数调用，没有充分考虑 `infer.sh` 会创建独立 Python 子进程，因此出现了动态库未注册的新问题。AI 辅助的早期实现还采用了数学等价但数值舍入不等价的反量化乘法顺序，单算子容差测试未暴露问题，却改变了网络输出。

团队没有直接把建议当作正确结果，而是依据 `AttributeError` 追踪进程边界，补充子进程加载语句；针对反量化顺序，通过确定性网络对比和逐元素诊断将其识别为新 bug，并在修复后重新完成单算子、网络和 Profiler 验证。此次经历说明，AI 适合加速定位和生成候选方案，但硬件算子、依赖版本和多进程行为仍必须通过真实 NPU 运行结果确认。

#### 3.2.3 性能优化策略及效果

| 优化策略 | 实现方式 | 效果 |
| --- | --- | --- |
| 小 M 场景 N 方向切分 | M≤64 时仅切分 N 维度，避免每核 M 过小 | 降低 Decode 场景延迟 |
| 大 M 场景二维分块 | 同时切分 M/N 维度，优先使用更多 AIC | 提升 Prefill 并行度 |
| 分级基础块设置 | 根据规模选择 128×256 等基础分块 | 平衡 L1/L0 缓存利用率 |
| 实际 Tiling 返回值计算偏移 | 使用平台返回的 `singleCoreM/N` 重新计算块数和地址 | 避免 GM 偏移错误 |
| FRACTAL_NZ 对齐约束 | M 按 16 对齐，N/K 按 32 对齐 | 满足 Cube 计算单元要求 |
| Vector 分块反量化 | 两个 AIV 分摊行，以 1024/2048 元素分块 | 提升 Vector 流水效率 |
| scale 跨行复用 | 每 VEC_LEN 列块仅加载一次列 scale | 减少重复 GM 读取 |
| 按需 Workspace | 仅 BF16 路径申请 INT32 中间结果区 | 降低 INT32 路径内存开销 |
| 双 Scale 融合 | 先计算 per-channel × per-token，再乘累加值 | 减少精度敏感的运算次序差异 |

通过上述策略，模型 Decode 平均推理耗时稳定低于 100 ms，达到课程满分阈值。

## 四、收获与感悟

### 董国帅

通过负责 Tiling 与多核分块，我认识到算子正确性不仅取决于数学公式，还取决于对齐、存储格式、核数和 workspace 等硬件约束。将平台返回的实际 Tiling 参数用于地址计算，也让我理解了工程实现中"以真实运行约束为准"的重要性。针对 Qwen3-8B 中不同 shape 特点设计差异化的分块策略，让我深刻体会到"算法 + 架构"协同优化的价值。

### 宫浩然

通过实现 Cube-only Kernel，我掌握了 GM Tensor 映射、AscendC Matmul 对象配置、多核行划分和尾块处理方法。此次实践让我更直观地理解了 INT8 矩阵乘如何利用 Cube 单元完成 INT32 累加和结果写回。同时我也认识到，能够单独通过测试的模块不代表集成后一定正确，接口统一和精度回归同样重要。

### 黄鑫

Cube+Vector 路径让我更直观地理解了 AIC 与 AIV 的分工，以及跨核同步和数据搬运对结果正确性与性能的共同影响。面对尾块和反量化误差时，分层检查中间结果比只看最终输出更有效。通过实现 perChannelScale、perTokenScale 广播和 BF16 舍入，我也认识到单算子误差很小并不意味着可以忽略，仍需通过模型输出进行验证。

### 丁懿

从 Torch 接口到 Qwen3-8B 模型替换和 Profiling，我体会到算子接入是一个跨编译、框架、子进程和模型的完整链路。此次排查动态库注册与依赖版本问题，使我更加重视可复现运行、日志证据和提交前的全流程自检。报告工作不仅是记录结论，还需要回看代码与日志、辨别哪些结果能够被证据支持，并将团队成员的工作组织成可复现的技术过程。

### 团队总结

本次启航营不仅提升了我们的 Ascend C 编程能力，也培养了从硬件执行视角、工程正确性和性能验证三个角度共同分析问题的习惯。这些经验对后续开发更复杂的融合算子和开展系统化性能优化都有很大帮助。
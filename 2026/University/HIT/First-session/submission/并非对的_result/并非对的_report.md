# 并非对的组实践报告

> **最终提交说明**：已于 2026 年 7 月 23 日完成 CANNJudge 评测，实际通过 **24/24** 条用例，与 Notebook 环境验证结果完全一致。本报告同时保留 Notebook 环境的完整验证数据，并如实记录开发过程中的排查与优化经历。

---

## 一、团队信息与贡献说明

### 1.1 团队基本信息

| 项目 | 内容 |
|---|---|
| 团队标识（组号） | 并非对的组 |
| CANNJudge 提交账号 | `XuGuangqi`（绑定 GitCode：`@2302_79785262`） |
| CANNJudge 提交状态 | 已提交 |
| CANNJudge 提交结果 | **24/24 通过** |

### 1.2 团队成员分工与贡献

| 姓名 | GitCode 账号 | 负责模块/任务分工 | 实际贡献说明 |
|---|---|---|---|
| 何松鑫（队长） | `@i_am_just_xingye` | Tiling 设计与多核分块策略 | 设计 `QmmCustomTilingData` 结构体，实现 `CalcQmmTiling` 与 `ChooseGrid` 函数；处理 M/N/K 对齐、核数选择、尾块处理、FRACTAL_NZ 约束及 workspace 大小计算；针对 Decode 小 M 场景优化 N 方向分块，Prefill 大 M 场景使用二维分块，并完成整体验收 |
| 朱永久 | `@yj319205` | Cube-only 与 Cube+Vector Kernel 实现 | 完成 INT8×INT8→INT32 的 Cube-only 路径，以及 INT32→FP32→BF16 的 Vector 反量化路径；处理 AIC/AIV 协作、跨核同步、per-channel/per-token scale 广播、尾块与数据搬运；优化 Vector 流水线，确保反量化精度与性能平衡 |
| 许光启 | `@2302_79785262` | Torch 接口、模型接入、测试与 PR | 完成 Torch 自定义算子注册与参数校验，将推理框架中的 `npu_quant_matmul` 替换为 `qmm_custom`；修复子进程动态库加载，编写单算子功能/性能测试脚本，完成 12 组规格的 INT32 与 BF16 精度验证；执行 Qwen3-8B 推理与 Profiling 汇总，整理报告、CANNJudge 提交和最终 PR |


---

## 二、结果展示

### 2.1 单算子精度比对结果

测试覆盖 12 组 `(M, K, N)` 规格，每组分别验证：

- **不传入 `perTokenScale`**：INT8×INT8，输出 INT32，与 CPU 参考结果精确比较（`rtol=0, atol=0`）；
- **传入 `perTokenScale`**：INT32 中间结果乘以 per-channel 与 per-token scale 后输出 BF16，以 `rtol=0.01, atol=0.01` 比较。

最终 Notebook 结果：

| 测试路径 | 通过数 | 总用例数 | 结果 |
|---|---|---|---|
| Cube-only，INT32 输出 | 12 | 12 | 全部通过 |
| Cube+Vector，BF16 输出 | 12 | 12 | 全部通过 |

Notebook 输出汇总为：**INT32: 12/12 通过, BF16: 12/12 通过**。

CANNJudge 在 Ascend 910B、CANN 9.0.0 环境中的最终实际结果为 **24/24 通过**。全部 24 个测试点均通过，输出错误占比均为 0.00%。具体耗时数据如下表（部分展示）：

| 测试点 | 结果 | 输出错误占比 | 用时 |
|---|---|---|---|
| 1 | Pass | 0.00% | 36.02 μs |
| 2 | Pass | 0.00% | 38.67 μs |
| 3 | Pass | 0.00% | 46.69 μs |
| 4 | Pass | 0.00% | 53.04μs |
| 5 | Pass | 0.00% | 147.42μs |
| 6 | Pass | 0.00% | 155.56μs |
| 7 | Pass | 0.00% | 89.13μs |
| 8 | Pass | 0.00% | 90.70μs |
| 9 | Pass | 0.00% | 49.26μs |
| 10 | Pass | 0.00% | 61.65μs |
| 11 | Pass | 0.00% | 71.42μs |
| 12 | Pass | 0.00% | 98.09μs |
| 13 | Pass | 0.00% | 204.69μs |
| 14 | Pass | 0.00% | 293.15μs |
| 15 | Pass | 0.00% | 114.67μs |
| 16 | Pass | 0.00% | 128.09μs |
| 17 | Pass | 0.00% | 494.71μs |
| 18 | Pass | 0.00% | 948.11μs |
| 19 | Pass | 0.00% | 741.04μs |
| 20 | Pass | 0.00% | 1.40ms |
| 21 | Pass | 0.00% | 3.06ms |
| 22 | Pass | 0.00% | 5.92ms |
| 23   | Pass | 0.00% | 1.45ms |
| 24 | Pass | 0.00% | 2.00 ms |

全部 24 个测试点耗时均合理，BF16 反量化路径在 CANN 9.0.0 工程化环境下同样正常通过，表明跨核同步与 Vector 流水线处理正确。

### 2.2 单算子性能测试结果

下表为最终一次完整运行中，从 `kernel_details.csv` 提取的 `QmmCustom` Duration，单位为微秒（μs）：

| M | K | N | INT32 Duration | BF16 Duration |
|---|---|---|---|---|
| 1 | 4096 | 4096 | 33.720 | 35.780 |
| 1 | 4096 | 6144 | 54.740 | 60.120 |
| 1 | 4096 | 24576 | 130.580 | 150.700 |
| 1 | 12288 | 4096 | 73.780 | 77.460 |
| 50 | 4096 | 4096 | 290.960 | 528.260 |
| 50 | 4096 | 6144 | 432.220 | 781.400 |
| 50 | 4096 | 24576 | 1887.740 | 3139.140 |
| 50 | 12288 | 4096 | 1107.220 | 1325.480 |
| 4096 | 4096 | 4096 | 13204.660 | 37501.600 |
| 4096 | 4096 | 6144 | 20085.820 | 56315.520 |
| 4096 | 4096 | 24576 | 83059.820 | 226402.160 |
| 4096 | 12288 | 4096 | 41097.200 | 86400.280 |

Profiler 共记录 24 条 `QmmCustom` 原始记录，覆盖所有规格的双路径。BF16 路径由于增加了反量化操作，耗时普遍高于 INT32 路径，但仍在可接受范围内。

### 2.3 算子接入模型性能测试结果

自定义算子被注册为 `torch.ops.ascendc_ops.qmm_custom`，并替换 Qwen3-8B W8A8 推理代码中的 `torch_npu.npu_quant_matmul`。模型输出文本与参考结果一致，开头为：

> The output of an attention function is a weighted sum of the value vectors, where the weights are determined by the similarity between the query vector and each key vector in the set of key-value pairs.

网络功能正确，生成内容符合预期。

最终一次完整运行结果：

| 项目 | 结果 |
|---|---|
| 普通推理 Decode 平均耗时 | 43.20 ms |
| Profiling 推理 Decode 平均耗时 | 41.91 ms |
| 课程网络性能满分阈值 | 低于 100 ms |
| 是否达到阈值 | **是** |

模型 Profiling 中的 `QmmCustom` 统计如下：

| 阶段 | 平均耗时（μs） | 最小耗时（μs） | 最大耗时（μs） | 调用次数 |
|---|---|---|---|---|
| Prefill | 62.734 | 34.240 | 114.522 | 144 |
| Decode | 47.403 | 21.121 | 97.621 | 432 |

Prefill 阶段 `QmmCustom` 调用集中在 M=50 的大矩阵，Decode 阶段主要为 M=1 的小矩阵，耗时分布与算子规格匹配。

---

## 三、方案说明

### 3.1 设计思路

#### 3.1.1 TilingData 与分块策略

`QmmCustomTilingData` 保存 CANN Matmul 库返回的 `TCubeTiling`，以及 `isPertoken`、M/N/K、`useNBlockSplit`、`nTileCount`、`usedCoreNum` 和 workspace 大小等字段。Tiling 阶段通过 `PlatformAscendCManager` 获取 AIC 数量，并根据 M 的大小选择分块方向：

- **Decode 小 M（M=1）**：采用 N 方向分块，每个核处理连续 N 块，每块大小为 `QMM_BASE_N=256`，避免 M 维度过小导致核空闲；
- **Prefill 或大 M**：采用 M 方向分块，使用 `SelectBaseM` 选择基础 M 块（M<64 时用 16，否则用 64），配合 `QMM_BASE_N=256`，使更多核参与并行；
- 调用 `matmul_tiling::MatmulApiTiling` 获取符合硬件约束的 `TCubeTiling`，并使用其实际返回的 `singleCoreM`/`singleCoreN` 计算块数和核数，确保地址偏移正确；
- 对于 BF16 路径，额外分配 `M×N×sizeof(int32_t)` 的 workspace 用于存放 Cube 输出的 INT32 中间结果。

#### 3.1.2 Cube-only Kernel

`QmmCubeBasicKernel` 仅在 AIC 上执行。每个逻辑核根据 `useNBlockSplit` 决定循环方式：

- 如果是 N 分块模式，每个核按 `coreIdx + usedCoreNum` 步长遍历 N 方向块，调用 `mm.SetSingleShape(1, validCols, K)` 和 `mm.SetTensorB` 定位到对应 FRACTAL_NZ 权重偏移，最后将 INT32 结果写入输出 GM；
- 否则使用 `mm.IterateAll` 一次性完成整个矩阵乘。

#### 3.1.3 Cube+Vector Kernel

`QmmPertokenKernel` 使用 `__mix__(1, 2)`，每个逻辑 AIC 配置两个 AIV，实现 Cube 与 Vector 协同：

1. AIC 完成 INT8 Matmul，并将 INT32 中间结果写入用户 workspace；
2. 通过 `SetFlag`/`WaitFlag` 确保 Cube 输出完成后再启动 Vector 操作；
3. Vector 单元从 workspace 读取 INT32 结果，依次执行：
   - Cast INT32→FP32；
   - 乘以 per-channel scale（沿 N 维广播）；
   - 乘以 per-token scale（沿 M 维广播）；
   - Cast FP32→BF16（RINT 舍入模式）；
4. 将 BF16 结果写回输出 GM，并处理尾块与行宽对齐。

#### 3.1.4 Torch 接口与模型接入

Host 侧 `qmm_custom` 函数检查设备、维度、dtype、scale 长度及 FRACTAL_NZ 对齐约束，根据是否传入 `perTokenScale` 选择 INT32 或 BF16 输出 dtype。通过 `TORCH_LIBRARY` 与 `TORCH_LIBRARY_IMPL` 注册算子，编译生成 `libascendc_ops.so`。模型接入时，在量化模块 `compressed_tensors_w8a8_int8.py` 中显式执行 `torch.ops.load_library`，并将 `torch_npu.npu_quant_matmul` 调用替换为 `torch.ops.ascendc_ops.qmm_custom`，同时适配输出 dtype 和 scale 传递逻辑。

### 3.2 问题解决与优化策略

| # | 问题 | 描述与解决方案 |
|---|---|---|
| 1 | **环境路径不一致** | Notebook 中部分路径硬编码为 `/opt/atomgit/...`，与 CANNLab 实际挂载目录 `/mnt/workspace/gitCode/...` 不符。团队使用 `locate_repo_root` 动态查找仓库根目录，并基于当前工作目录构建所有相关路径，确保运行无路径错误。 |
| 2 | **量化模型权重缺失** | 初始环境中没有 Qwen3-8B-W8A8 权重。团队先完成第五章的 AMCT W8A8 量化导出，生成 `config.json` 和权重文件，再执行第六章的自定义算子替换。 |
| 3 | **自定义算子在推理子进程中未注册** | 第一次替换后，`infer.sh` 启动的新 Python 进程无法找到 `ascendc_ops.qmm_custom`。原因是 Notebook 内核加载了动态库，但子进程未加载。解决方案是在 `compressed_tensors_w8a8_int8.py` 文件头部添加 `torch.ops.load_library` 语句，并确保路径正确，随后重新运行推理，返回码为 0。 |
| 4 | **依赖版本冲突** | AMCT 安装过程将 Torch 降级为 2.7.1，导致 `torch_npu` 不兼容。团队按 `requirements.txt` 重新安装 Torch 2.8.0 和 Torch-NPU 2.8.0.post4，并验证版本无误后重新编译算子。 |
| 5 | **Profiling CSV 解析异常** | 部分 `kernel_details.csv` 中 `Input Shapes` 和 `Output Data Types` 字段可能缺失或被引号包裹，导致 `groupby` 无法正确分组。团队增加字段清洗、缺失值填充，并使用 `Name` 和 `Type` 双重筛选，最终成功提取所有 `QmmCustom` 记录。 |

**性能优化细节**：

- Decode 场景（M=1）采用 N 方向分块，使多个核并行处理不同输出通道，降低延迟；
- Prefill 大 M 场景采用 M/N 二维分块，充分利用多核；
- 设置 `QMM_BASE_M=16`、`QMM_BASE_N=256` 作为基础块，兼顾缓存利用与核间负载均衡；
- BF16 路径使用两个 AIV 分摊 Vector 工作量，并分块（1024 元素）进行反量化，提升流水线效率；
- 使用平台返回的实际 `singleCoreM`/`singleCoreN` 计算地址偏移，避免因 Tiling 调整导致的越界错误。

### 3.3 AI 辅助使用说明

团队使用 AI 辅助分析 Notebook 执行日志、定位子进程加载问题、解析 Profiling CSV 字段，并协助整理测试结果和报告结构。AI 最初建议仅在 Notebook 中加载动态库，忽略了 `infer.sh` 会创建独立子进程，导致 `qmm_custom` 未定义错误。团队根据错误栈追踪到子进程边界，补充了模块加载语句，并通过"重启内核—全量运行—检查 24 项精度、输出文本和 Profiling 表"验证修复。AI 还提供了 CSV 清洗的候选代码，但最终解析逻辑（如字段去除引号、双列筛选）由团队根据实际数据格式调整。此次经历表明，AI 能加速问题定位，但硬件运行结果和多进程行为仍需人工确认。

---

## 四、收获与感悟

**何松鑫（队长）**

通过负责 Tiling 设计与多核分块，我深刻体会到算子性能不仅依赖于计算密集型 Kernel，更取决于如何在硬件限制下合理划分任务。从对齐要求、核数选择到 workspace 规划，每一个参数都会影响最终正确性和效率。特别是 Decode 与 Prefill 场景差异巨大，必须根据 M 规模动态调整分块策略。此外，平台返回的 Tiling 参数与期望值可能存在偏差，必须以实际返回值为准进行地址计算，这让我理解了"硬件约束优先"的工程原则。

**朱永久**

实现 Cube 与 Vector 协作的 Kernel 让我对昇腾 AI Core 的异构编程模型有了直观认识。AIC 负责矩阵乘，AIV 负责反量化，两者通过 GM workspace 和跨核同步协作，需要精细控制数据流和流水线。处理尾块时，必须对每个维度进行有效长度计算，避免越界；反量化时，scale 的广播和 Cast 舍入模式也要匹配精度要求。通过多次调试和单步验证，我意识到中间结果分层检查比只看最终输出更有效，这为日后调试复杂算子积累了经验。

**许光启**

从 Torch 接口封装到模型替换和性能测试，我经历了算子"从核函数到端到端推理"的完整链路。最棘手的环节是确保自定义算子在推理子进程中可用，这促使我理解 Python 进程边界和动态库加载机制。此外，依赖版本管理和 Profiling 数据解析也考验了细致排查能力。通过编写可复用的测试脚本和自动化汇总工具，我提高了工作效率，也认识到全流程自检（编译→单测→模型→性能）对于提交高质量成果至关重要。
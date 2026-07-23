# 生活总和我作队实践报告

> 最终提交说明：已于 2026 年 7 月 23 日使用指定账号完成 CANNJudge 评测，实际通过 17/24 条用例。本报告同时保留 Notebook 环境的 24/24 验证结果，并如实记录两个环境之间的差异。

## 一、团队信息与贡献说明

### 1.1 团队基本信息

- 团队标识（组号）：生活总和我作队
- CANNJudge 提交账号：`liuwenshuo`（绑定 GitCode：`2401_83590400`）
- CANNJudge 提交状态：已提交
- CANNJudge 提交结果：**17/24 通过**（提交账号：`liuwenshuo`，最近一次评测任务编号：`105958`）

### 1.2 团队成员分工与贡献

| 姓名 | GitCode 账号 | 负责模块/任务分工 | 实际贡献说明 | 对应 commit hash |
| --- | --- | --- | --- | --- |
| 李墨研（队长） | `2401_88721366` | Tiling 设计与多核分块 | 设计 `QmmCustomTilingData`，完成 `ChooseGrid` 与 `CalcQmmTiling`；处理 M/N/K 对齐、核数选择、尾块、FRACTAL_NZ 约束及 workspace 规划，并参与整体验收 | `9764d83eccc3a514f24121721ed218dfa918237b` |
| 陈竟然 | `chenjingran` | Cube-only 与 Cube+Vector Kernel | 完成 INT8×INT8→INT32 的 Cube 路径，以及 INT32→FP32→BF16 的 Vector 反量化路径；处理 AIC/AIV 协作、跨核同步、per-channel/per-token scale、尾块与数据搬运 | `55ccfc04d0ab9c3cdcc433a0a9958e4ea9d87ecb` |
| 刘文硕 | `2401_83590400` | Torch 接口、模型接入、测试与 PR | 完成 Torch 自定义算子注册与参数校验，将 `npu_quant_matmul` 替换为 `qmm_custom`；修复子进程动态库加载，完成单算子功能/性能测试、Qwen3-8B 推理与 Profiling 汇总，并负责报告、CANNJudge 和最终 PR | `e4824876c0ab863f423a06d3d8fea6d6d4913a07` |

### 1.3 团队协作说明

团队按照“算子 Tiling—Kernel 实现—框架接入与验证”的链路拆分任务。李墨研负责 Tiling 数据结构、分块和 workspace 规划；陈竟然负责两条 Kernel 路径及 Cube/Vector 协作；刘文硕负责 Torch 接口、Notebook 6.2 模型替换、测试汇总和提交工作。各模块集成后，团队共同在 CANNLab NPU 环境中从头运行 Notebook，依次验证编译、12 组双输出类型精度、单算子 Profiling、Qwen3-8B 文本输出以及网络 Profiling。最终 Notebook 的 13 个代码单元格执行编号连续为 1—13，且无异常输出。

## 二、结果展示

### 2.1 单算子精度比对结果

测试覆盖 12 组 `(M, K, N)` 规格，每组分别验证：

- 不传入 `perTokenScale`：INT8×INT8，输出 INT32，与 CPU 参考结果精确比较（`rtol=0, atol=0`）；
- 传入 `perTokenScale`：INT32 中间结果乘以 per-channel 与 per-token scale 后输出 BF16，以 `rtol=0.01, atol=0.01` 比较。

最终结果：

| 测试路径 | 通过数 | 总用例数 | 结果 |
| --- | ---: | ---: | --- |
| Cube-only，INT32 输出 | 12 | 12 | 全部通过 |
| Cube+Vector，BF16 输出 | 12 | 12 | 全部通过 |

Notebook 中共完成 24 项精度验证，汇总输出为：`INT32: 12/12 通过, BF16: 12/12 通过`。

CANNJudge 在 Ascend 910B、CANN 9.0.0 环境中的最终实际结果为 **17/24 通过**。全部 12 条 INT32 路径均通过；BF16 路径中 5 条通过、7 条未通过。未通过测试点如下：

| CANNJudge 测试点 | 结果 | 输出错误占比 |
| ---: | --- | ---: |
| 4 | Runtime Error | 100.00% |
| 10 | Wrong Answer | 5.46% |
| 12 | Runtime Error | 100.00% |
| 18 | Runtime Error | 100.00% |
| 20 | Wrong Answer | 58.23% |
| 22 | Runtime Error | 100.00% |
| 24 | Wrong Answer | 29.09% |

其余测试点 1–3、5–9、11、13–17、19、21、23 均通过。评测日志显示未通过项集中在带 `perTokenScale` 的 BF16 Cube+Vector 路径；Notebook 所用 CANN 8.5.2 环境能够完成该路径，而 CANNJudge 的 CANN 9.0.0 工程化编译环境仍存在跨核协作兼容性问题。本次提交以 CANNJudge 的 17/24 结果作为单算子最终验收记录。

### 2.2 单算子性能测试结果

下表为最终一次完整运行中，从 `kernel_details.csv` 提取的 `QmmCustom Duration`，单位为微秒（μs）：

| M | K | N | INT32 Duration | BF16 Duration |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4096 | 4096 | 16.360 | 17.440 |
| 1 | 4096 | 6144 | 20.641 | 21.580 |
| 1 | 4096 | 24576 | 82.401 | 79.921 |
| 1 | 12288 | 4096 | 29.861 | 30.961 |
| 50 | 4096 | 4096 | 19.900 | 29.140 |
| 50 | 4096 | 6144 | 25.480 | 33.141 |
| 50 | 4096 | 24576 | 98.602 | 111.083 |
| 50 | 12288 | 4096 | 52.301 | 60.961 |
| 4096 | 4096 | 4096 | 407.029 | 555.291 |
| 4096 | 4096 | 6144 | 609.732 | 900.078 |
| 4096 | 4096 | 24576 | 3003.160 | 4447.528 |
| 4096 | 12288 | 4096 | 1332.826 | 1490.549 |

Profiler 共找到 24 条 QmmCustom 原始记录，覆盖上述 12 组规格的 INT32 与 BF16 两条路径。

### 2.3 算子接入模型性能测试结果

自定义算子被注册为 `torch.ops.ascendc_ops.qmm_custom`，并替换 Qwen3-8B W8A8 推理代码中的 `torch_npu.npu_quant_matmul`。模型能够针对 Attention 提示词输出符合逻辑的向量加权与缩放点积公式说明，网络功能正确。

最终一次完整运行结果：

| 项目 | 结果 |
| --- | ---: |
| 普通推理 Decode 平均耗时 | 39.98 ms |
| Profiling 推理 Decode 平均耗时 | 39.32 ms |
| 课程网络性能满分阈值 | 低于 100 ms |
| 是否达到阈值 | 是 |

模型 Profiling 中的 QmmCustom 统计如下：

| 阶段 | 平均耗时（μs） | 最小耗时（μs） | 最大耗时（μs） | 调用次数 |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 62.734 | 34.240 | 114.522 | 144 |
| Decode | 47.403 | 21.121 | 97.621 | 432 |

## 三、方案说明

### 3.1 设计思路

#### 3.1.1 TilingData 与分块策略

`QmmCustomTilingData` 保存 CANN Matmul 库返回的 `TCubeTiling`，以及 `isPertoken`、M/N/K、单核 M/N、二维网格块数、Vector tile 大小和 workspace 信息。Tiling 阶段通过 `PlatformAscendCManager` 获取 AIC 数量，再由 `ChooseGrid` 选择逻辑核网格：

- Decode 或小 M（M≤64）场景只切 N，避免每核 M 过小；
- Prefill 或大 M 场景同时切分 M/N，优先使用更多 AIC；
- M 块按 16 对齐，INT8 FRACTAL_NZ 的 N/K 按 32 对齐；
- 同核数候选中倾向接近 `singleN ≈ 2 × singleM` 的分块，并根据规模选择最高 128×256 的基础块；
- 调用 `GetTiling` 后使用其实际返回的 `singleCoreM/N` 重新计算块数和地址，避免平台为满足 L1/L0 约束调整分块后产生 GM 偏移错误。

对于 BF16 反量化路径，在系统 workspace 后按 512 字节对齐分配 `M×N×sizeof(int32_t)` 的用户 workspace；INT32 路径无需该中间结果区。

#### 3.1.2 Cube-only Kernel

`QmmCubeBasicKernel` 仅在 AIC 上执行。每个逻辑核根据二维网格确定 M/N 起点和尾块大小，A 输入采用 ND，B 权重采用 INT8 FRACTAL_NZ，C 输出采用 ND INT32。Kernel 设置全局原始 shape 与实际尾块后调用 Matmul 接口，将结果直接写入输出 GM。

#### 3.1.3 Cube+Vector Kernel

`QmmPertokenKernel` 使用 `__mix__(1, 2)`，每个逻辑 AIC 配置两个 AIV：

1. AIC 完成 INT8 Matmul，并把 INT32 中间结果写入用户 workspace；
2. 通过 `CrossCoreSetFlag`/`CrossCoreWaitFlag` 保证 GM 搬出完成后再启动同组 AIV；
3. 两个 AIV 分摊当前 M 块的行，以 1024 个元素为 Vector tile 读取中间结果和 per-channel scale；
4. 依次执行 INT32→FP32 Cast、乘 per-channel scale、乘 per-token scale，再以 RINT 模式 Cast 为 BF16 并写回 GM。

#### 3.1.4 Torch 接口与模型接入

Host 侧接口检查设备、维度、dtype、M/N/K、scale 长度及 FRACTAL_NZ 对齐约束，根据是否传入 `perTokenScale` 选择 INT32 或 BF16 输出。通过 `TORCH_LIBRARY` 与 `TORCH_LIBRARY_IMPL` 注册 `qmm_custom`，编译得到 `libascendc_ops.so`。模型接入时，在推理子进程所导入的量化模块中显式执行 `torch.ops.load_library(...)`，再将原量化矩阵乘调用替换为 `torch.ops.ascendc_ops.qmm_custom(...)`。

### 3.2 问题解决与优化策略

1. **工程路径不一致**：Notebook 原始环境中的部分路径与当前 CANNLab 挂载目录不一致。团队将工程定位改为基于当前仓库和 `/mnt/workspace/gitCode/cann/cann-learning-hub` 的实际路径，并在运行前验证模型、YAML、动态库和推理脚本是否存在。
2. **量化模型缺失**：组员环境中最初没有 Qwen3-8B-W8A8 权重。团队先完成第五章 AMCT W8A8 量化导出，确认 `config.json` 和权重目录存在后再运行第六章。
3. **自定义算子在推理子进程中未注册**：第一次仅替换 Python 调用，推理时报出 `ascendc_ops` 没有 `qmm_custom` 属性。原因是 Notebook 内核加载了动态库，但 `infer.sh` 创建的新进程没有加载。修复方式是在目标量化模块导入阶段写入 `torch.ops.load_library`，随后重新推理并确认返回码为 0。
4. **依赖版本变化**：AMCT 安装过程一度将 Torch/Torch-NPU 切换为 2.7.1。团队按第六章依赖恢复到 Torch 2.8.0 与 Torch-NPU 2.8.0.post4，并实际导入验证版本，再重新编译算子。
5. **Profiling 表为空**：Profiler 的 `Input Shapes` 与 `Output Data Types` 字段可能为空，原始 `groupby` 会丢弃记录。团队增加 `Name/Type` 双列筛选、缺失值填充和数值转换，最终获得 Prefill/Decode 的非空耗时表。
6. **性能优化**：小 M 场景采用 N 方向切分，大 M 场景使用二维分块；设置 128×256 等分级基础块；使用实际 Tiling 返回值计算偏移；BF16 路径由两个 AIV 分摊行并分块反量化。这些措施兼顾 Decode 延迟、Prefill 并行度、FRACTAL_NZ 地址正确性和 Vector 流水效率。
7. **CANNJudge 环境差异**：Notebook 在 CANN 8.5.2 环境中完成 24/24 验证；将核函数直调代码调整为 CANN 9.0.0 自定义算子工程后，编译与全部 INT32 用例通过，但部分 BF16 Cube+Vector 用例出现数值错误或跨核等待超时。团队根据日志尝试了 Vector 流水屏障、混合核映射和双向同步等兼容性调整，最终评测仍为 17/24，因此在报告中保留真实结果，并将该问题记录为后续优化方向。

### 3.3 AI 辅助使用说明

团队使用 AI 辅助分析环境路径、Notebook 执行日志、动态库注册位置和 Profiling CSV 字段，并协助整理测试结果及报告结构。AI 的第一次模型替换建议只关注了函数调用，没有充分考虑 `infer.sh` 会创建独立 Python 子进程，因此出现了动态库未注册的新问题。团队没有直接把建议当作正确结果，而是依据 `AttributeError` 追踪进程边界，补充子进程加载语句，并通过“重启内核—依次运行全部单元格—检查返回码、24 项精度结果、模型回答和 Profiling 表”的方式完成验证。此次经历说明，AI 适合加速定位和生成候选方案，但硬件算子、依赖版本和多进程行为仍必须通过真实 NPU 运行结果确认。

## 四、收获与感悟

### 李墨研

通过负责 Tiling 与多核分块，我认识到算子正确性不仅取决于数学公式，还取决于对齐、存储格式、核数和 workspace 等硬件约束。将平台返回的实际 Tiling 参数用于地址计算，也让我理解了工程实现中“以真实运行约束为准”的重要性。

### 陈竟然

Cube+Vector 路径让我更直观地理解了 AIC 与 AIV 的分工，以及跨核同步和数据搬运对结果正确性与性能的共同影响。面对尾块和反量化误差时，分层检查中间结果比只看最终输出更有效。

### 刘文硕

从 Torch 接口到 Qwen3-8B 模型替换和 Profiling，我体会到算子接入是一个跨编译、框架、子进程和模型的完整链路。此次排查动态库注册与依赖版本问题，使我更加重视可复现运行、日志证据和提交前的全流程自检。

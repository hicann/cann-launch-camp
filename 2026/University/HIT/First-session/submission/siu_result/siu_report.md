# siu 实践报告

## 一、团队信息与贡献说明

### 1.1 团队基本信息

- 团队标识（组号）：siu
- 本次提交对应 Notebook：`siu_result.ipynb`（已完整运行，保留全部单元格输出）

### 1.2 团队成员分工与贡献

| 姓名 | GitCode 账号 | 负责模块/任务分工 | 实际贡献说明 | 对应 commit hash |
| --- | --- | --- | --- | --- |
| 乔首智 | @2303_79681815 | Tiling 计算、Cube-only（无 pertoken_scale）Kernel 实现 | 完成 `QmmCustomTilingData` 结构体设计与 `CalcQmmTiling` host 侧 tiling 函数：基于 `matmul_tiling::MatmulApiTiling` 配置 A/B/C 的 `TPosition`/`CubeFormat`/`DataType`，针对 decode（M=1）场景设计按 N 维切分（`useNBlockSplit`，`QMM_BASE_N=256`）以避免单核瓶颈，针对 prefill/大 M 场景设计 `SelectBaseM` 分级取 `baseM`（M<16 取 16，否则取 64）；实现 `QmmCubeBasicKernel`，完成无 pertoken_scale 时纯 Cube `INT8×INT8→INT32` 的整表 `IterateAll` 及按 N 分块并行两条路径 
| 高晨宇 | @2301_79619654 | Cube+Vector（pertoken_scale）Kernel 实现与优化、Notebook 集成与模型验证、PR 汇总发起 | 实现 `QmmPertokenKernel`：Cube 侧通过 `TPosition::VECIN` 把 INT32 累加结果直接交给 Vector 侧（避免 GM 往返），Vector 侧实现 `MulScaleBroadcast`（基于 `AscendC::BinaryRepeatParams`，`src1RepStride=0`，一次 `Mul` 覆盖多行复用同一份 per-channel scale）与 `MulPertokenBroadcast`（基于 `AscendC::Brcb` + `BrcbRepeatParams` 把 per-token scale 从"每行一个标量"展开为逐 8 元素 block，再一次 `Mul` 跨列广播，非 8 对齐尾部行逐行 `Muls` 兜底）两处向量化优化，替代原始逐行反量化循环；完成 Notebook 第 4 节编译调试、第 5 节 12 组 shape × 2 条路径的功能与性能测试、第 6 节模型集成（替换 `torch_npu.npu_quant_matmul` 为 `qmm_custom`）与 Qwen3-8B 真实推理验证，定位并修复 `scheduler_config.max_new_tokens` 配置过小导致的输出截断问题，完成第 7 节 Profiling 性能采集与分析 

### 1.3 团队协作说明

乔首智负责自定义算子的 Tiling 设计与无 pertoken_scale 的纯 Cube 矩乘路径（`QmmCustomTilingData`/`CalcQmmTiling`/`QmmCubeBasicKernel`），高晨宇负责有 pertoken_scale 的 Cube+Vector 反量化路径（`QmmPertokenKernel`）及其向量化优化，并完成 Notebook 后续所有编译、测试、模型集成与性能分析工作。双方在算子 kernel 入口（`__mix__(1,2)` 混合核）、PyTorch 算子注册（`TORCH_LIBRARY`/`TORCH_LIBRARY_IMPL`）、ACL profiling 埋点等课程提供的参考实现部分保持一致，未做改动。成果统一汇总到 `submission/siu_result`，由高晨宇作为 PR 发起人，将两人各自 Fork 分支的有效提交合并后发起最终 Pull Request。

## 二、结果展示

### 2.1 单算子精度比对结果

Notebook 第 5.1 节对 12 组 shape（覆盖 M=1/50/4096，K=4096/12288，N=4096/6144/24576）分别执行 INT32（无 pertoken_scale）与 BF16（有 pertoken_scale）两条路径，共 24 条用例：

| 路径 | 判定标准 | 通过数 |
| --- | --- | ---: |
| Cube-only，INT32 输出 | `rtol=0, atol=0`（严格精确匹配 float64 参考实现） | 12/12 |
| Cube+Vector，BF16 输出 | `rtol=0.01, atol=0.01` | 12/12 |
| 合计 | — | 24/24 |

全部 12 组 shape 的 INT32/BF16 两条路径均为 `PASS`，无失败用例。

### 2.2 单算子性能测试结果

Notebook 第 5.2 节使用 `torch_npu.profiler` 采集单算子 Duration（单位 us），12 组 shape 结果如下：

| M | K | N | INT32 Duration(us) | BF16 Duration(us) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4096 | 4096 | 32.939 | 35.440 |
| 1 | 4096 | 6144 | 55.639 | 60.979 |
| 1 | 4096 | 24576 | 131.778 | 148.417 |
| 1 | 12288 | 4096 | 73.118 | 76.318 |
| 50 | 4096 | 4096 | 291.274 | 479.070 |
| 50 | 4096 | 6144 | 431.751 | 714.106 |
| 50 | 4096 | 24576 | 1883.943 | 2858.923 |
| 50 | 12288 | 4096 | 1106.377 | 1249.815 |
| 4096 | 4096 | 4096 | 13098.138 | 33747.005 |
| 4096 | 4096 | 6144 | 20160.217 | 50125.459 |
| 4096 | 4096 | 24576 | 82674.649 | 201459.895 |
| 4096 | 12288 | 4096 | 40538.070 | 81254.516 |

可以看到 decode 场景（M=1）下 INT32/BF16 耗时接近（按 N 分块切分、Cube+Vector 额外开销较小），而 prefill 大 M 场景下 BF16 路径（含反量化）耗时明显高于纯 Cube 的 INT32 路径，符合 Cube+Vector 协同计算的预期开销。


### 2.3 算子接入模型性能测试结果

QmmCustom 已替换 Qwen3-8B（W8A8 量化）推理链路中 `CompressedTensorsW8A8Int8LinearMethod.apply_weights` 里的 `torch_npu.npu_quant_matmul` 调用（Notebook 6.2 节），并完成端到端验证：

- **推理功能**：替换生效后，模型可正常执行 prefill + decode 推理。修复 `max_new_tokens` 配置后（见下文），对于测试输入 "An attention function can be described as mapping a query and a set of key-value pairs to an output..."，模型生成了完整、语义正确的关于 attention 机制的说明文字，未出现由算子替换引入的功能错误。
- **推理耗时**（batch_size=1，`max_new_tokens=512`）：Warm-up prefill 耗时 192.70 ms；decode 阶段 27 个 token 平均耗时 42.52 ms/token。
- **输出截断问题排查**：第一次运行 6.3 节替换后的推理时，发现输出在 "…The output is a vector that is a" 处戛然而止。经排查确认这不是算子的 bug，而是运行时 YAML（`qwen3_8b_a8w8_custom.yaml`，继承自模板 `qwen3_8b_a8w8_1tp.yaml`）中 `scheduler_config.max_new_tokens` 默认值为 256，推理引擎在生成满 256 个 token 后按预期停止，输出内容本身就是这么长，并非日志显示被截断。将该值改为 512 并重新推理后，输出可以正常说完整（见上文完整生成文本）。同时发现 `prepare_runtime_yaml` 每次调用都会用模板重新生成自定义 YAML、覆盖之前对该字段的手动修改，因此把 `max_new_tokens` 的覆盖代码合并进同一个推理 cell（放在 `prepare_runtime_yaml` 之后、`subprocess.run` 之前），保证重复运行该 cell 时改动不会被覆盖。
- **第 7 节 Profiling 结果**：开启 Profiler 后按 `Input Shapes` 分组统计 QmmCustom 各 shape 的平均耗时（`kernel_details.csv`）：

| Phase | Input Shapes (M,K;K,N;N[;M]) | Output Dtype | Avg Duration(us) | Calls |
| --- | --- | --- | ---: | ---: |
| Prefill | 50,4096; 4096,24576; 24576 | INT32 | 2076.60 | 36 |
| Prefill | 50,12288; 12288,4096; 4096; 50 | BF16 | 1342.21 | 36 |
| Prefill | 50,4096; 4096,6144; 6144; 50 | BF16 | 750.14 | 36 |
| Prefill | 50,4096; 4096,4096; 4096; 50 | BF16 | 503.14 | 36 |
| Decode | 1,4096; 4096,24576; 24576 | INT32 | 129.69 | 108 |
| Decode | 1,12288; 12288,4096; 4096; 1 | BF16 | 87.64 | 108 |
| Decode | 1,4096; 4096,6144; 6144; 1 | BF16 | 64.53 | 108 |
| Decode | 1,4096; 4096,4096; 4096; 1 | BF16 | 39.20 | 108 |

可以看到模型真实推理场景中出现的 shape 与第 5 节离线基准测试的部分 shape 一致（如 M=50/1，K=4096/12288，N=4096/6144/24576），QmmCustom 在 prefill 和 decode 阶段均被正确调用且平均耗时稳定。

## 三、方案说明

### 3.1 设计思路

算子输入为 INT8 激活 `x1[M,K]` 与 INT8 权重 `x2[K,N]`（FRACTAL_NZ 格式），`scale` 为 per-channel（沿 N 维，shape `[N]`）缩放，`pertoken_scale` 为可选的 per-token（沿 M 维，shape `[M]`）缩放。`QmmCustomTilingData` 包含 `TCubeTiling`、模式标记 `isPertoken`、原始 `M/N/K`、decode 场景的按 N 分块标记 `useNBlockSplit`/`nTileCount` 及 `usedCoreNum`/`workspaceSize`，由 `CalcQmmTiling` 在 host 侧统一计算：decode（M=1）时按 `QMM_BASE_N=256` 把 N 维切成多个独立 tile 分配到各 AIC（`FIRSTN` 遍历），避免只有单核工作；其余场景按 `SelectBaseM(M)`（M<16 取 16，否则取 64）与 `QMM_BASE_N` 做二维分块（`FIRSTM` 遍历）。

无 `pertoken_scale` 时，`QmmCubeBasicKernel` 执行纯 Cube 的 `INT8×INT8 → INT32` Matmul：decode 场景下按 N 分块（`ProcessNBlockSplit`），其余场景整表 `IterateAll` 直接写出，不涉及 Vector 侧计算。

有 `pertoken_scale` 时，`QmmPertokenKernel` 采用 Cube+Vector 协同：Cube 侧的输出类型设为 `TPosition::VECIN`，通过 `Iterate<true>()` + `GetTensorC<true>(...)` 把 INT32 累加结果直接交给 Vector 侧（避免 GM 往返），Vector 侧完成 FP32 转换、双 scale 反量化和 BF16 转换。反量化阶段的两项优化：

1. **Scale（per-channel）广播**：`MulScaleBroadcast` 利用 `AscendC::BinaryRepeatParams` 将 `src1RepStride` 设为 0，使一次 `Mul` 调用的多个 repeat（每个 repeat 对应一行）复用同一份 scale 数据，替代逐行调用 `Mul` 的写法；由于 tile 的物理行跨度固定为 `baseN`（尾块列数不足时后面几列是无效数据），`dstRepStride`/`src0RepStride` 均按 `baseN/8` 计算。
2. **Pertoken（per-token）广播**：`MulPertokenBroadcast` 先用 `AscendC::Brcb`（配合 `BrcbRepeatParams`）把"每行一个标量"的 `pertoken_scale` 展开成"每行一个完整 8 元素 block"，再用一次 `Mul`（`src1BlkStride=0, src1RepStride=1`）把展开后的数据广播到该行所有列；仅对 8 对齐的行数（`alignedM`）走该路径，其余尾部行仍使用逐行 `Muls`（通过 `LocalTensor::GetValue(row)`）兜底，保证正确性。

```text
x1(INT8) + x2(INT8, FRACTAL_NZ)
          |
          v
   Cube INT8 Matmul
          |
          +----------------------> INT32 输出（无 pertoken_scale，QmmCubeBasicKernel）
          |
          v
  INT32（VECIN，Cube 直接写入）
          |
          v
 Vector: FP32 Cast -> MulScaleBroadcast(BinaryRepeatParams)
                   -> MulPertokenBroadcast(Brcb + BinaryRepeatParams)
                   -> BF16 Cast
          |
          v
      BF16 输出（有 pertoken_scale，QmmPertokenKernel）
```

### 3.2 问题解决与优化策略

1. 反量化阶段若对 per-channel scale 和 per-token scale 均采用逐行标量乘法，行数越多标量循环开销越明显。改用 `MulScaleBroadcast`/`MulPertokenBroadcast` 整行、跨列广播乘法后，同一个 tile 内的乘法指令下发次数大幅减少，同时保持结果与参考实现（float64 matmul + 手工反量化）完全一致（INT32 路径 `atol=0` 精确匹配，BF16 路径在 `rtol=atol=0.01` 内全部通过）。
2. decode（M=1）场景若按 M 维切分，会导致单次调用只有一个核在工作、其余核空闲；改为按 `QMM_BASE_N=256` 对 N 维切分（`useNBlockSplit`），把同一行、不同输出列的独立计算分配到多个 AIC，充分利用多核资源，这一设计同时用于 `QmmCubeBasicKernel` 和 `QmmPertokenKernel` 的 decode 路径。
3. Notebook 6.3 节模型集成测试中，第一次运行发现推理输出被截断，定位为运行时 YAML 配置 `scheduler_config.max_new_tokens` 默认值 256 导致，而非算子替换引入的问题；同时发现 `prepare_runtime_yaml` 每次调用都会用模板重新生成自定义 YAML、覆盖之前对该字段的手动修改，因此将该字段的覆盖代码合并进同一个推理 cell（放在 `prepare_runtime_yaml` 之后、`subprocess.run` 之前），使其在重复运行该 cell 时保持幂等，修复后模型可以正常输出完整语句。
4. 将 `QmmCubeBasicKernel`/`QmmPertokenKernel` 拆分为纯 Cube 与 Cube+Vector 两条独立路径（而非用同一套代码硬套两种输出类型），使无 pertoken_scale 场景不承担任何 Vector 侧开销，实测该场景在多组 decode/prefill shape 下的 Duration 明显低于对应的 BF16 路径（如 M=4096,K=4096,N=24576 时 INT32 为 82674.649us，BF16 为 201459.895us）。
5. 本实践使用 AI 辅助进行代码审查、API 用法确认（如确认 `AscendC` 命名空间下正确的结构体名为 `BrcbRepeatParams` 而非误写的 `BrcbRepParams`）、模型集成阶段"输出截断是配置问题还是算子 bug"的排查思路梳理。

## 四、收获与感悟

乔首智：这次实践让我对 Tiling 设计有了更具体的理解——同一个算子在不同 shape 下的最优分块策略并不统一。刚开始我按照惯性思路统一用 M 维切分多核，调试 decode（M=1）场景时才发现这样会导致只有一个核在工作、其余核全部空闲，性能非常差；改成按 N 维切 `QMM_BASE_N=256` 个 tile 分给多个 AIC 之后，decode 场景才真正利用起了多核。这让我意识到 Tiling 不是把 `MatmulApiTiling` 的 API 调对就完事了，而是要先分析清楚不同 shape（尤其是 M=1 这种极端场景）下计算和数据的实际分布，再决定分块维度。另外在使用 `SetFixSplit`/`SetTraverse` 时也体会到，`FIRSTN`/`FIRSTM` 遍历顺序、`baseM`/`baseN` 的取值都会直接影响多核间的负载均衡，需要针对具体场景验证而不能照搬默认配置。

高晨宇：我负责的 `QmmPertokenKernel` 让我第一次系统接触到 Ascend C 里"用向量化指令代替标量循环"的优化思路。一开始很容易写出逐行调用 `Mul`/`Muls` 的朴素实现，功能上没问题，但意识到反量化这一步本质上是"同一份 scale 数据在多行/多列间重复使用"之后，才想到可以用 `BinaryRepeatParams` 的 `RepStride=0` 和 `Brcb` 展开去把这种重复利用起来，一次指令处理多行数据，而不是让向量单元反复读取同一份小数据。这个过程也让我理解到广播优化不是无脑套用，必须先保证非对齐场景（比如行数不是 8 的倍数）有正确的兜底逻辑，否则很容易在边界 case 上出错。另外在模型集成阶段，"输出被截断"最初看起来很像算子引入的 bug，但通过对照配置文件、确认是 `max_new_tokens` 限制而非计算错误，让我体会到定位问题时要先划清"数据/配置层面的限制"和"计算逻辑本身的错误"这两类完全不同的原因，不能看到异常现象就急着往算子实现上找原因。

# 大狗嚼俱乐部实践报告

## 一、团队信息与贡献说明

### 1.1 团队基本信息

- 团队标识（组号）：大狗嚼俱乐部
- CANNJudge 提交账号：卢明逸
- CANNJudge 提交结果：24/24 测试点通过，误差均为 0.00%，得分 96.94，榜单第 1 名
- CANNJudge 链接：https://cannjudge.cn/hit/20260721/qmmcustom/ranking

### 1.2 团队成员分工与贡献

| 姓名 | GitCode 账号 | 负责模块/任务分工 | 实际贡献说明 | 对应 commit hash |
| --- | --- | --- | --- | --- |
| 卢明逸 | ctzlyiii | Tiling、Cube-only、Cube+Vector Kernel 与性能优化 | 完成 QmmCustom 主体、Notebook 与 CANNJudge 工程；实现任务类型拆分、5×4 Tiling、Scale/Token 缓存、三缓冲及批量 DMA，取得 96.94 分和榜单第 1 | `8427c6f`、`49cce28` |
| 王乐天 | 2301_81258087 | 单算子复测、模型接入、Profiling 与兼容性分析 | 在 CANN 8.5.2/Ascend 910 环境完成单算子复测与模型接入，采集性能，定位并修复反量化乘法结合顺序导致的网络输出差异 | `7b790a9`、`be83f61` |

### 1.3 团队协作说明

卢明逸负责自定义算子的 Tiling、Kernel 主体和 CANNJudge 性能迭代，并将阶段成果提交到团队 Fork。王乐天从团队提交 `8427c6f` 建立个人分支，在独立云端 NPU 环境复测算子，定位 CANN 版本及 FRACTAL_NZ 内部格式问题，随后将算子接入 Qwen3-8B-W8A8 并完成模型推理与 Profiling。卢明逸在模型正确性修复基础上继续完成任务类型、缓存、队列和 DMA 优化，CANNJudge 最终达到榜单第 1。成果统一汇总到 `submission/大狗嚼俱乐部_result`，保留双方各自 GitCode 身份形成的有效提交。

## 二、结果展示

### 2.1 单算子精度比对结果

在 Ascend 910、CANN 8.5.2、torch 2.8.0+cpu、torch_npu 2.8.0.post4 环境复测 12 组 shape，每组覆盖 INT32 与 BF16 两条路径，共 24 条用例：

| 路径 | 通过数 | 最大绝对误差 |
| --- | ---: | ---: |
| Cube-only，INT32 输出 | 12/12 | 0 |
| Cube+Vector，BF16 输出 | 12/12 | 0.0 |
| 合计 | 24/24 | 0 |

覆盖 M=1、50、4096，K=4096、12288，N=4096、6144、24576。测试前设置 `torch.npu.config.allow_internal_format = True`，确保 `npu_format_cast(..., 29)` 生成真实 FRACTAL_NZ 数据。

### 2.2 单算子性能测试结果

CANNJudge 由卢明逸账号提交，最终 24 个测试点全部通过，输出误差均为 0.00%，综合得分 96.94，榜单第 1 名。平台记录时间为 2026-07-23 13:52:29。各 Shape 按 INT32/BF16 成对对应测试点 1-24，最终用时如下：

| M | K | N | INT32（μs） | BF16（μs） |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4096 | 4096 | 27.74 | 32.65 |
| 1 | 4096 | 6144 | 37.60 | 42.71 |
| 1 | 4096 | 24576 | 138.20 | 146.89 |
| 1 | 12288 | 4096 | 71.40 | 76.66 |
| 50 | 4096 | 4096 | 26.86 | 37.42 |
| 50 | 4096 | 6144 | 37.90 | 49.71 |
| 50 | 4096 | 24576 | 147.50 | 162.71 |
| 50 | 12288 | 4096 | 73.24 | 84.61 |
| 4096 | 4096 | 4096 | 465.21 | 519.29 |
| 4096 | 4096 | 6144 | 687.88 | 907.68 |
| 4096 | 4096 | 24576 | 2750.00 | 3700.00 |
| 4096 | 12288 | 4096 | 1340.00 | 1450.00 |

表中最后四项原始显示为毫秒的数值已统一换算为微秒。Notebook 仍保留 Ascend 910/CANN 8.5.2 模型接入环境的完整 Profiler 输出；由于 CANNJudge 运行环境、计时范围和自定义算子工程入口不同，单算子最终验收以平台排行榜数据为准。

### 2.3 算子接入模型性能测试结果

QmmCustom 已替换 `CompressedTensorsW8A8Int8LinearMethod.apply_weights` 中的 `torch_npu.npu_quant_matmul`，并加载自定义 `libascendc_ops.so`。

| 测试路径 | Prefill | Decode 平均 | 判定 |
| --- | ---: | ---: | --- |
| 优化后自定义 QmmCustom | 91.52 ms | 81.99 ms | 低于 100 ms，PASS |
| 优化后自定义 QmmCustom，Profiler 开启 | 109.30 ms | 82.99 ms | 数据采集成功 |
| 内置 npu_quant_matmul 基线（前次记录） | 37.70 ms | 33.07 ms | 历史基线 |

网络端到端耗时受云实例负载、CANN/模型运行时版本和 Profiler 开销影响，本轮环境的绝对值不与前次历史基线直接计算加速比；Tiling 优化收益采用同一进程、同一 NPU 的单算子 A/B 中位数评价。

网络输入为：

```text
An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is
```

自定义算子接入后的输出以如下 attention 描述开头：

```text
The output of an **attention function** is typically a **weighted sum** of the **value vectors**, where the weights are determined by the **similarity** between the **query vector** and the **key vectors** from the set of key-value pairs.
```

该结果符合课程要求的 attention 逻辑描述。进一步使用确定性 `argmax` 解码与内置算子基线自动比较，双方完整输出均为 740 个字符，逐字符严格一致，网络功能验证 PASS。

Profiler 的 `kernel_details.csv` 中，CANN 8.5.2 通过 kernel 名 `_Z17qmm_custom_kernel...` 识别自定义算子：

| 阶段 | 调用次数 | 平均耗时 | 总耗时 | 最小耗时 | 最大耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill | 144 | 57.871 us | 8333.380 us | 30.360 us | 101.200 us |
| Decode 采样窗口 | 432 | 46.783 us | 20210.280 us | 20.560 us | 84.140 us |

推理框架采用确定性的 `argmax` 贪心解码。网络首 token 差异定位为反量化乘法结合顺序：原实现为 `(acc × per-channel) × per-token`，内置算子为 `acc × (per-channel × per-token)`。修复后，自定义与内置基线的完整生成文本均为 740 个字符，逐字符严格一致，网络功能判定 PASS。

## 三、方案说明

### 3.1 设计思路

算子输入为 ND 格式 INT8 激活 `x1[M,K]` 与 FRACTAL_NZ 格式 INT8 权重 `x2[K,N]`。Host 侧按照题目约束将 `x2` 原型声明为 ND，Kernel 内按 Cube 原生 FRACTAL_NZ 排布读取。

`QmmCustomTilingData` 包含 `TCubeTiling`、模式标记 `isPertoken`、原始 `M/N/K`、每核 `singleCoreM/singleCoreN` 和 workspace 大小。Tiling 根据矩阵形状和可用 AIC 数选择核网格：小 M 主要沿 N 维切分，N=6144 时 M=1 使用 16 个 N 分块、M=50 使用 12 个均衡的 512 列分块；其他小 M 形状按 256 列目标宽度切分。大 M 使用 `mDim=5、nDim=4` 的二维切分覆盖 20 个 AIC，并将 M/N 分块分别按 16/32 对齐。Kernel 再根据 `iterateOrder`、block id 和尾块范围计算每核 A/B/C 地址，兼容 M=1、M=50 等非 32 整倍数场景。

无 `pertoken_scale` 时启动纯 AIC 的 Cube-only Kernel，执行 `INT8 x INT8 -> INT32` Matmul 并直接写入输出。存在 `pertoken_scale` 时走 Cube+Vector：M=1 使用 1:1 AIC/AIV，其余形状使用 1:2 AIC/AIV。Cube 先把 INT32 累加结果写入独立 intermediate tensor，Matmul KFC 通信区使用单独的系统 workspace；Vector 使用 6144 列 tile 和三缓冲队列，按列块缓存 per-channel scale，大 M 再缓存本核负责的 per-token scale。stride 可编码时，多行 GM 搬运合并为 `DataCopyPad`，否则回退逐行路径，最后完成 FP32 转换、`scale[n] * pertoken_scale[m]` 合并缩放和 BF16 转换。

数据流如下：

```text
x1(ND) + x2(FRACTAL_NZ)
          |
          v
   Cube INT8 Matmul
          |
          +----------------------> INT32 输出（非 per-token 模式）
          |
          v
 INT32 intermediate tensor
          |
          v
 Vector: FP32 Cast -> 合并双 Scale -> BF16 Cast
          |
          v
      BF16 输出（per-token 模式）
```

### 3.2 问题解决与优化策略

1. 队友源码中的 `aclprofStr2Id`、`aclprofRangePushEx`、`aclprofRangePop` 在当前 CANN 8.5.2 中不存在。复测副本仅移除 Profiling 元数据与 range 标记，保留 Tiling、workspace、pertoken 指针、Kernel launch 和计算逻辑。不同 CANN 环境可按 API 实际可用性调整这些标记。
2. 初次复测中，未打开内部格式支持时 `npu_format_cast(..., 29)` 没有得到真实 FRACTAL_NZ 数据，导致用例失败。设置 `torch.npu.config.allow_internal_format = True` 后，24/24 用例全部通过且误差为 0。
3. CANN 8.5.2 缺少上述 range 标记后，Profiler CSV 的 `Type` 列不显示 `QmmCustom`。统计逻辑增加按 `Name` 包含 `qmm_custom_kernel` 的兼容筛选，成功获取 Prefill 与 Decode 数据。
4. 优化后普通推理的 Prefill/Decode 平均耗时为 91.52/81.99 ms，满足课程低于 100 ms 的目标。Profiler 开启时 Prefill 为 109.30 ms，属于采集开销下的观测值，不作为普通推理判定。
5. 本实践使用 Codex 辅助检查版本 API、组织 24 条复测、分析 Profiler CSV 和补充 Notebook/报告。AI 建议没有直接作为通过依据：所有关键结论均由 Ascend 910 实际运行日志、误差结果和 CSV 数据验证。AI 辅助的早期实现采用了数学等价但数值舍入不等价的反量化乘法顺序，单算子容差测试未暴露问题，却改变了网络输出；团队通过确定性网络对比和逐元素诊断将其识别为新 bug，而不是把语义相似误判为严格一致，并在修复后重新完成单算子、网络和 Profiler 验证。
6. 对真实 Prefill `M=50` 的四组 K/N 逐元素诊断发现，原实现与内置算子每组仅有 3 至 5 个 BF16 元素发生 1 ULP 舍入差异，但经过多层传播足以改变最大 logit。将 Vector 路径改为先计算 `per-channel × per-token`，再乘 INT32 累加值后，四组算子输出与内置实现逐位一致，完整网络文本也恢复一致。
7. 性能优化以多核切分和 Cube/Vector 流水为主：Tiling 从单核执行扩展为最多 20 个 AIC 的 M/N 网格，Vector epilogue 使用 `TQue` 管理搬入、计算和搬出依赖，并由两个 AIV 分摊每个 AIC 的行处理。针对 CANNJudge 性能差距最大的 `M=50、K=4096、N=6144`，同一台 Ascend 910B3 上连续采样 6 次并取中位数：原 `1x20` 网格的 INT32/BF16 Kernel 为 46.37/53.07 us；将该 N 维改为 12 个均衡的 512 列分块后降至 23.12/29.50 us，分别降低 50.1%/44.4%。候选 `4x5` 网格为 46.61/54.14 us，未带来收益，因而未采用。优化版再次按官方阈值完成 24/24 精度复测，再进入模型接入验证。
8. 最终性能迭代进一步将 INT32 与 BF16 拆为不同任务类型：INT32 仅启动 AIC，M=1 BF16 使用 1:1 混合核，其余 BF16 使用 1:2；大矩阵改用 `5x4` 网格。Vector 侧把 tile 扩大到 6144，per-channel scale 从逐行搬入改为每列块一次，大 M 的 per-token scale 缓存在 UB，并使用三缓冲及带 stride 的多行 DMA。直接测试 24/24 通过，最终 CANNJudge 24/24 Pass、误差 0.00%、96.94 分、榜单第 1；相较上一版，第 18 点从 875.91 us 降至 519.29 us，第 20 点从 1.25 ms 降至 907.68 us，第 24 点从 1.91 ms 降至 1.45 ms。

## 四、收获与感悟

卢明逸：通过从 Tiling 到 Cube+Vector Kernel 的完整实现，理解了 Ascend C 中矩阵分块、混合核协作及量化反量化数据流，并通过 CANNJudge 验证了实现的通用性。

王乐天：通过跨 CANN 版本复测和模型接入，认识到单算子精度通过只是网络正确性的必要条件。内部数据格式、Profiler API、模型层调用约定和确定性输出都需要独立验证；实验报告应保留失败证据和环境边界，避免用语义相似替代严格功能判定。

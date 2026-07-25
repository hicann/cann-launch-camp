# 老吴 —— QmmCustom 自定义量化 Matmul 算子开发与 Qwen3-8B 集成实践报告

## 一、结果展示

### 1.1 实验环境与最终状态

- 目标设备：Huawei Developer Space，Ascend910_9362
- CANN：9.0.0
- AscendC 编译架构：`dav-2201`
- PyTorch：2.7.1
- torch-npu：2.7.1.post4
- 模型：Qwen3-8B-W8A8
- 最终 Notebook：32 个单元格，其中 12 个代码单元格已依次执行
- Notebook 错误输出：0
- 最终 Run All 返回码：0
- 运行结束后 NPU Health：OK，无残留 NPU 进程

### 1.2 单算子精度比对结果

测试覆盖 Qwen3-8B 中实际出现的 12 组 `(M,K,N)`，每组分别验证 INT32 和 BF16 两种输出，共 24 项。

- INT32：要求逐元素精确相等；
- BF16：使用 `rtol=0.01, atol=0.01`；
- 权重输入在 NPU 上转换为 FRACTAL_NZ 格式 29；
- 最终结果：`TOTAL=24 PASS=24 FAIL=0 EXECUTED=24`。

|    M |     K |     N | INT32 | BF16 | 最终 bad_count |
| ---: | ----: | ----: | :---: | :--: | -------------: |
|    1 |  4096 |  4096 | PASS  | PASS |              0 |
|    1 |  4096 |  6144 | PASS  | PASS |              0 |
|    1 |  4096 | 24576 | PASS  | PASS |              0 |
|    1 | 12288 |  4096 | PASS  | PASS |              0 |
|   50 |  4096 |  4096 | PASS  | PASS |              0 |
|   50 |  4096 |  6144 | PASS  | PASS |              0 |
|   50 |  4096 | 24576 | PASS  | PASS |              0 |
|   50 | 12288 |  4096 | PASS  | PASS |              0 |
| 4096 |  4096 |  4096 | PASS  | PASS |              0 |
| 4096 |  4096 |  6144 | PASS  | PASS |              0 |
| 4096 |  4096 | 24576 | PASS  | PASS |              0 |
| 4096 | 12288 |  4096 | PASS  | PASS |              0 |

### 1.3 单算子性能测试结果

以下数据来自最终 Notebook Run All 中的 `torch_npu.profiler`。每个 Shape/模式先 warmup 一次，再采集一次。BF16 路径由 Cube 和独立 Vector Kernel 组成，因此 `BF16 Total = BF16 Cube + BF16 Vector`。

|    M |     K |     N | INT32 Cube (us) | BF16 Cube (us) | BF16 Vector (us) | BF16 Total (us) |
| ---: | ----: | ----: | --------------: | -------------: | ---------------: | --------------: |
|    1 |  4096 |  4096 |           21.96 |          19.24 |             6.04 |           25.28 |
|    1 |  4096 |  6144 |           27.10 |          25.92 |             7.88 |           33.80 |
|    1 |  4096 | 24576 |           93.94 |          86.46 |             7.90 |           94.36 |
|    1 | 12288 |  4096 |           35.52 |          34.82 |             6.16 |           40.98 |
|   50 |  4096 |  4096 |           21.64 |          19.92 |            11.70 |           31.62 |
|   50 |  4096 |  6144 |           25.82 |          27.96 |            11.42 |           39.38 |
|   50 |  4096 | 24576 |           97.26 |          93.56 |            20.72 |          114.28 |
|   50 | 12288 |  4096 |           63.30 |          61.96 |            11.24 |           73.20 |
| 4096 |  4096 |  4096 |          383.26 |         385.68 |           187.88 |          573.56 |
| 4096 |  4096 |  6144 |          597.72 |         596.94 |           221.52 |          818.46 |
| 4096 |  4096 | 24576 |         2389.86 |        2382.14 |           645.36 |         3027.50 |
| 4096 | 12288 |  4096 |         1231.68 |        1217.46 |           190.14 |         1407.60 |

12 个 Shape 的本轮采样合计：

- INT32 Cube：4989.06 us；
- BF16 Total：6280.02 us。

这些数据是一次 warmup 后的单次采样，用于复现实验流程，不将其表述为稳定统计中位数。

### 1.4 CANNJudge 稳定版性能优化结果

在标准 CANNJudge 工程中，最终探索版保持 24/24 正确性与 memcheck 通过。完整 24 项 Profiling 总耗时从 v19 的 `12775.62 us` 降至 v32 的 `11239.06 us`，整体提升 `12.027283%`。

对 8 个主要耗时 Shape 进行同机交错顺序 A/B 测试：

- main：11468.04 us；
- explore：9788.90 us；
- 8/8 Shape 全胜；
- 合计提升：14.641909%。

Torch 直调 `.asc` 版本与 CANNJudge 标准算子交付形态不同。前者需要处理 Torch Tensor 校验、动态库注册、直调 workspace 和模型子进程加载，因此本报告不把两个工程的绝对耗时直接混为同一评分口径。

### 1.5 算子接入 Qwen3-8B 的功能结果

最终模型运行使用同一份 Qwen3-8B-W8A8 权重、单卡、batch size 1、eager 模式和 256 个 Decode step。

执行结果：

- 模型推理返回码：0；
- 37/37 权重分片加载完成；
- Warm-up、Prefill 和 Decode 均完成；
- QmmCustom 总调用次数：37296；
- INT32 调用：9324；
- BF16 调用：27972；
- 最终普通推理 Decode 平均耗时：44.28 ms，低于 100 ms；
- Traceback：0；
- AICORE_EXCEPTION：0。

原生 `npu_quant_matmul` 与 QmmCustom 均生成了对 Attention 机制的连贯解释，均覆盖 attention、weighted sum、query、key、value、softmax、output 和 dimension 等关键概念。两份文本语义一致，但并非逐字符一致。量化和反量化中的细小舍入差异会使自回归生成在早期 Token 分叉，因此没有把该结果描述为 exact match。

### 1.6 算子接入模型性能测试结果

最终 Run All 的 QmmCustom Profiler 结果：

| Phase   | 逻辑 Qmm 调用 | Cube Total (us) | BF16 Vector 调用 | Vector Total (us) | Qmm Total (us) |
| ------- | ------------: | --------------: | ---------------: | ----------------: | -------------: |
| Prefill |           144 |         8610.58 |              108 |           1455.40 |       10065.98 |
| Decode  |           432 |        21098.04 |              324 |           2102.84 |       23200.88 |

同配置的原生与自定义公平对比曾独立串行采集，原生 Qmm 总耗时为：

- Prefill：6576.22 us；
- Decode：19338.98 us。

按最终复跑的自定义采样计算，Custom / Native 分别约为：

- Prefill：1.530664×；
- Decode：1.199695×。

当前 Torch 直调版本以稳定正确为优先。BF16 路径先由 Cube Kernel 把 INT32 中间结果写入 GM workspace，再由独立 Vector Kernel 读取并完成反量化，因此存在第二次 Kernel 发射和 INT32 中间结果的 GM 往返，性能低于成熟的原生融合算子。该负结果由真实 CSV 支撑并如实保留。

## 二、方案说明

### 2.1 TilingData 设计

`QmmCustomTilingData` 保存以下信息：

- `M/N/K` 与是否包含 per-token scale；
- `usedCoreNum`、M/N 多核网格和每核逻辑范围；
- Vector 分段列宽；
- 系统 workspace、INT32 中间结果偏移和总 workspace；
- CANN Matmul 高阶 API 使用的 `TCubeTiling`。

Host 侧优先调用 `GetMatmulTiling` 获取合法自动 Tiling。当自动结果不满足 FRACTAL_NZ 对齐、核数、单核范围或 workspace 约束时，回退到 N 方向对齐的安全切分。FRACTAL_NZ 的 N 方向切分保持 32 对齐，避免物理布局偏移错误。

### 2.2 INT32 Kernel

INT32 路径完成：

```text
INT8 ND × INT8 FRACTAL_NZ → INT32 ND
```

每个逻辑核根据 TilingData 计算 M/N tile 和 GM 偏移，使用 Matmul 高阶 API 完成 Cube 计算并直接写入输出。INT32 测试要求逐元素精确相等。

### 2.3 BF16 Kernel

BF16 路径计算：

```text
BF16 = INT32 Matmul Result × perChannelScale × perTokenScale
```

最终实现采用同一 NPU Stream 上的两阶段发射：

1. Cube Kernel 把 INT32 中间结果写入用户 workspace；
2. 独立 Vector Kernel 将 INT32 转为 FLOAT32，乘 per-channel 与 per-token scale，再转换为 BF16 写回输出。

Stream 的顺序语义保证两个阶段的全局可见性，不再依赖开发者手工管理 AIC/AIV CrossCore flag。

### 2.4 Torch 与模型接入

Torch 侧注册接口为：

```python
torch.ops.ascendc_ops.qmm_custom(x1, x2, scale, pertoken_scale)
```

接入模型时替换 `CompressedTensorsW8A8Int8LinearMethod.apply_weights()` 中的 `torch_npu.npu_quant_matmul`。模型推理由 `infer.sh` 启动子进程，因此不能只在 Notebook 父进程加载动态库。自定义实现通过 `QMM_CUSTOM_SO_PATH` 在 recipes 子进程第一次调用量化 Linear 时加载 SO，并验证 `torch.ops.ascendc_ops.qmm_custom` 已注册。

接口同时检查：

- bias 必须为空；
- weight scale 必须为 FLOAT32；
- INT32 路径不传 per-token scale；
- BF16 路径传入动态量化产生的 per-token scale。

## 三、问题解决与优化策略

### 3.1 Cube 与 Vector 同 Kernel 同步异常

早期实现曾在大 Shape BF16 路径触发 AIV MTE DDR 地址越界。结合故障 PC、重复真机结果和官方文档，确认 Matmul 高阶 API 内部会使用 CrossCore，同一 Kernel 再使用手工 flag 可能冲突；部分实验 flagId 也超出 Atlas A2 合法范围。最终改为 Stream 顺序的 Cube + 独立 Vector 两阶段实现，重新通过 24/24 与完整模型门禁。

### 3.2 系统 workspace 与用户 workspace

直调 Kernel 需要同时满足 Matmul 系统 workspace 和 BF16 INT32 中间结果空间。最终实现显式计算两部分大小及偏移，并保证区域不重叠，避免把编译成功或 Kernel 提交成功误判为数值正确。

### 3.3 动态库在模型子进程中未注册

Notebook 父进程的 `torch.ops.load_library()` 不会自动传播到 `infer.sh` 创建的 Python 子进程。修复后由模型量化模块读取 `QMM_CUSTOM_SO_PATH` 并在子进程内加载，只加载一次，同时记录调用计数作为真实接入证据。

### 3.4 Notebook Run All 环境问题

最终整本执行发现并修复了以下工程问题：

1. Notebook 原 kernelspec 指向不存在的 `myenv`；改为已成功运行 Qwen3-8B 的 `qwen3-recipes` 内核；
2. 模型 helper 所需的 `qwen3_npu_adaptation.py` 未包含在精简远端副本中；补齐公共模块；
3. 在 Notebook 进程直接加载自定义 SO 前需先导入 `torch_npu`，保证 `libtorch_npu.so` 已加载；
4. Profiling 脚本原本未生成 Notebook 读取的 `qmm_operator_duration_summary.csv`；新增 Shape 级汇总，并处理 Profiler 偶尔把 BF16 Cube `Input Shapes` 记录为 `N/A` 的情况。汇总同时依据固定发射顺序绑定记录，在存在 Shape 元数据时执行反向校验。

修复后整本 Notebook 返回码为 0，12 个代码单元格均有执行序号，错误输出为 0。

### 3.5 AI 工具使用说明

实践过程中使用了 AI 辅助完成官方文档定位、错误日志归纳、测试脚本补全、Profiler CSV 解析和 Notebook 执行链审计。AI 建议不直接作为正确性依据，所有修改都需要通过 clean build、24 项数值测试、完整模型运行、Profiler CSV 和 NPU 健康检查验证。

AI 辅助也曾引入过不准确判断，例如最初把远端缺失的顶层公共模块误判为相对导入问题。通过重新核对本地完整目录结构，撤销了不必要的导入改动，只补齐缺失文件。这说明 AI 适合加速搜索和提出假设，但工程结论必须以源码、环境和真实运行结果为准。

### 3.6 后续性能优化方向

当前正确性优先版的主要瓶颈是 BF16 两阶段发射和 INT32 GM 往返。后续可继续研究：

1. 使用官方支持的 Cube/Vector 融合范式，避免与 Matmul 内部 flag 冲突；
2. 将反量化迁移到受支持的 Fixpipe 或融合输出能力；
3. 为 M=1 和 M=50 设计短 M 专用 Tiling；
4. 通过 L1/L0 数据复用减少大 Shape 的重复搬运；
5. 每次优化后重新执行 24 项、memcheck、完整模型和 Prefill/Decode Profile 门禁。

## 四、收获与感悟

本次实践打通了从 Tiling、AscendC Kernel、Torch 自定义算子注册到 Qwen3-8B W8A8 模型接入的完整链路。最大的体会是“编译成功”“Kernel 成功提交”“单个 Shape 通过”“24 项通过”和“完整模型稳定运行”是不同层级的验收门槛，任何一层都不能替代下一层。

性能优化也不能只依赖理论上的访存减少。缓存生命周期、Kernel 发射次数、同步方式和真实 Shape 分布都会改变最终收益。对稳定提交版而言，正确性与可复现证据优先；探索性优化应放在独立分支，通过同机 A/B 和完整矩阵评估后再决定是否保留。

## 五、可复核证据

最终 Notebook 已保留全部执行输出。额外证据位于：

```text
integration/results/final_run/
├── executed_result.ipynb
├── run-all-v3.log
├── run-all-v3.rc
├── qmm_operator_duration_summary.csv
├── qmm_operator_duration_summary.json
├── qmm_profile_summary.json
├── custom_prefill_kernel_details.csv
└── custom_decode_kernel_details.csv
```

阶段报告位于：

```text
integration/reports/
├── stage01_qmm_operator_validation.md
├── stage02_w8a8_quantization.md
├── stage03a_native_w8a8_baseline.md
├── stage03b_qmm_custom_model_integration.md
└── stage04_model_profiling_comparison.md
```

所有远端回传文件均进行了双端 SHA-256 校验。
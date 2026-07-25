# 思瑞three Qwen3-8B QmmCustom 团队实践报告

## 一、团队信息与贡献说明

### 1.1 团队基本信息

- 团队标识：`思瑞three`
- 团队成员：张瑞瑞、左思琪
- CANNJudge 提交账号：张瑞瑞
- CANNJudge 最终提交 ID：`106797`
- CANNJudge 结果：`24/24 Pass`

### 1.2 团队成员分工与贡献

| 姓名 | GitCode 账号 | 负责内容 | 实际贡献说明 | 对应 commit hash |
| --- | --- | --- | --- | --- |
| 张瑞瑞 | `zhangruirui` | Tiling 与 Kernel 开发、CANNJudge 验证、Qwen3-8B 模型接入与最终集成 | 设计 `QmmCustomTilingData` 和 Tiling；实现 INT8×INT8→INT32 Cube 路径以及带 per-channel/per-token scale 的 BF16 反量化路径；完成 CANNJudge 验证；接力处理 Qwen3-8B 模型接入、异步调度诊断和最终交付整理 | `1dbc842`、`419f6ac` |
| 左思琪 | `ZuoSiqi` | CANNLab 算子编译、单算子功能验证与性能采集 | 在 CANNLab 完成自定义算子编译、课程 12 组 shape 的 INT32/BF16 功能验证及 Profiler 性能采集，并将真实输出同步至团队 Notebook | `2d52ed0`、`7a1113f` |

### 1.3 团队协作说明

团队采用分阶段接力方式协作。张瑞瑞先在个人 Fork 和本人分支完成 Tiling、Cube-only Kernel、带反量化的 Cube+Vector 路径及 CANNJudge 验证；左思琪在自己的 Fork 和分支中完成 CANNLab 编译、单算子功能测试和性能采集。模型接入初期由左思琪推进，受阻后团队明确调整为张瑞瑞接力处理。最终汇总保留双方原始作者身份，没有 squash 成单一提交。

团队统一使用 `思瑞three_result` 目录提交以下三项成果：

```text
思瑞three_result.ipynb
思瑞three_qmm_custom.asc
思瑞three_report.md
```

## 二、结果展示

### 2.1 编译结果

最终 Notebook 在 CANNLab A3 环境中完成算子编译，保存的构建输出为：

```text
[ 50%] Building ASC object CMakeFiles/ascendc_ops.dir/qmm_custom.asc.o
[100%] Linking ASC shared library libascendc_ops.so
[100%] Built target ascendc_ops
```

编译生成文件为：

```text
src/op_custom/qmm_custom/build/libascendc_ops.so
```

### 2.2 单算子精度比对结果

课程测试覆盖 Qwen3-8B 推理中实际出现的 12 组 `(M,K,N)` 规格。每组分别测试：

1. 不传入 `pertoken_scale` 的 INT32 输出路径；
2. 传入 `pertoken_scale` 的 BF16 反量化路径。

最终 Notebook 保存的汇总结果为：

```text
INT32: 12/12 通过
BF16: 12/12 通过
```

详细结果如下：

| M | K | N | INT32 | BF16 |
| ---: | ---: | ---: | :---: | :---: |
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

CANNJudge 最终提交 ID `106797` 的结果为 `24/24 Pass`。Notebook 采用 `<<<>>>` 核函数直调形式，CANNJudge 使用注册算子工程形式；两者工程封装不同，但核心计算逻辑一致。正式单算子功能评分以 CANNJudge 结果为准。

### 2.3 单算子性能测试结果

Notebook 使用 `torch_npu.profiler` 采集 QmmCustom 的 Kernel Duration，并从实际生成的 `kernel_details.csv` 中提取数据，单位为微秒。

| 规格 `(M,K,N)` | INT32 Duration (μs) | BF16 Duration (μs) |
| --- | ---: | ---: |
| 1,4096,4096 | 19.100 | 22.339 |
| 1,4096,6144 | 26.159 | 29.919 |
| 1,4096,24576 | 99.378 | 102.158 |
| 1,12288,4096 | 39.059 | 43.119 |
| 50,4096,4096 | 19.639 | 26.839 |
| 50,4096,6144 | 27.639 | 35.539 |
| 50,4096,24576 | 105.857 | 113.618 |
| 50,12288,4096 | 50.559 | 55.259 |
| 4096,4096,4096 | 438.811 | 531.209 |
| 4096,4096,6144 | 654.646 | 849.183 |
| 4096,4096,24576 | 3100.218 | 3870.062 |
| 4096,12288,4096 | 1475.490 | 1564.909 |

结果表明：

- `M=1` 和 `M=50` 的小批量场景主要在几十微秒量级；
- 随 M、N 增大，Cube 计算量和 INT32 中间结果规模同步增大；
- BF16 路径通常慢于 INT32，因为它增加了 INT32 结果写回、两类 scale 读取、Vector 乘法和 BF16 转换；
- 大 M、大 N 场景仍有进一步优化空间，重点是 Cube 分块、负载均衡和减少 GM 往返。

Profiler 运行中出现过：

```text
Incorrect schedule: Stop profiler while current state is RECORD
```

随后日志显示 Profiler 成功完成解析并生成 `kernel_details.csv`。因此表中数据来自真实采集文件，但后续仍可通过规范 schedule 重复采样，提高性能统计的稳定性。

### 2.4 算子接入模型验证边界

模型接入点位于：

```text
CompressedTensorsW8A8Int8LinearMethod.apply_weights
```

目标是将：

```python
torch_npu.npu_quant_matmul(...)
```

替换为：

```python
torch.ops.ascendc_ops.qmm_custom(
    x,
    layer.weight,
    layer.weight_scale,
    pertoken_scale=x_scale,
)
```

最终 Notebook 已保存模型接入与网络验证的完整执行过程：

1. 6.2 输出确认 `npu_quant_matmul` 已替换为 `qmm_custom`；
2. 6.3 完成模型加载、KV Cache 初始化、warm-up 和 256 token decode；
3. 普通推理平均 decode 耗时为 `59.50 ms`；
4. Profiling 推理平均 decode 耗时为 `54.59 ms`；
5. 两次生成文本均为连续感叹号，不符合课程要求的 attention 语义。

因此，网络性能耗时低于 100 ms，但网络功能不通过，不能宣称完成了正确的模型输出。Notebook 开头出现：

```text
rm: cannot remove '/root/atc_data/': Permission denied
```

后续模型仍然成功启动并完成推理；该行是清理无权限目录时的非致命提示，不改变最终网络功能失败的结论。

网络 Profiling 还保存了 QmmCustom 在真实 prefill/decode 中的分 shape 数据：

| Phase | 输入 shape | 输出类型 | 平均耗时 (μs) | 调用次数 |
| --- | --- | --- | ---: | ---: |
| Prefill | `50,4096;4096,6144;6144;50` | DT_BF16 | 36.180000 | 1 |
| Prefill | `50,4096;4096,4096;4096;50` | DT_BF16 | 8.835643 | 14 |
| Decode | `1,4096;4096,6144;6144;1` | DT_BF16 | 7.716857 | 7 |
| Decode | `1,4096;4096,4096;4096;1` | DT_BF16 | 3.361280 | 107 |
| Decode | `1,12288;12288,4096;4096;1` | DT_BF16 | 2.660000 | 1 |

## 三、方案说明

### 3.1 算子接口与计算公式

算子接口为：

```text
qmm_custom(x1, x2, scale, pertoken_scale=None) -> y
```

输入及输出含义：

- `x1`：INT8、ND 布局，逻辑形状 `[M,K]`；
- `x2`：INT8、FRACTAL_NZ 布局，逻辑形状 `[K,N]`；
- `scale`：FP32 per-channel scale，长度为 `N`；
- `pertoken_scale`：可选 FP32 per-token scale，长度为 `M`；
- 无 per-token scale 时，输出 INT32 `[M,N]`；
- 有 per-token scale 时，输出 BF16 `[M,N]`。

INT32 路径为：

$$
C_{m,n}=\sum_{k=0}^{K-1}X_{m,k}W_{k,n}
$$

BF16 路径为：

$$
Y_{m,n}=\operatorname{BF16}
\left(
C_{m,n}\cdot
\left(s^{channel}_{n}\cdot s^{token}_{m}\right)
\right)
$$

实现中先合并 per-channel 与 per-token scale，再乘 INT32 累加结果，以对齐原生量化算子的 FP32 运算与 BF16 舍入顺序。

### 3.2 TilingData 与 Tiling 设计

`QmmCustomTilingData` 保存：

- `TCubeTiling cubeTilingData`；
- 是否存在 per-token scale；
- 系统、用户和总 workspace 大小；
- M、N、K；
- Vector tile 长度；
- Vector 核数。

Host 侧通过 `PlatformAscendCManager` 获取 AIC/AIV 核数和 Matmul 高阶 API 所需 workspace。Cube 部分使用 `MultiCoreMatmulTiling` 生成分块参数，当前最多选择 16 个逻辑组，以覆盖课程给出的 M/N/K 组合并保持 FRACTAL_NZ 的 N 方向切分边界对齐。

BF16 路径的 INT32 中间结果使用单独的 PyTorch Tensor 保存；Matmul KFC 使用系统 workspace。二者分离，避免把普通数据区和系统 KFC workspace 混在同一地址中。

### 3.3 Kernel 数据流

当前源码的活动执行路径分为两个同流顺序 Kernel：

```text
x1 INT8 ND ─┐
            ├─ MIX(1,2) Cube Kernel ─→ INT32 intermediate
x2 INT8 NZ ─┘                               │
                                           ▼
channel scale ───────────────────────┐  AIV dequant Kernel ─→ BF16 y
token scale ─────────────────────────┘
```

#### Cube Kernel

- 使用 CANN Matmul 高阶 API；
- 完成 INT8×INT8→INT32；
- 按 Tiling 计算各逻辑 Cube 的 M/N 分块和 GM 起点；
- 权重按 FRACTAL_NZ 物理布局计算 N 方向偏移。

#### Vector 反量化 Kernel

- 按行分配到 AIV；
- 每次处理最多 2048 个 N 元素；
- 完成 INT32→FP32；
- 计算 `channel_scale × token_scale`；
- 计算 `FP32 accumulator × combined_scale`；
- 采用 `CAST_RINT` 转为 BF16；
- 显式维护 MTE2、Vector、MTE3 的依赖。

两个 Kernel 在同一 NPU stream 上顺序启动。Host 返回前同步当前 stream，保证 INT32 临时 Tensor 在 Vector Kernel 完成之前不会被分配器复用。

## 四、问题解决与优化策略

### 4.1 FRACTAL_NZ 地址计算

早期把权重按普通 ND 矩阵理解，导致 N 分块后的 B 矩阵起点错误。最终明确区分逻辑 shape 与物理布局，保证 N 分块 16 对齐，并按 NZ 规则计算权重地址。

### 4.2 workspace 与临时结果生命周期

Matmul KFC 需要系统 workspace，BF16 路径还需要完整 INT32 中间结果。将二者混用会导致越界、旧地址读取或异步复用。最终使用独立 Tensor 保存 INT32 中间结果，并在 Host 返回前同步 stream。

### 4.3 BF16 舍入顺序

调试中发现极少数 BF16 元素可能因 FP32 乘法结合顺序不同产生一个 ULP 的差异。团队比较了先乘 channel、先乘 token 和先合并 scale 三种顺序，最终采用：

```text
INT32 × (channel_scale × token_scale)
```

### 4.4 测试口径恢复

AI 调试阶段曾引入逐元素 exact 判定，使符合课程 BF16 容差的结果被错误标记为失败。团队重新核对课程原始 Notebook，恢复官方 allclose 口径；exact mismatch 仅作为额外诊断信息，不替代正式判定标准。

### 4.5 网络级异步问题

单算子 24 条用例通过后，网络仍出现连续感叹号。设备级同步实验曾使短文本恢复正常开头，说明量化模型和 YAML 不是唯一疑点，问题与 KFC Matmul、临时 Tensor 生命周期及 Cube/Vector 异步完成边界有关。

团队对比过融合 CrossCore 协同和同流顺序 Kernel 两种方案，并保留每次实验的 Git 提交和真机输出。当前 Notebook 证明最新版仍能通过 24 条单算子测试，但由于网络输出未达到语义要求，不能宣称完整网络接入成功。

### 4.6 性能优化方向

当前优化遵循“正确性优先”：

1. 先保证 24 条单算子回归全部通过；
2. 使用 Profiler 定位大 M、大 N 场景；
3. 优化 Cube 分块和核间负载均衡；
4. 研究减少 INT32 中间结果 GM 往返的方式；
5. 在不改变 BF16 舍入语义的前提下优化 Vector 流水和双缓冲；
6. 每次 Tiling 或 Kernel 变化后重新执行完整 24 条回归。

### 4.7 AI 辅助说明

团队使用 AI 辅助：

- 阅读编译、单算子和模型推理日志；
- 分析 FRACTAL_NZ 地址、workspace 和异步同步问题；
- 生成精度诊断脚本；
- 比较不同 BF16 缩放顺序；
- 检查 Notebook、报告和 Git 变更范围。

AI 也曾引入新的问题，包括过严的 exact 测试、未经充分验证的核数调整、错误的同步插入位置等。团队通过以下方式约束 AI：

- 以课程原始 Notebook 为测试口径；
- 检查 Git diff，防止误改测试；
- 每次 Kernel/Tiling 变化后执行 24 条真机回归；
- 只把真实保存的 CANNLab、CANNJudge 和模型日志写入报告；
- 对已执行但未通过的网络结果明确标注限制。

## 五、团队成员收获与感悟

### 5.1 张瑞瑞

本次实践让我把量化公式、Tiling、Cube、Vector、FRACTAL_NZ、workspace 和模型推理串成了一条完整数据流。最重要的认识是：单算子通过并不等于网络一定通过。随机输入的一次调用难以覆盖模型连续调用中的 stream、临时 Tensor 和 KFC 调度问题。算子优化必须分层验证，任何看似合理的修改都要经过完整回归。

在使用 AI 时，我也认识到 AI 更适合辅助提出假设和生成诊断工具，而不能替代真机结果。错误建议只有通过原始测试、Git diff 和真实 NPU 输出才能被及时发现。

### 5.2 左思琪

通过 CANNLab 编译、12 组 shape 功能测试和 Profiler 采集，我进一步理解了自定义算子从源码到共享库、再到 PyTorch 接口调用的完整过程。性能优化不能只看单个小 shape，需要同时观察 decode、小规模 prefill 和长序列 prefill；同时，性能数据只有在功能正确的前提下才有意义。

团队异步协作也要求明确版本、路径和证据来源。把编译输出、功能汇总、性能 CSV 和 commit 一起交接，比只说“已经跑过”更可靠。

## 六、最终验证结论

根据提交的 `思瑞three_result.ipynb`：

| 验证项 | 结论 | 证据 |
| --- | --- | --- |
| Notebook 结构 | 有效 | nbformat 4.5，共 30 个单元 |
| 自定义算子编译 | 通过 | `[100%] Built target ascendc_ops` |
| 单算子 INT32 | 通过 | 12/12 |
| 单算子 BF16 | 通过 | 12/12 |
| CANNJudge | 通过 | ID `106797`，24/24 Pass |
| 单算子性能采集 | 已完成 | Notebook 中保存 12 组 Profiler 汇总 |
| 模型替换执行 | 已完成 | 6.2 输出确认使用 `qmm_custom`，6.3 完成 256 token 推理 |
| 网络功能 | 未通过 | 保存输出为连续 `!`，不符合 attention 语义 |
| 网络平均耗时 | 低于 100 ms | 普通推理 59.50 ms；Profiling 推理 54.59 ms |
| 网络 QmmCustom Profiling | 已完成 | Notebook 保存 prefill/decode 分 shape 平均耗时和调用次数 |

本报告以最终 Notebook 中的真实输出为证据边界，不补造任何模型结果。

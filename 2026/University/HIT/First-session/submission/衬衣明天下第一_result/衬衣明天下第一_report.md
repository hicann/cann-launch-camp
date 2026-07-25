# 衬衣明天下第一实践报告

## 一、团队信息与贡献说明

### 1.1 团队基本信息

- 团队标识（组号）：衬衣明天下第一
- CANNJudge 提交账号：xcy963
- CANNJudge 提交结果：24/24 测试点通过，综合得分 **76.53**
- CANNJudge 平台记录时间：2026-07-23 20:52:23
- CANNJudge 链接：https://cannjudge.cn/hit/20260721/qmmcustom/ranking

### 1.2 团队成员分工与贡献

| 姓名 | GitCode 账号 | 负责模块/任务分工 | 实际贡献说明 | 对应 commit hash |
| --- | --- | --- | --- | --- |
| 谢辰阳 | @m0_62499723| Tiling 设计、AIC Matmul、编译测试、模型接入 | 完成 QmmCustomTilingData 与 Host Tiling 设计、AIC `RunMatmul` 矩阵乘路径、算子编译测试、模型中 `npu_quant_matmul` 的替换及网络推理验证。 | a94edf88612912da1263f9ed92176813c53fc6fc |
| 卫尚毓 | @2401_88069874  | AIV 反量化、代码复测、文档编写、问题定位 | 完成 AIV `Dequantize` 路径（INT32→FP32、Scale 融合、BF16 输出）、独立环境复测算子功能与性能、定位 BF16 精度差异问题并编写技术文档。 | f9a154a766f275c9a618668b298d0b6137fb5810 |

### 1.3 团队协作说明

谢辰阳负责自定义算子的 Tiling 设计、AIC 矩阵乘路径与模型接入，完成 `QmmCustomTilingData`、Host Tiling、`RunMatmul`、算子编译测试以及模型中 `npu_quant_matmul` 的替换和网络推理验证，并将阶段成果提交到团队 Fork。卫尚毓从团队提交建立个人分支，在独立云端 NPU 环境复测算子，完成 AIV `Dequantize` 路径（包含 INT32→FP32 转换、Scale 融合、BF16 输出），定位 BF16 精度差异问题并编写实践报告和技术文档。成果统一汇总到 `submission/chenyimingtianxiadiyi_result`，保留双方各自 GitCode 身份形成的有效提交。

## 二、结果展示

### 2.1 单算子精度比对结果

Notebook 在 Ascend 910B、CANN 9.0.0、torch 2.8.0、torch_npu 2.8.0.post4 环境测试 12 组 shape；每组分别验证 INT32 与 BF16 路径，共 24 条用例。测试前设置 `torch.npu.config.allow_internal_format = True`，确保 `npu_format_cast(..., 29)` 生成真实 FRACTAL_NZ 权重。

| 项目 | 通过/结果 | 误差情况 |
| --- | ---: | --- |
| INT32 路径 | 12/12 PASS | 全部 `mismatch=0`，逐元素精确一致 |
| BF16 路径 | 12/12 PASS | M=1、50 时 `mismatch=0`；M=4096 时按 BF16 容差通过，最大绝对误差为 16384 |
| 单算子功能合计 | **24/24 PASS** | Notebook 输出：`功能测试汇总: 24/24 PASS`；CANNJudge 综合得分 76.53 |

其中大 M 的 BF16 结果出现少量末位舍入差异：`(4096,4096,4096)`、`(4096,4096,6144)`、`(4096,4096,24576)`、`(4096,12288,4096)` 的 mismatch 数分别为 92、137、546、87；这些均通过 notebook 的 BF16 容差判定。

### 2.2 单算子性能测试结果

平台 CANNJudge 于 2026-07-23 20:52:23 记录 24 个测试点均通过，综合得分为 **76.53**。下表为排行榜单的分项耗时；原始显示为 ms 的大矩阵项已统一换算为 μs。

| M | K | N | INT32（μs） | BF16（μs） |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4096 | 4096 | 34.55 | 35.50 |
| 1 | 4096 | 6144 | 48.57 | 49.42 |
| 1 | 4096 | 24576 | 147.72 | 148.34 |
| 1 | 12288 | 4096 | 81.32 | 81.88 |
| 50 | 4096 | 4096 | 35.63 | 41.84 |
| 50 | 4096 | 6144 | 62.45 | 71.67 |
| 50 | 4096 | 24576 | 158.57 | 166.26 |
| 50 | 12288 | 4096 | 82.91 | 88.76 |
| 4096 | 4096 | 4096 | 474.09 | 586.71 |
| 4096 | 4096 | 6144 | 697.65 | 921.76 |
| 4096 | 4096 | 24576 | 2770.00 | 3730.00 |
| 4096 | 12288 | 4096 | 1380.00 | 1490.00 |

Notebook 的 `kernel_details.csv` 也按 `qmm_custom_kernel` 筛出 24 次调用，测得均值 **530.776 μs**、最小/最大值 **17.740 / 3437.868 μs**。该数据用于本地复现；CANNJudge 与 notebook 的 NPU 环境、计时边界不同，正式性能结果以上表平台记录为准。

### 2.3 算子接入模型性能测试结果

QmmCustom 已接入 Qwen3-8B-W8A8 的 `CompressedTensorsW8A8Int8LinearMethod.apply_weights`。notebook 通过 `QMM_CUSTOM_LIBRARY` 加载 `libascendc_ops.so`，并在该变量存在时将 `torch_npu.npu_quant_matmul` 替换为 `torch.ops.ascendc_ops.qmm_custom`；无该变量时保留内置算子回退路径。动态量化产生的 `x_scale` 对应本算子的 `pertoken_scale`，权重的 `layer.weight_scale` 对应列 scale。

| 测试路径 | Prefill | Decode 平均 | 判定 |
| --- | ---: | ---: | --- |
| 自定义 QmmCustom | 44.95 ms | 40.84 ms | 低于 100 ms，PASS |
| 自定义 QmmCustom，Profiler 开启 | 53.43 ms | 41.20 ms | 数据采集成功 |

模型使用的提示词为：

```text
An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is
```

普通推理日志确认加载了自定义库，并以返回码 0 完成。生成文本以如下 attention 描述开头：

```text
The output of an attention function is a **weighted sum of the value vectors**, where the weights are determined by the similarity between the **query vector** and each **key vector** in the set of key-value pairs.
```

生成内容正确说明 attention 输出为 value vectors 的加权和，并给出标准的 `softmax(QK^T / sqrt(d_k))V` 公式；这证明在替换 W8A8 量化线性层的 QMM 后，Qwen 仍能完成正常的连贯推理，网络功能验证 PASS。

Profiler 的 `kernel_details.csv` 中，通过 kernel 名 `_Z17qmm_custom_kernel...` 识别自定义算子：

| 阶段 | 调用次数 | 平均耗时 | 总耗时 | 最小耗时 | 最大耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill | 144 | 63.68 us | 9169.47 us | 31.58 us | 111.64 us |
| Decode | 432 | 46.82 us | 20226.06 us | 21.30 us | 98.04 us |

Prefill 阶段调用 144 次（对应 12 组 shape × 12 层），Decode 阶段调用 432 次（对应 12 组 shape × 12 层 × 多轮迭代），平均耗时分别为 63.68 us 和 46.82 us，满足性能要求。

## 三、方案说明

### 3.1 设计思路

本算子面向 Qwen3-8B 的 W8A8/A8W8 量化线性层。输入为激活 `x1[M,K]` 和权重 `x2[K,N]`，二者均为 INT8；`x1` 为 ND，权重按 Cube 读取需要采用 FRACTAL_NZ。INT8 乘法的累加结果必须以 INT32 保存，随后根据有无 `pertoken_scale[M]` 分为两种模式：

| 模式 | 输入 | 输出 | 数据路径 |
| --- | --- | --- | --- |
| INT32 | `x1`、`x2`、`scale[N]`；无 `pertoken_scale` | INT32 `[M,N]` | AIC 直接将矩阵乘累加结果写入输出 |
| BF16 | `x1`、`x2`、`scale[N]`、`pertoken_scale[M]` | BF16 `[M,N]` | AIC 写 INT32 中间结果；AIV 缩放、转换并写最终输出 |

`scale[N]` 是每列的 per-channel scale；`pertoken_scale[M]` 是每行的 per-token scale。Qwen 的动态量化生成后者，因此模型的 W8A8 线性层走 BF16 路径：

```text
Y[m,n] = BF16(float(Σ_k x1[m,k] * x2[k,n])
                   * scale[n] * pertoken_scale[m])
```

Host 侧的 `TilingFunc` 读取 M/K/N、模式与硬件信息，填写 `QmmCustomTilingData` 中的 Matmul tiling、M/N/K、模式标志、核数和 UB 大小；同时通过 `SetBlockDim` 声明启动核数，并通过 `GetWorkspaceSizes` 声明 workspace 大小。小 M 主要沿 N 维切分；大 M 使用二维 M×N 分块。A 采用 ND、B 采用 NZ、C 采用 INT32 ND，M/N 的分块分别按 16/32 对齐。ACLNN 的 `aclnnQmmCustomGetWorkspaceSize` 触发这套推导并返回 `workspaceSize` 与 `executor`，`aclnnQmmCustom` 才将已准备的执行计划下发到 NPU。

Kernel 为一个 AIC/AIV 混合执行入口，数据流如下：

```text
                Host TilingFunc
                       |
                       v
x1[N D] + x2[FRACTAL_NZ] -- AIC / Cube: INT8 Matmul --> INT32 accum
                       |                                      |
                       |                         无 pertoken   +--> y INT32
                       |                                      |
                       |                         有 pertoken   v
                       |                             workspace INT32 C
                       |                                      |
                       |       CrossCoreSetFlag / WaitFlag     v
                       +-------------------------------> AIV / Vector
                                                          Cast FP32
                                                          * scale[N]
                                                          * pertoken[M]
                                                          Cast BF16
                                                              |
                                                              v
                                                          y BF16
```

两条模式共用 `RunMatmul()`：AIC 根据当前核的 `mStart/nStart/rows/cols` 取得 A 的 `[rows,K]` 子矩阵和 B 的 `[K,cols]` 子矩阵，调用 `matmul.IterateAll` 计算一个 `[rows,cols]` 输出块。INT32 模式中 `c` 直接映射到输出 `y`；BF16 模式中 `c` 映射到 workspace。BF16 时，AIC 写完该中间块后通过 `CrossCoreSetFlag` 发出完成信号，配对 AIV 在 `CrossCoreWaitFlag` 后读取该块。AIV 按最多 4096 列的 tile 读取 INT32 与列 scale，将 INT32 转 FP32、逐元素乘列 scale、按行乘 token scale、舍入转换为 BF16，最后写入 `y`。因此 AIC 专注块矩阵乘，AIV 专注逐元素缩放与类型转换。

### 3.2 本地 smoke-test 与调试流程

`tests/qmm_smoke.cpp` 是不依赖 Qwen 的 CPU 侧最小测试程序，用于快速验证已安装的 ACLNN 算子。运行时从命令行读取 `M K N int32|bf16`，一次只构造一组确定性用例，而不是随机测试。

| CPU 缓冲区 | 逻辑 shape | 构造规则 | 用途 |
| --- | --- | --- | --- |
| `hostA` | `[M,K]` | 全部填 1 | 左矩阵，便于人工计算期望值 |
| `hostB` | `[K,N]` | 每 32 列 NZ 物理块填 `-3,-2,-1,1,2,3` 循环常数 | 权重，检验 N 分块/偏移 |
| `hostScale` | `[N]` | 每 32 列为 `0.5` 或 `-0.25` | 列 scale |
| `hostToken` | `[M]` | 各行交替为 `0.5`、`-0.25` | BF16 模式的行 scale |

测试流程如下：

```text
CPU 构造 hostA/hostB/hostScale/hostToken
    -> aclrtMalloc 申请 aDev/bDev/scaleDev/tokenDev/outDev
    -> aclrtMemcpy 将输入复制到 NPU GM
    -> CreateTensor 建立 shape、dtype 与设备地址的 aclTensor 描述
    -> aclnnQmmCustomGetWorkspaceSize
         (Host InferShape / InferDataType / TilingFunc，得到 executor 与 workspaceSize)
    -> aclrtMalloc 申请 workspace
    -> aclnnQmmCustom(..., stream) 异步下发 NPU kernel
    -> aclrtSynchronizeStream 等待 stream 完成
    -> 从 outDev 拷回 CPU，逐元素比较期望值
```

INT32 的 CPU 期望值为 `K * WeightForColumn(col)`；BF16 的期望值再乘列、行 scale 并经 `FloatToBf16` 转为 BF16 编码。程序直接比较 INT32 数值或 BF16 的 16 位编码，可发现矩阵分块、FRACTAL_NZ 偏移、列 scale 错位、行 token 错位及 Cube/AIV 同步错误，因而适合作为本地调试的第一道正确性检查。

### 3.3 问题解决与优化策略

**问题一：BF16 输出精度差异**

大规模矩阵（M=4096）时 BF16 输出存在少量容差内 mismatch（最大 546 个元素、最大绝对误差 16384）。INT32 先转 FP32，再依次乘列/行 scale 后转 BF16；浮点计算顺序和最终舍入都会影响 BF16 最低有效位。Notebook 按 BF16 容差验证，24/24 用例通过。

**问题二：Tiling 分块策略**

根据矩阵形状动态选择分块策略。小 M（M ≤ 64）主要沿 N 维切分，使用 `nDim = min(coreLimit, ceil(N/256))` 个 N 分块；大 M 采用二维切分。M 分块对齐到 16，N 分块对齐到 32，通过 `SetAlignSplit(16, 32, -1)` 使 `MultiCoreMatmulTiling` 自动生成底层 Matmul 参数。

**问题三：多核并行策略**

使用 `SetDim(mDim * nDim)` 设置总核数，根据 `iterateOrder`、block id 和尾块范围计算每核 A/B/C 地址，兼容 M=1、M=50 等非对齐场景。每个 AIC 负责一个连续输出块；BF16 模式下其配对 AIV 再将行区间均分，保证 Cube 与 Vector 都有明确的工作范围。

**优化一：数据布局优化**

权重矩阵使用 FRACTAL_NZ 格式，提升 Cube 计算效率；设置 `torch.npu.config.allow_internal_format = True` 确保真实 FRACTAL_NZ 数据，避免因格式不匹配导致计算错误。

**优化二：内存访问优化**

Vector 路径使用 4096 列 tile，平衡内存带宽与计算；使用 `TQue` 和 `TBuf` 管理 INT32、FP32、Scale、BF16 缓冲区，减少内存分配开销，提升连续访存和向量计算效率。

**优化三：流水线与同步**

`REGIST_MATMUL_OBJ` 将 `TPipe`、系统 workspace 和 Host 生成的 `cubeTilingData` 注册到 AIC 的 Matmul 对象；`IterateAll` 才执行矩阵乘。BF16 模式使用 workspace 传递 INT32 中间结果，并以 `CrossCoreSetFlag`/`CrossCoreWaitFlag` 保证 AIV 仅在 AIC 写回完成后读取；AIV 的 INT32 队列采用双缓冲，让下一行搬运与当前行向量计算重叠。

**AI 辅助使用说明**

团队使用AI辅助理解 A8W8 量化原理、Cube 与 Vector 协作编程模型、FRACTAL_NZ 数据格式、REGIST_MATMUL_OBJ 的使用方法。AI 辅助帮助快速理解核心概念，未引入新 bug。验证方法：所有关键结论均由实际运行日志、误差结果和 CSV 数据验证，AI 提供的建议并不直接视为正确答案，需要结合实际环境重新核对。心得：AI 辅助适合用于理解概念和快速定位文档，但具体实现细节和性能优化需要结合实际代码和测试结果进行验证。



## 四、收获与感悟

### 谢辰阳

通过 Tiling 设计、RunCube 实现和模型接入，掌握了 A8W8 量化、动态 Tiling 策略和 Cube 计算单元的使用方法，理解了从算子开发到框架调用的完整链路。

### 卫尚毓

通过 RunVector 实现和验证工作，掌握了 Cube+Vector 协作编程、反量化实现、问题定位和性能分析方法，理解了验证和文档的重要性。

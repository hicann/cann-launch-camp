# 团队实践报告

## 一、团队信息与贡献说明

### 1.1 团队基本信息

- **团队标识（组号）**：EDG  
- **CANNJudge 提交账号**：吴启贤，cjfu2005
**CANNJudge 提交链接**：https://cannjudge.cn/hit/20260721/qmmcustom/ranking?page=2&size=20
### 1.2 团队成员分工与贡献

| 姓名 | GitCode 账号 | 负责模块/任务分工 | 实际贡献说明 | 对应 commit hash |
| --- | --- | --- | --- | --- |
| 吴启贤 | WQXwqx2004 | Tiling 设计、Cube-only Kernel、Host 侧 Torch 接口与编译 | 完成 `QmmCustomTilingData` 结构体设计、`CalcQmmTiling` 分块计算函数；实现 `QmmCubeBasicKernel` 的 INT8→INT32 纯 Cube 路径；封装 `qmm_custom` Torch 接口并完成算子编译（生成 `libascendc_ops.so`）；修复 Tiling 编译错误并协调模块接口。 | 97e2a1cf1e4f3deceee4553eb28b11591922eed6 |
| 陈杰夫 | cjfu2005 | Pertoken 反量化 Kernel、单算子测试、性能分析与模型接入 | 实现 `QmmPertokenKernel` 的 Cube+Vector 反量化路径（INT32→BF16）；编写并运行 12 组 shape 的 INT32/BF16 精度测试；使用 `torch_npu.profiler` 采集单算子性能数据；将自定义算子接入 Qwen3-8B 模型并验证端到端推理。 | 738cb1eee910e31504fed3bc449690e1a11caede |

### 1.3 团队协作说明

本次实践按照完整算子开发流程进行模块拆分：吴启贤负责 Host 侧 Tiling 和纯 Cube 计算路径，陈杰夫负责 Vector 反量化路径及后续验证工作。双方基于统一的 `QmmCustomTilingData` 结构体约定数据接口，并通过每日沟通同步 Tiling 参数调整。代码汇总时，由吴启贤从成员分支合并，保留独立 commit 历史，随后共同审查和运行全量测试。单算子测试（12 组 shape）和模型推理验证均通过，证明了模块协作的有效性。

---

## 二、结果展示

### 2.1 单算子精度比对结果

我们使用官方测试脚本对 QmmCustom 算子进行了功能验证，覆盖 Qwen3-8B 实际推理中常见的 12 组 (M, K, N) 规格。分别测试了 **INT32 输出模式（无 perTokenScale）** 和 **BF16 输出模式（有 perTokenScale）**，采用 `torch.allclose` 与 PyTorch 参考实现对比。

所有测试用例全部通过，结果如下：

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

>![alt text](image.png)

### 2.2 单算子性能测试结果

使用 `torch_npu.profiler` 采集了各规格下 QmmCustom 算子的执行耗时（单位 μs）。结果如下：

| M | K | N | INT32 Duration(μs) | BF16 Duration(μs) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4096 | 4096 | 141.8 | 2.36 |
| 1 | 4096 | 6144 | 209.1 | 3.10 |
| 1 | 4096 | 24576 | 904.7 | 8.00 |
| 1 | 12288 | 4096 | 410.9 | 2.22 |
| 50 | 4096 | 4096 | 151.0 | 7.10 |
| 50 | 4096 | 6144 | 219.0 | 7.78 |
| 50 | 4096 | 24576 | 926.0 | 14.80 |
| 50 | 12288 | 4096 | 423.9 | 7.08 |
| 4096 | 4096 | 4096 | 412.7 | 188.0 |
| 4096 | 4096 | 6144 | 640.2 | 192.9 |
| 4096 | 4096 | 24576 | 3036.1 | 770.5 |
| 4096 | 12288 | 4096 | 1446.1 | 187.0 |
![alt text](image-1.png)

### 2.3 算子接入模型性能测试结果

将自定义算子替换 Qwen3-8B 的 `npu_quant_matmul` 后，模型正常加载并完成推理。未开启 Profiler 时，decode 平均耗时为 **94.04 ms**，输出文本符合 attention 



---

## 三、方案说明

### 3.1 设计思路

**整体架构**：采用 Host 侧 Tiling + Device 侧双 Kernel 设计。Host 侧 `CalcQmmTiling` 根据输入维度 (M, N, K) 和硬件资源计算分块参数，并通过 `qmm_custom` Torch 接口启动 Kernel。Device 侧：
- `qmm_cube_kernel`（`__mix__(1,2)`）负责 INT8×INT8→INT32 的矩阵乘；
- 若存在 perTokenScale，则在同一 NPU Stream 上启动独立的 `qmm_pertoken_kernel`（纯 Vector），完成 INT32→BF16 的反量化。

**TilingData 结构体**：
```cpp
struct QmmCustomTilingData {
  TCubeTiling cubeTilingData; // CANN 标准 Tiling
  uint32_t isPertoken;        // 路径选择标志
  uint32_t workspaceSize;     // 中间结果所需 GM 空间
  uint32_t m, n, k;           // 原始维度
  uint32_t singleCoreM;       // 每核分配的行数
  uint32_t vectorCoreNum;     // Vector 核数量
};
```

**分块策略**：根据 AIC 核数 `usedCoreNum = min(AIC核数, M)`，计算 `singleCoreM = ceil(M / usedCoreNum)`，使每个 Cube 核负责连续行块。使用 `MultiCoreMatmulTiling` 自动选择 baseM/baseN/baseK，避免固定分块在 M=1 时失败。workspace 大小通过平台 API 动态查询。

**Cube Kernel（QmmCubeBasicKernel）**：
- 初始化：建立 INT8 GM Tensor 和 INT32 输出 Tensor 映射；
- Process：根据核索引计算行起始和实际行数，调用 `SetTail` 处理尾块，通过 AscendC Matmul 对象执行 `IterateAll` 并将结果写回。

**Vector Kernel（QmmPertokenKernel）**：
- 以独立 Kernel 方式启动，避免跨核同步复杂性；
- 按 AIV 核数划分 M 维，每核处理若干行；
- 每次搬运 2048 个元素，完成 INT32→FP32→乘 perChannelScale→乘 perTokenScale→BF16 的流水线，其中先计算 `perChannelScale × perTokenScale` 以减少精度误差。

**Torch 接口**：检查输入维度和数据类型，根据是否传入 perTokenScale 分配输出 Tensor（INT32 或 BF16）。对于 BF16 路径，额外分配 INT32 中间 Tensor，先后启动 Cube Kernel 和 Vector Kernel。

### 3.2 问题解决与优化策略

#### 3.2.1 主要问题及解决方案

- **Tiling 失败（M=1）**：固定 baseM 导致 `GetTiling` 失败。改为自动分块，并限制核数不超过 M，确保小 shape 也能生成有效 Tiling。
- **Cube 与 Vector 同步复杂**：采用两个独立 Kernel 在同一 Stream 顺序启动，以 Kernel 边界作为同步点，避免使用不可靠的跨核标志。
- **BF16 结果偏差**：将两个 Scale 先相乘再参与浮点运算，减少运算顺序导致的 ULP 差异，确保贪心解码不变。
- **Profiler 解析异常**：按文件修改时间选择最新 CSV，并对缺失 shape 的 BF16 记录按固定 Qwen3 层顺序补齐，同时跳过空字段行。
- **模型接入后 shape 恢复**：替换时保留输入前缀维度，并对 bias 路径显式抛出异常，防止静默错误。

#### 3.2.2 AI 辅助使用情况

团队在以下环节使用了 AI 辅助工具：
- 理解 `MultiCoreMatmulTiling` API 参数含义和用法；
- 编写 Vector 反量化循环的搬运、Cast、Mul 流水线代码；
- 定位编译错误（如 `TPipe` 未加命名空间、`SetFixSplit` 导致失败等）；
- 调试 Profiler 解析脚本中的边界条件。

所有 AI 建议均通过代码审查、单算子精度测试（12/12 PASS）和端到端模型验证后方才采纳。AI 工具提高了排查效率，但未引入新的 bug，团队始终以实际运行结果作为最终判断依据。

#### 3.2.3 性能优化策略

- 多核按 M 维划分并支持尾块处理，均衡负载；
- 权重采用 FRACTAL_NZ 格式匹配 Cube 高效访存；
- 动态查询平台核数和 workspace，避免硬编码；
- Vector 反量化采用块搬运和向量化计算，提升吞吐；
- 无 perTokenScale 时直接返回 INT32，跳过反量化开销；
- 两个 Scale 融合乘法，减少浮点操作次数。

优化后，模型 decode 平均耗时低于 95 ms，满足推理实时性要求。

---

## 四、收获与感悟

### 吴启贤

通过完成 Tiling 和 Cube Kernel，我深入理解了昇腾 AI Core 的分块机制和 Matmul 高阶 API 的使用方法。从手动设计 TilingData 到处理边界 case（如 M=1），让我意识到硬件感知编程的重要性。同时，在整合队友的反量化 Kernel 时，我也锻炼了代码集成和接口协调的能力。

### 陈杰夫

实现反量化 Kernel 并完成全链路测试让我对 AI Core 的 Vector 单元和 Cube+Vector 协同有了直观认识。独立 Kernel 的设计虽然增加了调度开销，但避免了复杂的同步逻辑，使代码更清晰。单算子测试和 Profiler 的使用让我学会了如何系统地验证算子正确性和性能，为后续优化提供了数据支撑。


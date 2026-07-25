# 团队实践报告

## 一、团队信息与贡献说明

### 1.1 团队基本信息

- 团队标识（组号）：园艺
- CANNJudge 提交账号：h846463
- CANNJudge 提交结果或链接：全部24个测试用例通过 ✅，误差0.00%

### 1.2 团队成员分工与贡献

### 1.2 团队成员分工与贡献

| 姓名 | GitCode 账号 | 负责模块/任务分工 | 实际贡献说明 | 对应 commit hash |
| --- | --- | --- | --- | --- |
| 袁子婷 | h846463 | Kernel侧算子实现、Tiling设计、单算子测试与性能优化 | 完成QmmCustom算子的Kernel侧核心实现，包括INT8矩阵乘法、反量化逻辑、Cube-Vector流水并行；完成Tiling结构体设计与Host侧适配；完成单算子功能测试与性能调优，CANNJudge 24/24通过 | `489ed64` |
| 罗艺笳 | (2301_79608650) | Notebook结果整理、模型接入验证、实践报告撰写 | 在独立云端NPU环境复测算子，完成单算子精度验证；将算子接入Qwen3-8B-W8A8模型并完成模型推理验证；完成实践报告撰写和Notebook结果整理 | `3ef7301` |

### 1.3 团队协作说明

袁子婷负责自定义算子的Tiling、Kernel主体和CANNJudge性能迭代，并将阶段成果提交到团队Fork。罗艺笳从团队提交建立个人分支，在独立云端NPU环境复测算子，完成单算子精度验证和模型接入验证，随后将算子接入Qwen3-8B-W8A8并完成模型推理。袁子婷在模型正确性验证基础上继续完成性能优化，最终CANNJudge 24/24通过。成果统一汇总到 submission/园艺_result，保留双方各自GitCode身份形成的有效提交。



## 二、结果展示

### 2.1 单算子精度比对结果

QmmCustom算子共24个测试用例，全部通过：

| 测试类型 | 测试用例数 | 通过数 | 通过率 |
| --- | --- | --- | --- |
| 非pertoken模式（INT32输出） | 12 | 12 | 100% |
| pertoken模式（BF16输出） | 12 | 12 | 100% |
| 总计 | 24 | 24 | 100% |

精度说明：
- INT32输出：完全准确，零误差
- BFLOAT16输出：满足精度要求（相对误差 < 1e-2，绝对误差 < 1e-2）

### 2.2 单算子性能测试结果

CANNJudge 最终所有24个测试点全部通过，输出误差均为0.00%。

| 测试点 | M | K | N | 模式 | 测试结果 | 用时 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 4096 | 4096 | INT32 | PASS | 34.74μs |
| 2 | 1 | 4096 | 6144 | INT32 | PASS | 36.42μs |
| 3 | 1 | 4096 | 24576 | INT32 | PASS | 306.76μs |
| 4 | 1 | 12288 | 4096 | INT32 | PASS | 333.86μs |
| 5 | 50 | 4096 | 4096 | INT32 | PASS | 146.14μs |
| 6 | 50 | 4096 | 6144 | INT32 | PASS | 156.40μs |
| 7 | 50 | 4096 | 24576 | INT32 | PASS | 80.28μs |
| 8 | 50 | 12288 | 4096 | INT32 | PASS | 82.58μs |
| 9 | 4096 | 4096 | 4096 | INT32 | PASS | 81.37μs |
| 10 | 4096 | 4096 | 6144 | INT32 | PASS | 53.83μs |
| 11 | 4096 | 4096 | 24576 | INT32 | PASS | 1.54ms |
| 12 | 4096 | 12288 | 4096 | INT32 | PASS | 839.38μs |
| 13 | 1 | 4096 | 4096 | BF16 | PASS | 347.91μs |
| 14 | 1 | 4096 | 6144 | BF16 | PASS | 219.80μs |
| 15 | 1 | 4096 | 24576 | BF16 | PASS | 216.34μs |
| 16 | 1 | 12288 | 4096 | BF16 | PASS | 129.18μs |
| 17 | 50 | 4096 | 4096 | BF16 | PASS | 857.63μs |
| 18 | 50 | 4096 | 6144 | BF16 | PASS | 1.87ms |
| 19 | 50 | 4096 | 24576 | BF16 | PASS | 1.24ms |
| 20 | 50 | 12288 | 4096 | BF16 | PASS | 2.97ms |
| 21 | 4096 | 4096 | 4096 | BF16 | PASS | 5.35ms |
| 22 | 4096 | 4096 | 6144 | BF16 | PASS | 12.0ms |
| 23 | 4096 | 4096 | 24576 | BF16 | PASS | 2.52ms |
| 24 | 4096 | 12288 | 4096 | BF16 | PASS | 5.05ms |

**精度统计**：
- INT32 输出：12/12 通过 ✅
- BF16 输出：12/12 通过 ✅
- **总计：24/24 通过 ✅**

### 2.3 算子接入模型性能测试结果

QmmCustom算子成功接入Qwen3-8B模型，替换原有的量化线性层。模型推理输出符合预期，平均推理耗时满足要求。

网络输出验证：

网络输入文本：
An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is

自定义算子接入后的输出以如下 attention 描述开头：
The output of an attention function is a weighted sum of the value vectors, where the weights are determined by the similarity between the query vector and each key vector in the set of key-value pairs.

该结果符合课程要求的 attention 逻辑描述，网络功能验证 PASS。


## 三、方案说明

### 3.1 设计思路

数据流图：

x1(ND) + x2(FRACTAL_NZ) -> Cube INT8 Matmul -> INT32中间结果 -> 无pertoken直接输出INT32 / 有pertoken则Vector反量化 -> BF16输出

Tiling设计：

1. 多核切分：基于Cube核数量对M和N维度进行切分。对于小M形状（M=1, M=50），主要沿N维切分；对于大M形状（M=4096），使用5x4二维网格覆盖20个AIC
2. 非对齐适配：对M维度非32整倍数场景（M=1, M=50），通过动态计算 mUse_ 和 nUse_ 进行适配，确保尾块正确处理
3. workspace管理：pertoken模式需要 M * N * sizeof(int32) 的workspace存放Cube中间结果
4. 特殊处理：N=6144且M=1时使用1个Cube核，避免NZ子矩阵偏移问题

Kernel实现：

1. 非pertoken模式（INT32输出）：Cube核执行INT8矩阵乘法，直接输出INT32结果到y
2. pertoken模式（BF16输出）：Cube核执行INT8矩阵乘法，输出INT32中间结果到workspace；M=1时使用1:1 AIC/AIV混合核，其余形状使用1:2；Vector核从workspace读取中间结果，执行per-channel和per-token反量化；使用AscendDequant接口进行per-channel反量化；使用BroadCast接口广播pertoken_scale；最终转换为BFLOAT16输出

### 3.2 问题解决与优化策略

问题1：NZ格式权重偏移计算
- 问题：x2使用FRACTAL_NZ格式，按ND格式计算偏移导致结果错误
- 解决：修正 x2Offset 计算公式为 nIndex_ * singleCoreN * Kb，确保正确读取NZ格式数据

问题2：M维度非对齐场景处理
- 问题：M=1、M=50等非32整倍数场景，固定分块导致越界或结果错误
- 解决：动态计算 mUse_ = MinU32(singleCoreM, M - mIndex_ * singleCoreM)，确保尾块只处理有效数据

问题3：Cube-Vector流水同步
- 问题：Vector核需要在Cube核完成计算后才能开始反量化
- 解决：使用 CrossCoreSetFlag 和 CrossCoreWaitFlag 实现核间同步，AIC核设置标志位，AIV核等待标志位

AI辅助使用说明：
- 使用AI工具协助生成代码框架和调试建议
- AI建议使用 AscendDequant 和 BroadCast 接口实现反量化
- 团队通过CANNJudge单算子测试验证AI建议的正确性

性能优化策略：
1. 多核并行：根据矩阵大小动态分配Cube核数量，充分利用多核计算能力
2. 流水并行：pertoken模式下Cube核和Vector核并行工作，Cube计算下一tile的同时Vector反量化当前tile
3. UB复用：使用 TBuf 复用UB空间，减少内存分配开销
4. 非对齐场景优化：对M=1、M=50等场景单独优化分块策略，避免无效计算


## 四、收获与感悟

### 袁子婷

通过本次启航营实践，我深入理解了Ascend C算子开发的全流程，从Tiling设计、Kernel实现到单算子测试和模型接入。特别是在处理NZ格式权重和非对齐场景时，对NPU的内存布局和计算特性有了更深刻的认识。Cube-Vector流水并行的实现也让我体会到了异构计算中任务分解和同步的重要性。

### 罗艺笳

通过本次实践，我了解了算子开发的全流程，包括单算子测试、性能分析和模型接入验证。特别是在模型接入验证过程中，通过确定性网络对比确认了算子替换的正确性，体会到了端到端验证的重要性。
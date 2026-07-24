# 团队实践报告

## 一、团队信息与贡献说明

### 1.1 团队基本信息

- 团队标识（组号）：KSKBL
- CANNJudge 提交账号：Drdarling
- CANNJudge 提交结果:
![RESULT](./Images/屏幕截图%202026-07-23%20170601.png)

### 1.2 团队成员分工与贡献

| 姓名 | GitCode 账号 | 负责模块/任务分工 | 实际贡献说明 | 对应 commit hash |
| --- | --- | --- | --- | --- |
| 员蕾皓、刘科言|@Drdarling、@orkneyrain  | 单文件算子开发与调试 | 完成 A8W8 QmmCustom 算子的单文件版本（Pertoken + CubeBasic 双路径）开发与精度验证 |   |
| 刘科言、庞程予| @orkneyrain、@Strike_Hawk  | 项目结构拆分与对接 CANNJudge | 将单文件算子拆分为标准 Ascend C 项目结构（op_host + op_kernel），适配 GE API 与 npu_op_code_gen 构建框架，解决编译兼容性问题 |   |
|庞程予、员蕾皓|@Strike_Hawk、@Drdarling  | 性能测试与模型接入 | 完成 12 组 shape 性能测试，实现 Qwen3-8B 模型接入与端到端验证 |   |

- gitcode分支地址
- 员蕾皓 ： https://gitcode.com/Drdarling/cann-launch-camp/tree/KSKBL-Drdarling

- 庞程予 ： https://gitcode.com/Strike_Hawk/cann-launch-camp/tree/KSKBL-Strike_Hawk

- 刘科言 ： https://gitcode.com/orkneyrain/cann-launch-camp/tree/KSKBL-orkneyrain

### 1.3 团队协作说明

本项目按照"开发 → 拆分适配 → 测试验证"三个阶段分工协作：

1. **单文件开发阶段**：在 Notebook 环境中完成 QmmCustom 算子的完整实现，包括 A8W8 矩阵乘法（CubeBasic 路径输出 INT32）和 Per-token 量化矩阵乘法（Pertoken 路径输出 BF16），验证两种路径的数值精度。

2. **项目拆分阶段**：将单文件代码拆分为 Ascend C 标准项目结构（`op_host/qmm_custom.cpp` + `op_kernel/qmm_custom.cpp`），使用 GE API 进行算子注册（`OP_ADD` + `OpDef`），替换原有 Torch API（`TORCH_LIBRARY`），确保与 CANNJudge 判题系统的 `npu_op_code_gen` 构建框架兼容。重点解决了 host/kernel 头文件分离、GE API 适配、`tiling/tiling_api.h` 与 `kernel_operator.h` 宏冲突等问题。

3. **测试验证阶段**：完成单算子精度比对（12 组 shape）、性能 profiling，以及 Qwen3-8B 模型接入后的端到端精度与性能验证。

## 二、结果展示

### 2.1 单算子性能测试结果

![单算子性能测试结果](./Images/屏幕截图%202026-07-23%20165237.png)

### 2.2 算子接入模型性能测试结果

![算子接入模型性能测试结果](./Images//屏幕截图%202026-07-23%20165349.png)

## 三、方案说明

### 3.1 设计思路

#### 整体架构

QmmCustom 算子实现 A8W8 量化矩阵乘法 `Y = (X @ W) * scale`，支持两种计算路径：

- **Path 1 — CubeBasic**：输入 INT8，输出 INT32（A8W8 标准矩阵乘法）
- **Path 2 — Pertoken**：输入 INT8，经过 Per-token 反量化后输出 BF16

```
输入:
  x1: [M, K] INT8  ND   (激活量化值)
  x2: [K, N] INT8  NZ   (权重量化值，FRACTAL_NZ 格式)
  scale: [N]   FLOAT ND  (per-channel 量化系数)
  pertoken_scale: [M] FLOAT ND (per-token 量化系数，可选)

输出:
  y: [M, N] BF16 / INT32 ND
```

#### TilingData 结构

```cpp
struct QmmCustomTilingData {
    TCubeTiling cubeTilingData;  // Matmul 库的标准 tiling 数据
    uint32_t isPertoken;         // 0=CubeBasic, 1=Pertoken
    uint32_t numBlocks;          // Block 数量（多核并行）
    uint64_t workspaceSize;      // 工作空间大小
};
```

#### Tiling 策略

使用 `matmul_tiling::MultiCoreMatmulTiling` 进行多核分块：

- **分块粒度**：
  - M ≤ 16 时 baseM=16, baseN=256（例外：N=6144 时 baseN=128）
  - M ≤ 64 时 baseM=32, baseN=128
  - M > 64 时 baseM=128, baseN=128
- **核数分配**：M ≤ 64 时根据 N 方向 tile 数量自适应选择对齐 AIV 核数（`ChooseAlignedAivNum`）；M > 64 时使用全部 AIV 核
- **A 矩阵**：GM / ND / INT8
- **B 矩阵**：GM / NZ / INT8（FRACTAL_NZ 格式）
- **C 矩阵**：CubeBasic 路径 → GM / ND / INT32；Pertoken 路径 → VECIN / ND / INT32
- **遍历方向**：`FIRSTM`，减少 scale/pertoken_scale 重复加载
- **工作空间**：`platform.GetLibApiWorkSpaceSize()` 动态获取

#### Kernel 实现

**QmmCubeBasicKernel**（Path 1 — INT32 输出）：

- 使用 `BasicMatmul`（A=INT8/GM/ND, B=INT8/GM/NZ, C=INT32/GM/ND）
- 按 baseN 迭代 N 方向，B 矩阵每个 tile 偏移量 = `nOffset * Kb`（NZ 格式特性）
- 最终输出 INT32 结果到 GM

**QmmPertokenKernel**（Path 2 — BF16 输出）：

- 使用 `PertokenMatmul`（A=INT8/GM/ND, B=INT8/GM/NZ, C=INT32/VECIN/ND）
- 反量化流水线：
  1. DataCopy 加载 scale `[tileN]` 和 pertoken_scale `[tileM]`
  2. MTE2→Scalar 同步等待
  3. `CastInSegments` INT32→FP32（分两段，每段 ≤ 255×64）
  4. Row-wise Mul：`fp32[row] *= scale[0:tileN]`
  5. Row-wise Muls：`fp32[row] *= pertoken_scale[row]`
  6. `CastInSegments` FP32→BF16（RINT 舍入）
  7. DataCopy 输出到 GM

### 3.2 问题解决与优化策略

#### 问题一：CANNJudge 无 Torch 头文件，Torch API 编译失败

**现象**：原始单文件代码使用 `TORCH_LIBRARY` 注册算子、`c10_npu::getCurrentNPUStream()` 获取流、`torch_npu/csrc/core/npu/NPUFormat.h` 判断 format。CANNJudge 环境仅有 CANN 原生头文件（`-I/home/judge/Ascend/cann-9.0.0/include`）。

**解决**：全部替换为 GE（Graph Engine）API：
- 算子注册：`ops::QmmCustom : public OpDef` + `OP_ADD(QmmCustom)`
- Stream：使用 `nullptr` 替代
- Format 判断：使用 `dim()` 替代 `get_npu_format()`

#### 问题二：`GetShapeDims` / `GetTensorDesc` 不可用

**现象**：CANN 9.0.0 中 `gert::Tensor` 没有 `GetShapeDims()` 和 `GetTensorDesc()` 方法。

**解决**：通过 `GetShapeSize()` 反推各维度大小：
```
N = scale->GetShapeSize()
K = x2->GetShapeSize() / N
M = x1->GetShapeSize() / K
```

#### 问题三：`DataType` / `Format` 列表长度导致 input dtype size=0

**现象**：`DataType({ge::DT_INT8})` 单元素列表导致框架无法正确推导 dtype，报错 "The dtype size of input[0] is 0"。

**解决**：改为双元素列表 `DataType({ge::DT_INT8, ge::DT_INT8})`，匹配 GE API 规范中 `DataType` 和 `Format` 需 2 元素的要求。

#### 问题四：`tiling/tiling_api.h` 与 `kernel_operator.h` 宏冲突

**现象**：Kernel 编译时 `kernel_operator.h` → `basic_api/kernel_type.h` 将 `DT_FLOAT`、`DT_INT8` 等定义为数值宏（0, 2, ...），而后 `qmm_custom_tiling.h` → `tiling/tiling_api.h` → `graph/types.h` 尝试使用这些符号作为枚举值，宏展开后变成非法语法 `0 = ::C_DT_FLOAT`。

**解决**：将 `tiling/tiling_api.h` 从公共头文件 `qmm_custom_tiling.h` 中移除，仅在 host 侧 `op_host/qmm_custom.cpp` 显式包含。Kernel 侧 `TCubeTiling` 由 `kernel_operator.h` 提供。

#### AI 辅助开发说明

- 使用 Claude Code 辅助完成 Torch API → GE API 的迁移，自动搜索并替换不兼容的 API 调用
- AI 辅助排查编译错误，快速定位头文件冲突根因（宏 vs 枚举）
- 所有 AI 生成/建议的代码均经过 CANNJudge 编译验证
- **经验**：AI 擅长 API 迁移和错误定位，但需人工验证最终正确性；复杂框架约束（如 `DataType` 列表长度、头文件组织规范）需要结合官方模板交叉确认

#### 性能优化策略

1. **B 矩阵 NZ 格式**：权重使用 FRACTAL_NZ 格式存储，提升 Cube 单元数据加载效率
2. **FIRSTM 遍历**：优先遍历 M 方向，减少 scale 和 pertoken_scale 在 N 方向迭代时的重复加载
3. **对齐核数选择**：`ChooseAlignedAivNum` 确保 tile 数量能被核数整除，避免负载不均衡
4. **VECIN 路径**：Pertoken 路径使用 VECIN 输出，避免矩阵乘法结果写回 GM 再读取的开销
5. **分段 Cast**：INT32→FP32 转换按 255×64 元素分段，突破硬件单次 cast 限制

## 四、收获与感悟

**成员 员蕾皓**：

本次启航营让我对华为昇腾 NPU 的算子开发有了从零到一的完整认识。在单文件算子开发阶段，我从 Ascend C 的 Matmul API 入手，逐步理解了 Cube 单元的数据流、NZ 格式的内存布局以及 Tiling 分块策略。最大的挑战是 Pertoken 路径的反量化流水线——需要在矩阵乘法结果还在 VECIN 队列中时就完成 INT32→FP32→BF16 的转换和 scale 乘加，涉及 MTE2 同步、分段 Cast 等硬件细节。调试过程中深刻体会到 NPU 上"数据搬运"和"计算"同样重要——一个错误的 DataCopy 参数就能让精度完全跑偏。在性能测试与模型接入阶段，看到自己写的算子成功替换 Qwen3-8B 的 Linear 层并跑出端到端推理结果时，那种成就感是无可替代的。感谢两位队友的密切配合，也感谢启航营提供的系统学习路径。

---

**成员 刘科言**：

这次实践中我最大的收获来自项目拆分和 CANNJudge 对接阶段。原本在 Notebook 里跑通的单文件代码，拆成 `op_host + op_kernel` 标准结构后遇到了大量的编译兼容性问题——从 Torch API 到 GE API 的迁移、`gert::Tensor` 接口差异、`DataType/Format` 列表长度的隐藏约束，再到最棘手的 `tiling/tiling_api.h` 与 `kernel_operator.h` 宏冲突。每一个问题都倒逼我去阅读 CANN 源码和官方模板，加深了对算子框架底层机制的理解。印象最深的是解决宏冲突的过程：通过逐层追踪 include 链，定位到 `kernel_type.h` 的宏定义与 `graph/types.h` 的枚举命名冲突，最终通过分离公共头文件的编译依赖解决了问题。这段经历让我认识到：在实际工程中，"适配环境"往往比"写对逻辑"更难。AI 工具（Claude Code）在这个阶段帮助巨大，大大加速了 API 迁移和错误排查的效率。

---

**成员 庞程予**：

我在本次实践中的主要工作集中在项目结构拆分和性能测试验证。参与算子从 Torch API 向 GE API 的迁移让我系统学习了 CANN 9.0.0 的算子注册机制——`OP_ADD` 宏、`OpDef` 基类、`TilingFunc` 回调——这些是 Ascend C 算子开发的"骨架"。在性能测试阶段，我从 12 组不同 shape 的 profiling 数据中直观感受到了 M、N、K 维度变化对 Cube 利用率和带宽的影响：小 M 场景下 pertoken 路径的 VECIN 策略有效避免了不必要的 GM 写回，大 M 场景下多核并行的负载均衡需要精细的核数选择策略。将 QmmCustom 接入 Qwen3-8B 模型并完成端到端推理是整个实践的高光时刻——看到量化模型的输出质量与 FP16 基线几乎一致，decode 延迟稳定在 41ms 左右，充分验证了 A8W8 量化方案在真实 LLM 推理场景中的价值。这次实践让我深刻理解了"算子开发不是孤立的代码编写，而是要从系统视角权衡精度、性能和工程可行性"。

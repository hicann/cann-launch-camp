# QmmCustom 自定义矩阵乘算子开发与 Qwen3-8B 集成报告

## 1. 项目概述

本项目基于 Ascend C 实现 `QmmCustom` 自定义量化矩阵乘算子，并将其注册为 PyTorch NPU 算子，最终接入 Qwen3-8B W8A8 模型推理流程。

算子支持两条计算路径：

1. 无 `pertoken_scale`：INT8 × INT8，INT32 累加并输出 INT32。
2. 带 `pertoken_scale`：Cube 完成 INT8 × INT8 → INT32，Vector 完成逐通道及逐 token 反量化，输出 BF16。

最终提交内容：

- `(幻影旅团)_qmm_custom.asc`
- `(幻影旅团)_result.ipynb`
- `(幻影旅团)_report.md`

---

## 2. 总体设计

算子整体分为五个部分：

1. 公共头文件、常量和辅助函数；
2. Host Tiling 与多核任务划分；
3. INT32 Cube Kernel；
4. BF16 反量化与 Vector 后处理 Kernel；
5. Kernel 入口、PyTorch NPU 注册、模型集成和 Profiling。

核心接口：

```cpp
struct QmmCustomTilingData {
    TCubeTiling cubeTilingData;
    uint32_t isPertoken;
    uint32_t workspaceSize;
};
```

设备侧入口采用 `__global__ __mix__(1, 2)`，workspace 参数保留 `__kfc_workspace__ GM_ADDR workspace`，PyTorch NPU 实现注册到 `PrivateUse1`。

---

## 3. 团队分工与 Git 提交

### 3.1 成员1：Tiling 与多核任务划分

GitCode 账号：`Yan1102649101`

主要提交：

```text
9b3c412 feat(qmm): implement tiling and multicore partition
```

完成内容：

- 实现 Host 侧 `CalcQmmTiling`；
- 根据 M、N、K 选择 `baseM`、`baseN` 和单核计算范围；
- 计算 `usedCoreNum` 和 workspace 大小；
- 实现 `QmmBlockInfo`；
- 根据逻辑核编号计算 `rowStart` 和 `colStart`；
- 处理 M、N 尾块和 inactive 核；
- 为成员2、成员3提供统一的多核划分接口。

### 3.2 成员2：INT32 Cube Kernel

GitCode 账号：`GUANGABRIEL`

主要提交：

```text
d1f969d feat(qmm): implement int32 cube kernel
```

完成内容：

- 实现 `KernelQmmInt32::Init` 和 `KernelQmmInt32::Process`；
- x1 使用 `rowStart * K` 计算输入偏移；
- FRACTAL_NZ 权重使用 `colStart * Kb` 计算偏移；
- 输出使用 `rowStart * N + colStart` 计算偏移；
- 使用 `SetTail(actualM, actualN)` 处理尾块；
- 使用 `IterateAll` 和 Fixpipe 将 INT32 结果直接写回 GM；
- inactive 核直接返回，避免越界访问和无效计算。

### 3.3 成员3：BF16 反量化与 Vector 优化

GitCode 账号：`Bei123Diana`

主要提交：

```text
826574b feat(qmm): implement bf16 dequant vector pipeline
```

完成内容：

- 实现 `KernelQmmPertoken::Init`、`KernelQmmPertoken::Process` 和 `DequantAndCopyOut`；
- Cube 输出类型设置为 `TPosition::VECIN / int32_t`；
- 每核心只搬运一次本核心负责的 scale 区间，并在多个 M tile 之间复用；
- 完整16行使用 `Cast`、`Brcb`、`Mul` 批量处理；
- M=1及不足16行的尾部使用逐行安全路径；
- 使用 `DataCopyParams` 处理有效列和目标 N stride；
- 反量化顺序为 `INT32 → FP32 → scale[n] → pertoken_scale[m] → BF16`。

### 3.4 成员4：入口注册、Notebook 与模型集成

GitCode 账号：`cxcxcxcx`

主要提交：

```text
c42740a feat(qmm): integrate torch npu operator
```

完成内容：

- 实现 `qmm_custom_kernel` 入口；
- 根据 `isPertoken` 分发 INT32 和 BF16 Kernel；
- 使用 `REGIST_MATMUL_OBJ` 与 `GetSysWorkSpacePtr()` 注册 Matmul 对象；
- 保留 `__kfc_workspace__`；
- 实现 PyTorch NPU 包装函数 `qmm_custom`；
- 根据是否存在 `pertoken_scale` 自动选择 INT32 或 BF16 输出；
- 实现 Profiling Tensor 信息、`GetAclDataType` 和 `GetFormat`；
- 使用 `TORCH_LIBRARY` 注册 schema；
- 使用 `TORCH_LIBRARY_IMPL(..., PrivateUse1, ...)` 注册 NPU 实现；
- 整合 Notebook、模型推理、Profiling 和最终报告。

---

## 4. Host Tiling 与多核划分

Host 侧根据输入矩阵形状选择计算方案，覆盖 M=1、M=50、M=4096，N=4096、6144、24576，以及 K=4096、12288 等规格。

设备侧通过 `GetQmmBlockInfo` 得到：

```text
rowStart
colStart
actualM
actualN
active
```

其中：

- `rowStart`：当前核心负责区域的起始行；
- `colStart`：当前核心负责区域的起始列；
- `actualM`：当前核心的有效行数；
- `actualN`：当前核心的有效列数；
- `active`：当前核心是否有有效任务。

该设计使两个 Kernel 共用相同的多核映射规则。

---

## 5. INT32 Cube Kernel

无 `pertoken_scale` 时执行：

```text
INT8 × INT8 → INT32
```

主要过程：

```cpp
matmulObj.SetTensorA(x1Global_);
matmulObj.SetTensorB(x2Global_);
matmulObj.SetTail(actualM_, actualN_);
matmulObj.IterateAll(yGlobal_);
matmulObj.End();
```

该路径由 Cube/Fixpipe 直接将 L0C 中的 INT32 结果写回 GM，避免额外的 VECIN → VECOUT → GM 搬运。

---

## 6. BF16 反量化与 Vector 流水

带 `pertoken_scale` 时，Cube 输出保存在 VECIN，随后由 Vector 完成反量化。

### 6.1 Scale 缓存

每个核心在 `Process` 开始时，将负责的整段 scale 搬入 UB：

```cpp
DataCopy(scaleLocal, scaleGlobal_[colStart_], actualN_);
```

同一核心负责的多个 M tile 复用该缓存，避免重复执行 GM → UB 搬运。

### 6.2 16行批处理

完整16行采用以下流程：

1. INT32 整块转换为 FP32；
2. 使用 `Brcb` 广播16个 token scale；
3. 按列乘以 per-channel scale；
4. 按行乘以 per-token scale；
5. 转换为 BF16。

### 6.3 尾行路径

对于 M=1，或 M=50 最后不足16行的部分，使用逐行路径：

```text
Cast → Mul → Muls → Cast
```

该方案避免在极小尾块上支付 `Brcb` 初始化开销，同时保证访问安全。

### 6.4 非连续写回

使用 `DataCopyParams` 设置有效列长度、UB 源 stride 和 GM 目标 stride，因此在 `validCols < baseN` 时仍能正确写入目标矩阵。

---

## 7. Kernel 入口与 PyTorch 注册

设备侧入口：

```cpp
__global__ __mix__(1, 2) __aicore__
void qmm_custom_kernel(...)
```

无 `pertoken_scale` 时创建 `KernelQmmInt32`；存在 `pertoken_scale` 时创建 `KernelQmmPertoken`。

Matmul 对象注册方式：

```cpp
REGIST_MATMUL_OBJ(
    &pipe,
    GetSysWorkSpacePtr(),
    op.matmulObj,
    &op.cubeTiling);
```

PyTorch Schema：

```cpp
qmm_custom(
    Tensor x1,
    Tensor x2,
    Tensor scale,
    Tensor? pertoken_scale=None
) -> Tensor
```

NPU 实现注册到：

```cpp
TORCH_LIBRARY_IMPL(ascendc_ops, PrivateUse1, m)
```

输出类型：

| 路径 | 输出类型 |
|---|---|
| 无 pertoken_scale | `torch.int32` |
| 有 pertoken_scale | `torch.bfloat16` |

---

## 8. 精度测试结果

Notebook 对12组矩阵规格分别测试 INT32 和 BF16 两条路径。

| M | K | N | INT32 | BF16 |
|---:|---:|---:|:---:|:---:|
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

最终结果：

```text
INT32：12/12 通过
BF16：12/12 通过
```

INT32 使用精确一致验证；BF16 使用允许 BF16 数值误差的容差验证。

CANNJudge 最终测试结果：

```text
24/24 通过
```

---

## 9. 算子性能结果

单位：微秒（us）。

| 规格 M,K,N | INT32 Duration | BF16 Duration |
|---|---:|---:|
| 1,4096,4096 | 35.780 | 39.400 |
| 1,4096,6144 | 46.300 | 53.560 |
| 1,4096,24576 | 154.780 | 178.380 |
| 1,12288,4096 | 77.360 | 81.960 |
| 50,4096,4096 | 34.500 | 42.280 |
| 50,4096,6144 | 52.640 | 72.240 |
| 50,4096,24576 | 159.760 | 202.960 |
| 50,12288,4096 | 86.020 | 99.040 |
| 4096,4096,4096 | 765.140 | 1353.500 |
| 4096,4096,6144 | 1166.280 | 1906.220 |
| 4096,4096,24576 | 5250.080 | 8190.780 |
| 4096,12288,4096 | 2539.100 | 3111.040 |

性能数据来自 Notebook 中保存的 `ASCEND_PROFILER_OUTPUT/kernel_details.csv`。

大矩阵规格下，BF16 路径包含额外的 INT32 → FP32、两次缩放和 FP32 → BF16 处理，因此耗时高于纯 INT32 路径。

---

## 10. Qwen3-8B 模型集成与 Profiling

项目将 `QmmCustom` 接入 Qwen3-8B W8A8 模型推理流程，模型目录为：

```text
Qwen3-8B-W8A8
```

模型权重由37个 safetensors shard组成。

Notebook 中已完成：

- 自定义动态库加载；
- Qwen3-8B W8A8 模型加载；
- 自定义算子替换；
- 离线推理；
- Prefill Profiling；
- Decode Profiling；
- `kernel_details.csv` 读取与统计。

Notebook 保存的模型 Profiling 文件包括：

```text
prof/prefill/.../ASCEND_PROFILER_OUTPUT/kernel_details.csv
prof/decode/.../ASCEND_PROFILER_OUTPUT/kernel_details.csv
```

---

## 11. 主要问题与解决过程

### 11.1 FRACTAL_NZ 权重偏移

问题：权重不是普通 ND 排布，若按照普通二维矩阵计算地址会产生错误结果。

解决：按照 FRACTAL_NZ 布局使用 `colStart * Kb` 计算权重偏移。

### 11.2 M/N 尾块越界

问题：M=50 等规格无法被基础块大小整除。

解决：Host 侧计算 `actualM`、`actualN`；设备侧调用 `SetTail(actualM, actualN)`；Vector 写回使用有效行列和 stride。

### 11.3 BF16 路径冗余数据搬运

问题：旧方案将 Cube 结果写入 GM workspace，再由 Vector 从 GM 读取，产生额外中间搬运。

解决：将 Matmul 输出位置改为 `TPosition::VECIN`，直接在 UB 中完成反量化处理。

### 11.4 Scale 重复搬运

问题：每个 Cube 输出 tile 都重新从 GM 读取 scale，影响大 M 规格性能。

解决：每核心只搬运一次负责的 scale 区间，并在多个 M tile 间复用。

### 11.5 小 M 与尾行性能

问题：M=1 和不足16行的尾块使用完整广播流程时，启动开销较高。

解决：完整16行使用批量广播路径，小块使用逐行路径。

### 11.6 新旧 Notebook 代码不一致

问题：部分环境中的 Notebook 仍保存旧类名、旧 workspace 处理方式和旧编译输出。

解决：选择最终验证版本 Notebook，并将四名成员拼接后的最终 ASC 写回 Notebook，再自动比较两份源码。

最终检查结果：

```text
Notebook内嵌ASC：743行
最终ASC：743行
NOTEBOOK_ASC_MATCH
```

---

## 12. AI 辅助使用说明

项目开发过程中使用 AI 工具辅助完成：

- 拆解题目要求和团队分工；
- 检查冻结接口与模块边界；
- 分析编译错误和运行日志；
- 对比新旧版本实现差异；
- 检查 FRACTAL_NZ 偏移、尾块和 stride；
- 生成静态检查命令；
- 整理测试结果和报告结构。

所有最终代码均经过实际编译、Notebook 测试、CANNJudge 测试和 Profiling 验证，未直接以未经验证的 AI 输出作为最终结果。

---

## 13. 最终提交自检

最终目录只包含：

```text
(幻影旅团)_result.ipynb
(幻影旅团)_qmm_custom.asc
(幻影旅团)_report.md
```

自检结果：

- 算子源码无 `TODO`；
- 不存在错误的 `PrivateUse2` 注册；
- workspace 保留 `__kfc_workspace__`；
- Matmul 使用 `GetSysWorkSpacePtr()`；
- INT32 12/12 通过；
- BF16 12/12 通过；
- CANNJudge 24/24 通过；
- 动态库加载成功；
- Notebook 与 ASC 源码完全一致；
- Qwen3-8B 推理和 Profiling 输出已保留；
- 四名成员的独立 Git 提交历史已保留，未 Squash。

---

## 14. 总结

本项目完成了一个同时支持 INT32 输出和 BF16 per-token 反量化输出的 Ascend C 自定义量化矩阵乘算子。

通过多核 Tiling、Cube 直接写回、Scale 缓存、16行 Vector 广播、尾块安全处理和 PyTorch NPU 注册，算子在全部测试规格中获得正确结果，并成功集成至 Qwen3-8B W8A8 模型推理流程。

团队通过独立分支和独立提交完成协作，最终代码、Notebook、测试输出和报告保持一致。

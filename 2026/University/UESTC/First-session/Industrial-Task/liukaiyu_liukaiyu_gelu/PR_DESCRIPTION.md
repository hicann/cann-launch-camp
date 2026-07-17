## 变更描述 / Description

<!-- 本 PR 做了什么，为什么需要 / What does this PR do and why -->

新增 **GELU（Gaussian Error Linear Unit，高斯误差线性单元）自定义算子**，基于华为昇腾 CANN 软件栈与 AscendC 编程语言实现，面向 `Ascend910B`，支持 `float16` / `float32` 数据类型与 `ND` 数据格式。

- **做了什么 / What**：提交完整的 GELU 算子包，包含 Host 侧算子定义 + 大核/小核两级 Tiling 策略（`op_host/`）、Device 侧 AICore 核函数实现（`op_kernel/`）、CMake 构建脚本与 README 使用文档。
- **为什么需要 / Why**：GELU 是 Transformer 类模型常用的激活函数。标准定义为精确 `erf` 形式 `gelu(x) = x * 0.5 * (1 + erf(x / √2))`，但 AscendC 的 `Erf` 属于慢指令、易成性能瓶颈。本算子改用 **5 阶近似实现**（保留 `x^5` 项的高阶 tanh 近似 exp 形式），在保持精度的同时规避慢指令，提升 AICore 吞吐。
- **近似公式 / Approximation**：`gelu(x) ≈ x / (1 + exp(-scale * (x + c1·x³ + c3·x⁵)))`，其中 `scale = 2·√(2/π) ≈ 1.59576912`、`c1 = 0.044715`、`c3 = 0.001072`。

**核心实现要点：**
- 5 阶近似计算（Device 侧 11 步）：`x² → x³ → x⁵(=x²·x³) → c1·x³ + c3·x⁵ → x + (…) → -scale·(…) → exp → +1 → x/(1+exp)`，使用 `Mul / Muls / Add / Adds / Exp / Div` 组合替代慢 `Erf` 指令。
- 大核/小核两级 Tiling：核数按 256B 对齐反推实际核数；核间不均分余数由前 `tailBlockNum` 个大核各多 1 个元素；核内按 `tileDataNum`（按总数据量分级：320/1040/1040/1020 × 32/typeLen）切分，尾块在 Host 侧完成 32B 对齐，Kernel 侧零开销使用。
- 双缓冲流水（`BUFFER_NUM = 2`）：`CopyIn → Compute → CopyOut` 通过 `TQue` 2 槽队列重叠搬运与计算；新增 `tmpBuf` 暂存 `x²` 供 `x^5` 复用。
- Tiling 数据结构协议（`GeluTilingData`）作为 Host 与 Kernel 之间的数据契约，含 `usedCoreNum` 记录实际核数供 Kernel launch 使用。

## 改动类型 / Change Type

- [ ] Bug 修复 / Bug Fix
- [x] 新功能 / New Feature
- [x] 性能优化 / Performance
- [ ] 代码重构 / Refactoring
- [ ] 文档更新 / Documentation
- [ ] 测试相关 / Test
- [ ] 其它 / Other

## 关联 Issue / Related Issues

<!-- Closes #000 可自动关闭 / Closes #000 to auto-close -->

- Closes #
- References #

## 测试信息 / Testing

<!-- 简要测试说明或关键结果 / Brief test description or key results -->

- 编译验证：参照 README 第 4 节流程（`mkdir build && cd build && cmake .. && make -j`），可正常生成 `cust_optiling`、`cust_opapi`、`ascendc_kernels` 三套产物。
- 功能正确性：实现遵循 5 阶近似 GELU 定义，在常用数值范围内与精确 `erf` 形式结果非常接近（近似实现，非逐位精确）。
- 调用方式：安装算子包后可通过 aclnn 单算子接口（`aclnnGelu` 系列）调用。

- [ ] 单元测试通过 / UT passed
- [ ] 集成测试通过 / ST passed
- [x] 人工验证通过 / Manual verified

## 检查清单 / Checklist

- [x] 代码符合规范 / Code follows style guide
- [ ] 测试添加并通过 / Tests added and passed
- [x] 文档已更新 / Docs updated if needed
- [x] 无硬编码敏感信息 / No secrets hardcoded
- [x] 提交信息符合规范 / Commit message follows convention

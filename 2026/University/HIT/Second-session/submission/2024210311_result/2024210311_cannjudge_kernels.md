# CANNJudge 赛题 Kernel 源码清单（标记说明）

- 赛事：**哈工大威海 · CANN启航营算子大赛**（https://cannjudge.cn/hitwh/cann ，共 3 道题目）
- 参赛账号：`azhengyyds`
- 个人标识：`2024210311`
- **运行说明与完成凭证**：见同目录 `2024210311_run_instructions.md`（含环境要求、三题运行/校验步骤、CANNJudge 提交记录与代码仓库链接、优化迭代记录、自检清单）

本目录内三个 `.asc` 文件，为本人针对该赛事三道题目实现的 Ascend C 算子核函数源码（「算子核函数工程(beta)」形式，直接 `#include` 使用）：

| 序号 | 题目（题目ID） | 题目链接 | 对应源文件 | 实现说明 |
| --- | --- | --- | --- | --- |
| 1 | **Gelu 算子** | https://cannjudge.cn/hitwh/cann/gelu | `2024210311_gelu_kernel.asc` | `KernelGelu`：分块搬运 + Vector 计算 + 尾部对齐处理，`run_kernel` 为入口 |
| 2 | **GlobalAvgPool 算子** | https://cannjudge.cn/hitwh/cann/globalavgpool | `2024210311_globalavgpool_kernel.asc` | 对 `(N, C, H, W)` 逐通道求空间均值并输出 `(N, C, 1, 1)`；按 spatial 是否 32B 对齐、能否整批放入 UB 分三条路径（对齐 → Pattern ReduceSum 批量归约；非对齐 → `DataCopyPad` 逐行补零 + 批量归约；spatial 过大 → 双缓冲软件流水，每 tile 独立归约后累加），统一 fp32 累加保证精度 |
| 3 | **QmmCustom 算子** | https://cannjudge.cn/hitwh/cann/qmmcustom | `2024210311_qmmcustom_kernel.asc` | A8W8 量化 matmul：`Int8MatmulKernel`（int32 输出）+ `DequantMatmulKernel`（经 per-channel scale 与 per-token scale 反量化输出 bf16），`qmm_entry` / `run_kernel` 为入口 |

## 赛题提交记录（CANNJudge 平台，提交账号 `azhengyyds`）

| 题目 | 最新提交时间 | 提交状态 | 提交链接 |
| --- | --- | --- | --- |
| Gelu 算子 | 2026/09/08 21:04:25 | **Pass** | https://cannjudge.cn/hitwh/cann/gelu/submission/6aa007d9c76b321ca60723c3 |
| GlobalAvgPool 算子 | 2026/09/08 23:10:02 | **Pass** | https://cannjudge.cn/hitwh/cann/globalavgpool/submission/6aa0254ac76b321ca6127da7 |
| QmmCustom 算子 | 2026/09/09 09:22:45（提交 ID 247178） | **Pass**，测试点 24/24 通过，输出错误占比 0.00% | https://cannjudge.cn/hitwh/cann/qmmcustom/submission/6aa0b4e5c76b321ca62dcde7 |

## 与课程作业提交文件的关系

- 以上三个文件是 **CANNJudge 赛事赛题 kernel 源码**，与本次 HIT Second-session 课程作业的三件套（`2024210311_result.ipynb`、`2024210311_qmm_custom.asc`、`2024210311_report.md`）相互独立，作为本人算子实现的补充材料随同一提交目录一并提交；
- `2024210311_qmm_custom.asc` 为课程作业中实际接入 Qwen3-8B 的算子版本（Cube + Vector 混合 + `TCubeTiling`）；`2024210311_qmmcustom_kernel.asc` 为赛题（CANNJudge）提交的实现版本。两者名称与用途不同，并存以体现不同实现路径，请勿混用。

## 说明与声明

- 三个 kernel 均由本人完成并核对，过程中部分环节使用了 AI 辅助（AI 工具与模型的声明见 PR 描述与本报告 3.2 节；提交信息中带有 `[AI: DeepSeek Harness, deepseek-v4-flash]` 标记）；
- 赛事平台上的提交记录与评测结果以 CANNJudge 平台为准（https://cannjudge.cn/hitwh/cann ）；
- 文件命名统一采用 `{个人标识}_` 前缀，便于与其它提交内容区分与追溯。

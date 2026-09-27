# 运行说明与完成凭证

- 姓名：张政
- 个人标识：2024210311
- GitCode 账号：azhengyyds
- 适用范围：哈工大威海 · CANN启航营算子大赛（校内赛）三道赛题，以及 HIT Second-session 课程实践（QmmCustom 接入 Qwen3-8B）

---

## 一、材料清单

| 文件 | 用途 |
| --- | --- |
| `2024210311_gelu_kernel.asc` | 校内赛题目 1「Gelu 算子」的 Ascend C 核函数源码 |
| `2024210311_globalavgpool_kernel.asc` | 校内赛题目 2「GlobalAvgPool 算子」的 Ascend C 核函数源码 |
| `2024210311_qmmcustom_kernel.asc` | 校内赛题目 3「QmmCustom 算子」的 Ascend C 核函数源码（CANNJudge 提交版本） |
| `2024210311_qmm_custom.asc` | 课程实践（接入 Qwen3-8B）所用的 QmmCustom 算子版本（Cube + Vector 混合 + `TCubeTiling`） |
| `2024210311_result.ipynb` | 课程实践完整 Notebook（保留全部单元格输出） |
| `2024210311_report.md` | 个人实践报告（结果展示、方案说明、AI 使用说明） |
| `2024210311_cannjudge_kernels.md` | 三道赛题 kernel 与题目的对应关系（标记清单） |
| `2024210311_run_instructions.md` | 本文件：运行说明与完成凭证 |

---

## 二、环境要求

- **硬件**：昇腾 NPU。校内赛使用 CANNLab 云开发环境，规格 `1*NPU 910B3 / 16vCPU / 32GiB`（模板 `cann_master-py3.12-A2-arm-*`）。
- **软件**：
  - CANN 工具包 `cann-9.0.0`（本实践环境路径 `/home/developer/Ascend/cann-9.0.0`）；
  - CMake ≥ 3.16 与 ASC 编译器（`find_package(ASC REQUIRED)`，Ascend C 直调工程自带 `CMakeLists.txt`）；
  - Python 3.11 + `torch` / `torch_npu`（课程实践接入模型时需要）。
- **可选工具**：VSCode + Jupyter（用于运行 `2024210311_result.ipynb`，连接方式见 `Second-session/prepare_env_guide.md`）。

---

## 三、三道校内赛题目的运行与验证方法

三道题目均为 CANNJudge「**算子核函数工程 (beta)**」形式：题目工程提供 `CMakeLists.txt`、`main.asc`、`data_utils.h`、`run.sh` 以及 `scripts/` 下的数据生成与校验脚本，参赛者只需实现核函数文件 `kernel.asc`。

**步骤：**

1. 在 CANNJudge 题目页下载对应题目的工程包（题目链接见第五节）；
2. 用本目录中对应的 kernel 文件**替换工程内的 `kernel.asc`**：

   | 题目 | 使用的文件 |
   | --- | --- |
   | Gelu 算子 | `2024210311_gelu_kernel.asc` → 重命名为 `kernel.asc` |
   | GlobalAvgPool 算子 | `2024210311_globalavgpool_kernel.asc` → 重命名为 `kernel.asc` |
   | QmmCustom 算子 | `2024210311_qmmcustom_kernel.asc` → 重命名为 `kernel.asc` |

3. 在工程目录下编译并运行：

   ```bash
   # 方式一：使用题目自带脚本
   bash run.sh

   # 方式二：手动构建（等价）
   mkdir -p build && cd build
   cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..
   make -j
   cd ..
   ```

4. **结果校验**：运行题目自带的校验脚本，或按 `run.sh` 输出的测试结果判断：

   ```bash
   python scripts/verify_result.py
   ```

5. **期望结果**：全部测试点 **Pass**（平台评测结论见第五节；本地验证与平台评测一致）。

---

## 四、课程实践（QmmCustom 接入 Qwen3-8B）的运行方法

1. 环境：CANNLab 云开发环境 + Jupyter Server（详细步骤见 `2026/University/HIT/Second-session/prepare_env_guide.md`）；
2. 直接查看结果：`2024210311_result.ipynb` **已保留全部执行输出**，打开即可看到编译日志、精度比对、模型推理与 Profiler 采集结果；
3. 如需复跑：按顺序执行 Notebook 全部单元格（"全部运行"），关键步骤依次为：
   - 环境变量与目录定位 → 写入算子 `qmm_custom.asc` → 编译算子（cmake + make，成功标志 `Built target ascendc_ops`）；
   - 单算子精度验证：规格1（M=1, K=4096, N=4096）与规格2（M=1, K=4096, N=6144），INT32 / BF16 两条输出路径 `allclose` 校验；
   - 将 `torch_npu.npu_quant_matmul` 替换为 `torch.ops.ascendc_ops.qmm_custom`，运行 Qwen3-8B 离线推理；
   - 打开 Profiler 重复推理，采集 `kernel_details.csv`（记录 `qmm_run_kernel`，内核类型 `MIX_AIC`）；
   - 结束后恢复原始 W8A8 源码（Notebook 最后单元格自动完成）。
4. 课程实践所用算子文件：`2024210311_qmm_custom.asc`（与 `2024210311_qmmcustom_kernel.asc` 为不同实现版本，勿混用）。

---

## 五、完成凭证（CANNJudge 提交记录 + 代码仓库提交链接）

### 5.1 CANNJudge 评测记录

| 题目 | 提交时间 | 评测结果 | 提交记录链接 |
| --- | --- | --- | --- |
| QmmCustom 算子 | 2026/09/09 09:22:45（提交 ID **247178**） | **Pass**，测试点 **24 / 24** 通过，输出错误占比 0.00% | https://cannjudge.cn/hitwh/cann/qmmcustom/submission/6aa0b4e5c76b321ca62dcde7 |
| Gelu 算子 | 2026/09/08 21:04:25 | **Pass** | https://cannjudge.cn/hitwh/cann/gelu/submission/6aa007d9c76b321ca60723c3 |
| GlobalAvgPool 算子 | 2026/09/08 23:10:02 | **Pass** | https://cannjudge.cn/hitwh/cann/globalavgpool/submission/6aa0254ac76b321ca6127da7 |

> 赛事首页（含题目列表与个人提交记录入口）：https://cannjudge.cn/hitwh/cann

### 5.2 代码仓库提交链接

| 项目 | 链接 |
| --- | --- |
| Pull Request（提交至 `cann/cann-launch-camp`） | https://gitcode.com/cann/cann-launch-camp/pull/603 |
| 个人分支（Fork） | https://gitcode.com/azhengyyds/cann-launch-camp/tree/2024210311 |
| 提交目录 | `2026/University/HIT/Second-session/submission/2024210311_result/` |

### 5.3 评测驱动的优化迭代过程（CANNJudge 提交历史）

| 题目 | 迭代情况 |
| --- | --- |
| Gelu 算子 | 早期提交出现多次 `Compile Error`（类型/接口用法不正确）与 `Wrong Answer`（边界与对齐处理问题）；逐步修正后于 2026/09/08 21:04:25 稳定 **Pass** |
| GlobalAvgPool 算子 | 迭代中先后出现 `Compile Error`、`Runtime Error`（分块/越界）与 `Wrong Answer`（非对齐场景精度）；修正三条路径（对齐 / `DataCopyPad` 逐行 / 双缓冲流水）后于 2026/09/08 23:10:02 稳定 **Pass** |
| QmmCustom 算子 | 在单算子精度（24 条用例）全部通过的基础上继续调优，最终提交 **24/24 Pass**，并给出各测试点用时与最优用时（详见实践报告 2.1 / 2.2 节） |

---

## 六、自检清单

- [x] 校内赛三题代码开发完成并完成本地验证（`allclose` / `verify_result.py`）
- [x] 三题均在 CANNJudge 完成有效提交，评测记录已保留（见 5.1）
- [x] 根据评测结果迭代优化，直至三题全部 Pass（见 5.3）
- [x] 最终代码、运行说明（本文件）与验证结果已提交至指定代码仓库（见 5.2）
- [x] CANNJudge 提交记录与代码仓库提交链接已集中保存（见第五节）

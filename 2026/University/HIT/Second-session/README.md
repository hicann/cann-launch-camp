# 哈工大威海启航营课程实践

本目录用于收录哈工大威海启航营课程实践结果。

## 实践内容

本次实践综合了启航营的所有学习内容，以Qwen3-8B在昇腾NPU上的推理优化为主线，从量化Qwen3-8B展开需求分析，开发自定义A8W8量化matmul算子并接入模型验证，实现需求分析——算子开发——单算子测试——测试驱动优化——算子接入模型测试的完整开发链路。

完整实践内容可见[自定义量化 A8W8 matmul 算子开发并接入 Qwen3-8B](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/llm_inference/qwen3_8b/06_custom_matmul_operator_development_and_integration_with_qwen3_8b.ipynb)

## 环境准备

详细操作步骤请参考[环境准备指南](./prepare_env_guide.md)，简要流程：

1. 创建CANNLab云开发环境并安装依赖
2. 启动Jupyter Server并复制terminal中打印出来的完整URL
3. 打开VSCode中的ipynb文件，点击`选择内核`、`现有Jupyter服务器`
4. 粘贴完整URL
5. 选择该Server下的python kernel
6. 保持terminal不关闭，运行notebook中的代码单元

## 作业提交要求

1. 本次作业要求**个人独立完成并以个人为单位提交**，每人只提交一份完整成果。
2. 提交完整的 Jupyter Notebook（.ipynb），**保留所有单元格输出**。
3. 提交前依次运行所有单元格，确保：
   - 自定义算子编译成功；
   - 算子功能测试通过；
   - 性能数据收集成功。
4. **个人标识与文件命名**：统一使用课程登记的本人标识（例如英文昵称、学号或姓名拼音）作为 `{个人标识}`，Notebook 命名为 `{个人标识}_result.ipynb`。提交到`cann-launch-camp/2026/University/HIT/Second-session/submission`目录下。
5. 每人需要将实现的自定义算子在[CANNJudge](https://cannjudge.cn/hitwh/cann/qmmcustom)上提交一次，并在个人报告中记录提交账号及测试结果。
6. 所有提交内容须使用本人 GitCode 账号完成可追溯的有效 commit，包括算子代码、Tiling、测试、性能优化、Notebook 或报告内容；纯空提交或仅修改格式不计为有效贡献。
7. **允许并且鼓励**使用 AI 工具辅助，但仍须独立思考，并在报告中简述 AI 工具解决了什么疑惑、是否引入新 bug，以及本人如何完成验证。
8. **不得抄袭**：作业须独立完成，严禁抄袭他人成果或与他人共用实质相同的提交内容。**一经发现抄袭，本次作业成绩记零分**，情节严重者取消后续评优资格。

## 提交目录结构

按以下规范创建个人目录，并将本人成果统一提交到 `submission` 目录下。每人只保留一个提交目录。

**个人提交目录命名格式**：
`{个人标识}_result`

**个人目录下的提交内容包括**：

- 个人 Notebook 结果：`{个人标识}_result.ipynb`
- 个人实现的自定义算子文件：`{个人标识}_qmm_custom.asc`
- 个人实践报告：`{个人标识}_report.md`

其中个人实践报告的模板可以参考[实践报告模板](./report_template.md)。

整体目录结构如下：

```
Second-session/
├── README.md
├── prepare_env_guide.md                # 环境准备指南
├── report_template.md                  # 实践报告模板
├── submission/                         # 结果提交目录
│   ├── {个人标识}_result                # 个人提交目录
│   │   ├── {个人标识}_result.ipynb      # 个人 Notebook 结果
│   │   ├── {个人标识}_qmm_custom.asc    # 个人实现的自定义算子文件
│   │   └── {个人标识}_report.md         # 个人实践报告
│   └── ...
└── ...
```

## 提交流程

详细操作步骤请参考 [PR 提交指南](./PR-submission-guide.md)，简要流程：

1. 使用本人的 GitCode 账号 Fork 代码仓，并在个人 Fork 中创建本人分支
2. 在自己的 `{个人标识}_result` 个人目录下完成全部内容
3. 在本人 Fork 中提交并推送可追溯的有效 commit
4. 自行完成代码、Notebook、测试结果和实践报告的自检后发起 PR

## 评分标准总览

> **实践任务**：本次实践共包含 6 个任务，其中任务一至任务三为算子核心实现（Tiling 设计、Cube-only Kernel、Cube+Vector Kernel），任务四为算子编译，任务五为单算子功能与性能测试，任务六为算子接入模型测试。任务之间存在依赖关系，需按顺序完成。

> **CANNJudge提交**： 实现的自定义算子需要在[CANNJudge](https://cannjudge.cn/hitwh/cann/qmmcustom) 上进行提交，单算子功能性能验收以CANNJudge结果为准。

<table>
  <tr>
    <th>考核项</th>
    <th>分值</th>
    <th>评分说明</th>
  </tr>
  <tr>
    <td>功能正确性-单算子</td>
    <td>15分</td>
    <td>以CANNJudge结果为准，共24条用例，全通过为满分，部分通过按比例给分</td>
  </tr>
  <tr>
    <td>功能正确性-网络</td>
    <td>5分</td>
    <td>自定义算子接入后，网络输出符合逻辑的attention描述为满分，否则为0分</td>
  </tr>
  <tr>
    <td>性能-单算子</td>
    <td>5分</td>
    <td>以CANNJudge结果为准，按性能排名从高到低给分</td>
  </tr>
  <tr>
    <td>性能-网络</td>
    <td>5分</td>
    <td>平均推理耗时低于100ms即为满分</td>
  </tr>
  <tr>
    <td rowspan="3">答辩</td>
    <td rowspan="3">10分</td>
    <td>3分：算子功能有问题，结果呈现不完整</td>
  </tr>
  <tr>
    <td>6分：算子功能正确，结果呈现完整，思路说明清晰</td>
  </tr>
  <tr>
    <td>10分：算子功能正确，结果呈现完整，思路说明清晰，有深入思考及优化实践</td>
  </tr>
</table>

> 性能分须先通过算子功能验证（未通过则性能 0 分）。

> 网络输入文本为：

```text
An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is
```

> 网络输出文本的参考结果如下：

```text
The output of an attention function is a **weighted sum of the value vectors**, where the weights are determined by the similarity between the **query vector** and each **key vector** in the set of key-value pairs.

### Mathematically, the attention function can be described as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where:
- $ Q $ is the **query matrix** (shape: $ [n \times d_k] $)
- $ K $ is the **key matrix** (shape: $ [m \times d_k] $)
- $ V $ is the **value matrix** (shape: $ [m \times d_v] $)
- $ d_k $ is the dimension of the keys and queries
- $ \text{softmax} $ is applied along the attention heads or across the keys

### Explanation of the Output:
- The **output** is a **vector** (or matrix, if multiple queries) of the same dimension as the values $ V $.
- Each element in the output is a **weighted combination** of the value vectors, with weights determined
```

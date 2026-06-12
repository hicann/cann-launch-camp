# cann-launch-camp仓

## 仓库简介
本仓库用于收纳、管理 CANN 开源社区启航营高校活动的课程作业、课设、毕设及各类课程实践作品，统一标准化目录层级与提交规范，保障社区作品提交规整、可追溯、可评审。所有贡献需遵循 CANN 社区开源规范、CLA 签署要求及本仓库提交流程。

## 目录结构
```
cann-launch-camp/
├── README.md                                   # 仓库总说明、准入规范、提交总览
├── 2026/                                       # 年度根目录（按年份分层管理）
│   ├── README.md                               # 2026年度启航营整体活动说明
│   └── University/                             # 高校启航营专项总目录
│       ├── UESTC/                              # 高校名称（英文大写缩写）
│           ├── README.md                       # 本校活动详情、开课说明
│           └── First-Session/                  # 期数目录（英文命名、连字符分隔）
│               └── Industrial-Task/            # 产业课题目录
│                   └── README.md               # 产业课题提交规范、合规要求
```

## 提交规范

### 命名规范
本仓库目录、文件夹命名严格统一规范，所有层级目录格式固定、禁止自定义修改，保障作品提交规整可追溯。
- 年度根目录：使用四位纯数字年份，示例：2026
- 高校总目录：固定英文名称，统一为 University
- 高校目录：采用高校官方英文大写缩写，示例：UESTC
- 期数目录：英文命名，单词间连字符分隔，示例：First-Session
- 产业课题目录：固定英文名称，统一为 Industrial-Task
- 用户提交目录：固定格式 gitcode账号_课设提交，示例：zhangsan123_课设提交

### 提交步骤
所有参与者需严格按照层级路径提交作品，禁止跨层级、自定义目录存放，标准提交流程如下：
1. 进入对应年度根目录下的 University 高校总目录
2. 进入对应高校目录下的期数目录
3. 进入期数目录下的 Industrial-Task 产业课题目录
4. 在产业课题目录下创建 gitcode账号_课设提交 命名的个人提交文件夹
5. 在个人文件夹内按课题要求，完整提交课设源码、文档、说明等全部贡献内容
6. 本地验证无误后，提交 Pull Request 至本仓库，等待社区审核合入
本仓库所有目录严格遵循英文标准化命名，禁止中文、空格、特殊字符，单词间使用连字符 - 分隔，层级固定不可自定义。

## 签署 CLA
参与项目贡献前，请根据您的贡献者身份签署相应的贡献者许可协议（CLA）。具体操作步骤请参阅：[CLA 使用指南](https://gitcode.com/cann/infrastructure/blob/main/docs/cla/cla%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md)。

## 参与贡献
完成 CLA 签署并确定贡献方向后，即可开始您的社区贡献之旅！社区欢迎各种形式（Issue、PR、会议、邮件、软件包等）的贡献，每一种贡献都将受到重视。您可以对 Issue 进行查找、创建、评论、处理等操作。具体操作步骤请参阅：[Issue 操作指南](https://gitcode.com/cann/community/blob/master/contributor/issue-operation.md)。

## 提交 PR
参与 CANN 社区代码贡献时，提交 PR 前需完成开发环境准备，并仔细了解项目特定的开发规范和版权声明要求（如涉及开源代码片段引用，请参考[片段引用指导](https://gitcode.com/cann/community/blob/master/contributor/third_party/snippet-reference-guildline.md)），确保您的贡献符合项目标准。具体操作步骤请参阅：[PR 操作指南](https://gitcode.com/cann/community/blob/master/contributor/pull_request_operation.md)。

## 许可证
本仓库采用MIT开源协议。

## 联系我们
Issues：https://gitcode.com/cann/cann-launch-camp/issues

## 致谢
感谢所有参与 CANN 开源社区竞赛的开发者和贡献者！


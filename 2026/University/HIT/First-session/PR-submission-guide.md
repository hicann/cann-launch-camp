# 📥 PR 提交指南

本文档指导哈工大启航营参赛团队完成从 Git 环境配置、团队协作到 Pull Request 提交的完整流程。每支队伍只提交一份成果和一个 PR，但每位队员都必须使用自己的 GitCode 账号完成可追溯的有效 commit。

## ✅ 前置条件
- 每位队员注册 [GitCode](https://gitcode.com) 账号并完成实名认证
- 每位队员签署 CANN 社区 CLA（贡献者许可协议），签署指南见 [CLA 使用指南](https://gitcode.com/cann/infrastructure/blob/main/docs/cla/cla%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md)

## ⚙️ 一、配置 Git 用户信息

每位队员在提交自己的工作前，都需要设置本人的用户名和邮箱（须与本人 GitCode 账号信息保持一致）：

```bash
git config --global user.name "您的GitCode用户名"
git config --global user.email "您的GitCode注册邮箱"
```

如果多人共用同一台电脑，建议在仓库目录内去掉 `--global`，仅为当前仓库配置提交身份。切换队员提交前必须同步切换为该队员本人的 GitCode 用户名和注册邮箱，禁止共用提交身份。

验证当前仓库实际使用的提交身份：

```bash
git config user.name
git config user.email
```

## 🍴 二、每位队员分别 Fork 代码仓

1. 每位队员使用本人的 GitCode 账号登录
2. 浏览器访问目标仓库：https://gitcode.com/cann/cann-launch-camp
3. 每位队员分别点击页面右上角 **Fork** 按钮，将仓库 Fork 到自己的账号空间
4. 每位队员 Fork 后的仓库路径为：`https://gitcode.com/{本人GitCode账号}/cann-launch-camp`

每位队员的代码必须提交并推送到本人 GitCode 账号下的 Fork，以便在 GitCode 上形成独立、可追溯的提交记录。

## 📦 三、克隆个人 Fork 并创建本人分支

```bash
git clone https://gitcode.com/{本人GitCode账号}/cann-launch-camp.git
cd cann-launch-camp
git checkout -b {团队标识}-{本人GitCode账号}
```

每位队员使用自己的 Fork 和分支独立工作，但所有成员都必须将成果放入同一个 `submission/{团队标识}_result` 团队目录，不得按个人账号创建多份结果目录。

## 🔄 四、同步上游仓库（可选）

**本步骤初次fork代码之后可不用操作，如果后续合并代码时有冲突，需要执行以下操作，拉取最新的代码来解决冲突**

为确保本地代码与上游最新版本一致，添加上游仓库为远程源：

```bash
git remote add upstream https://gitcode.com/cann/cann-launch-camp.git
git remote -v
```

拉取上游最新代码：

```bash
git pull upstream master --rebase
```

## 📁 五、创建团队提交目录

在本次实践的结果提交目录下（路径为 `cann-launch-camp/2026/University/HIT/First-session/submission`），按照命名规范创建唯一的团队目录。

**团队提交目录命名格式**：`{团队标识}_result`

例如，团队标识为 `team01`，提交结果目录为：

Windows 系统在 `2026/University/HIT/First-session/submission/` 目录下直接创建文件夹 `team01_result`。

Linux 系统可以执行以下命令创建文件夹：
```bash
mkdir -p 2026/University/HIT/First-session/submission/team01_result
```

将团队的 Notebook、算子文件和实践报告复制到该目录中，确保包含：

- 团队 Notebook 结果：`{团队标识}_result.ipynb`
- 团队实现的自定义算子文件：`{团队标识}_qmm_custom.asc`
- 团队实践报告：`{团队标识}_report.md`

Windows 系统可直接复制，Linux 系统可使用 `cp` 命令复制。团队报告须列出每位队员的姓名、GitCode 账号、具体分工、实际贡献和对应 commit hash。

## 📝 六、团队成员分别提交代码

每位队员应分别提交自己负责的代码或文档。开始提交前，先确认当前 Git 身份是本人账号：

```bash
git config user.name
git config user.email
```

如果信息不是本人 GitCode 用户名和注册邮箱，请先按第一节重新配置，再执行以下步骤。

### 提交本人负责的变更

#### 查看变更

```bash
git status
```

#### 暂存文件

```bash
# 请将占位内容替换为本人实际修改的文件路径，并精确暂存
git add 2026/University/HIT/First-session/submission/{团队标识}_result/{本人修改的文件}
```

#### 提交变更

```bash
git commit -m "feat(HIT): {本人负责内容的简要描述}"
```

提交信息建议使用 `feat(HIT): {简要描述}` 格式。每位队员至少应有一条能体现本人实际贡献的有效 commit；纯合并、空提交或仅修改格式不计为有效贡献。

#### 推送到远程

```bash
git push -u origin {团队标识}-{本人GitCode账号}
```

推送完成后，每位队员将本人分支地址和 commit hash 发给负责最终汇总的队员。例如：

```bash
https://gitcode.com/{本人GitCode账号}/cann-launch-camp/tree/{团队标识}-{本人GitCode账号}
```

### 追加修改

需要修复问题或补充材料时，应创建新的 commit 并正常推送：

```bash
git add 2026/University/HIT/First-session/submission/{团队标识}_result/{本人修改的文件}
git commit -m "fix(HIT): {本次修改的简要描述}"
git push origin {团队标识}-{本人GitCode账号}
```

> 不要使用空提交或仅修改格式来代替实际贡献。已将分支交给团队汇总后，不要使用 `git commit --amend` 或强制推送改写提交历史，否则可能导致报告中的 commit hash 失效。



## 🔀 七、创建 Pull Request

1. 推送成功后，浏览器访问 PR 发起人本人的 Fork 页面：`https://gitcode.com/{PR发起人GitCode账号}/cann-launch-camp`
2. 点击进入 **Pull Request** 标签页；
3. 在该标签页右上角有 **“+ 新建Pull Request”** 黑色按钮，点击进入；
4. 确认源分支为 `{团队标识}-{PR发起人GitCode账号}`，目标仓库为 `cann/cann-launch-camp` 的 `master` 分支，然后点击下一步；
5. 填写 PR 标题和描述，并在描述中注明团队标识及全体队员的姓名和 GitCode 账号；
6. 点击 **“创建”** 提交
7. 等待社区审核，审核意见将通过 PR 评论反馈，请及时关注并响应

## ❓ 常见问题

**Q：提交后 CLA 校验失败？**
A：请确认已在 CANN 社区完成 CLA 签署，签署指南见 [CLA 使用指南](https://gitcode.com/cann/infrastructure/blob/main/docs/cla/cla%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md)。

**Q：PR 审核不通过怎么办？**
A：审核人会提出修改建议。由负责该问题的队员在自己的 Fork 中使用本人 GitCode 身份完成新的 commit；PR 发起人将该提交合并到汇总分支并再次 push 后，PR 会自动更新，无需重新创建。

**Q：为什么不建议使用 `git commit --amend` 或强制推送？**
A：团队作业要求保留每位队员在 GitCode 上的独立有效 commit。改写提交历史可能导致 commit hash 变化或贡献记录丢失，因此应通过新增 commit 完成修复。

**Q：Fork 的仓库落后于上游仓库？**
A：执行 `git pull upstream master --rebase` 同步上游最新代码，再推送至个人仓库。

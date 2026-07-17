# 📥 PR 提交指南

本文档指导电子科技大学 CANN 创新实训基地学员完成从 Git 环境配置到 Pull Request 提交的完整流程。

## ✅ 前置条件
- 注册 [GitCode](https://gitcode.com) 账号并完成实名认证
- 签署 CANN 社区 CLA（贡献者许可协议），使用自己的电子邮箱，签署指南见 [CLA 使用指南](https://gitcode.com/cann/infrastructure/blob/main/docs/cla/cla%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md)

## ⚙️ 一、配置 Git 用户信息

- 1、下载git客户端，在网上搜索git，找到下载页面下载，然后安装到本地。
- 2、打开cmd终端，输入git验证是否安装成功，如果安装成功，终端会输出内容。
- 3、打开gitcode网页，点击“个人头像”---“个人设置”---左侧“个人资料”：修改容易记住的个人昵称。
- 4、继续在该页面，左侧“电子邮件”：添加自己的个人邮件，不要使用默认邮件。

打开终端，设置用户名和邮箱（需与 GitCode 账号信息保持一致）：

```bash
git config --global user.name "您的GitCode用户名"
git config --global user.email "您的GitCode注册邮箱"
```

验证配置：

```bash
git config --global user.name
git config --global user.email
```

## 🍴 二、Fork 代码仓

1. 浏览器访问目标仓库：https://gitcode.com/cann/cann-launch-camp
2. 点击页面右上角 **Fork** 按钮
3. 选择 Fork 到个人账号空间，等待 Fork 完成
4. Fork 完成后将自动跳转到您个人空间下的仓库副本，路径为：`https://gitcode.com/您的用户名/cann-launch-camp`

## 📦 三、克隆个人仓到本地

```bash
git clone https://gitcode.com/您的用户名/cann-launch-camp.git
cd cann-launch-camp
```
克隆之后，会在本地出现cann-launch-camp目录，windows系统点击进去即可。

## 🔄 四、同步上游仓库（可选）

**本步骤初次fork代码之后可不用操作，如果后续合并代码时有冲突，需要执行以下操作，拉取最新的代码来解冲突**

为确保本地代码与上游最新版本一致，添加上游仓库为远程源：

```bash
git remote add upstream https://gitcode.com/cann/cann-launch-camp.git
git remote -v
```

拉取上游最新代码：

```bash
git pull upstream master --rebase
```

## 📁 五、创建个人提交目录

在对应期次（Session）的产业课题目录下，按照命名规范创建个人目录。

**目录命名格式**：`姓名_gitcode账号_任务名称`

例如，姓名张三，账号 `zhangsan` 提交 `gelu` 算子任务的成果：

windows系统在`2026/University/UESTC/First-session/Industrial-Task/`目录下直接创建文件夹`zhangsan_zhangsan_gelu`

linux系统可以执行以下命令创建文件夹
```bash
mkdir -p 2026/University/UESTC/First-session/Industrial-Task/zhangsan_zhangsan_gelu
```

将您的代码及文档复制到该目录中，确保包含：

- 可运行源码，op_host、op_kernel文件夹
- 配套说明文档（README）
- 必要的构建或运行脚本

windows系统直接复制，linux系统使用`cp -r`命令复制

## 📝 六、提交代码
以下分为`首次提交`和`后续修改代码多次提交`的场景，

### 首次提交

完成代码编写后，按以下步骤提交：
首先在本地cann-launch-camp目录下打开cmd终端或者git bash

#### 查看变更

```bash
git status
```
可以通过此命令看到自己增加的文件，此阶段确保只增加或者修改自己的代码内容，不能修改其他人提交过的内容。

#### 暂存文件

```bash
# 添加指定目录下的所有文件
git add 2026/University/UESTC/First-session/Industrial-Task/zhangsan_zhangsan_gelu/
```
此命令会将已经修改的代码添加到git的暂存区。

#### 提交变更

```bash
git commit -m "feat(UESTC): 提交 gelu 算子课题成果"
```
此命令会给存在暂存区的代码增加提交日志，可以通过`git log`查看提交记录。

提交信息建议使用 `feat(UESTC): {简要描述}` 格式。

#### 推送到远程

```bash
git push origin master
```
此命令会将本地的代码推送到远程个人仓库中。

### 多次提交（追加修改）

如果首次提交后需要补充或修改代码（如修复问题、补充文档），请使用 `--amend` 参数合并到上一次提交中，避免产生多条重复提交记录：

#### 暂存追加修改

```bash
git add 2026/University/UESTC/First-session/Industrial-Task/zhangsan_zhangsan_gelu/
```

#### 追加到上一次提交

```bash
git commit --amend -m "feat(UESTC): 提交 gelu 算子课题成果"
```

> 如果提交信息无需修改，可省略 `-m` 参数，直接使用 `git commit --amend` 保留原提交信息。

#### 强制推送

```bash
git push origin master --force
```

> 由于 `--amend` 修改了提交历史，推送时需加 `--force` 参数。请确认仅修改了自己的提交，避免覆盖他人代码。

## 🔀 七、创建 Pull Request

1. 推送成功后，浏览器访问您 Fork 的仓库页面：`https://gitcode.com/您的用户名/cann-launch-camp`
2. 点击进入 **Pull Request** 标签页；
3. 在该标签页右上角有 **“+ 新建Pull Request”** 黑色按钮，点击进入；
4. 确认源分支（您推送代码的分支，示例是master分支）和目标仓库（`cann/cann-launch-camp` master 分支），然后点击下一步；
5. 填写 PR 标题和描述，具体内容按照页面提示填写；
6. 点击 **“创建”** 提交
7. 等待社区审核，审核意见将通过 PR 评论反馈，请及时关注并响应

## ❓ 常见问题

**Q：提交后 CLA 校验失败？**
A：请确认已在 CANN 社区完成 CLA 签署，签署指南见 [CLA 使用指南](https://gitcode.com/cann/infrastructure/blob/main/docs/cla/cla%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md)。

**Q：PR 审核不通过怎么办？**
A：审核人会提出修改建议，在本地修改后再次 commit 并 push，PR 会自动更新，无需重新创建。

**Q：Fork 的仓库落后于上游仓库？**
A：执行 `git pull upstream master --rebase` 同步上游最新代码，再推送至个人仓库。

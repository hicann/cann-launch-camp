# 📥 PR 提交指南

本文档指导电子科技大学 CANN 创新实训基地学员完成从 Git 环境配置到 Pull Request 提交的完整流程。

## ✅ 前置条件
- 注册 [GitCode](https://gitcode.com) 账号并完成实名认证
- 签署 CANN 社区 CLA（贡献者许可协议），签署指南见 [CLA 使用指南](https://gitcode.com/cann/infrastructure/blob/main/docs/cla/cla%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md)

## ⚙️ 一、配置 Git 用户信息

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
4. Fork 完成后将自动跳转到您个人空间下的仓库副本，路径为：`https://gitcode.com/{您的用户名}/cann-launch-camp`

## 📦 三、克隆个人仓到本地

```bash
git clone https://gitcode.com/{您的用户名}/cann-launch-camp.git
cd cann-launch-camp
```

## 🔄 四、同步上游仓库（可选）

本步骤初次fork代码之后可不用操作，如果后续有代码冲突时可执行以下操作

为确保本地代码与上游最新版本一致，可添加上游仓库为远程源：

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

**目录命名格式**：`{姓名}_{gitcode账号}_{任务名称}`

例如，账号 `zhangsan` 提交 `selu` 算子任务的成果：

```bash
mkdir -p 2026/University/UESTC/First-Session/Industrial-Task/zhangsan_gitcode账号_selu
```

将您的代码及文档复制到该目录中，确保包含：

- 可运行源码
- 配套说明文档（README）
- 必要的构建或运行脚本

## 📝 六、提交代码

### 首次提交

完成代码编写后，按以下步骤提交：

#### 查看变更

```bash
git status
```

#### 暂存文件

```bash
# 添加指定目录下的所有文件
git add 2026/University/UESTC/First-Session/Industrial-Task/zhangsan_gitcode账号_selu/
```

#### 提交变更

```bash
git commit -m "feat(UESTC): 提交 selu 算子课题成果"
```

提交信息建议使用 `feat(UESTC): {简要描述}` 格式。

#### 推送到远程

```bash
git push origin master
```

### 多次提交（追加修改）

如果首次提交后需要补充或修改代码（如修复问题、补充文档），请使用 `--amend` 参数合并到上一次提交中，避免产生多条重复提交记录：

#### 暂存追加修改

```bash
git add 2026/University/UESTC/First-Session/Industrial-Task/zhangsan_gitcode账号_selu/
```

#### 追加到上一次提交

```bash
git commit --amend -m "feat(UESTC): 提交 selu 算子课题成果"
```

> 如果提交信息无需修改，可省略 `-m` 参数，直接使用 `git commit --amend` 保留原提交信息。

#### 强制推送

```bash
git push origin master --force
```

> 由于 `--amend` 修改了提交历史，推送时需加 `--force` 参数。请确认仅修改了自己的提交，避免覆盖他人代码。

## 🔀 七、创建 Pull Request

1. 推送成功后，浏览器访问您 Fork 的仓库页面：`https://gitcode.com/{您的用户名}/cann-launch-camp`
2. 点击进入**Pull Request** 标签页；
3. 在该标签页右上角有 **“+ 新建Pull Request”** 黑色按钮，点击进入；
3. 确认源分支（您推送代码的分支，示例是master分支）和目标仓库（`cann/cann-launch-camp` master 分支），然后点击下一步。、；
4. 填写 PR 标题和描述，建议格式：

```
## 标题
feat(UESTC): 提交 selu 算子课题成果 - zhangsan

## 描述
- 实现了 selu 算子的 Ascend C 开发
- 包含完整源码、使用文档及运行说明
- 已通过本地精度验证
```

5. 点击 **Create Pull Request** 提交
6. 等待社区审核，审核意见将通过 PR 评论反馈，请及时关注并响应

## ❓ 常见问题

**Q：提交后 CLA 校验失败？**
A：请确认已在 CANN 社区完成 CLA 签署，签署指南见 [CLA 使用指南](https://gitcode.com/cann/infrastructure/blob/main/docs/cla/cla%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md)。

**Q：PR 审核不通过怎么办？**
A：审核人会提出修改建议，在本地修改后再次 commit 并 push，PR 会自动更新，无需重新创建。

**Q：Fork 的仓库落后于上游仓库？**
A：执行 `git pull upstream master` 同步上游最新代码，再推送至个人仓库。

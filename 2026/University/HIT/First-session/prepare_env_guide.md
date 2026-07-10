# 实践环境准备指南

本指南将介绍如何基于 CANNLab 云开发环境运行课程实践的 `.ipynb` 文件。

## 一、创建 CANNLab 云开发环境

打开 [cann-learning-hub](https://gitcode.com/cann/cann-learning-hub) 仓库后，可在仓库页面看到 **CANNLab** 图标。

将鼠标移至 **CANNLab** 图标，会弹出 **云开发** 和 **950 尝鲜体验** 两个选项。

点击**云开发**，使用华为云账号登录并进入开发者空间。进入页面后，点击 **创建** 按钮。

选择创建 NPU 环境，规格配置如下：
- **开发环境名称**：自行命名。
- **处理器类型**：昇腾 NPU。
- **模板名称**：`cann_master-py3.12-A2-arm-20260514`。
- **规格**：`1*NPU 910B3 16vCPUs 32GiB`。

点击 **创建** 按钮后，可以看到 NPU 环境已创建。点击 **开机** 按钮启动环境。

> 注意：如果开机时提示资源不足，说明当前时间段使用人数较多，可稍后再尝试。

环境成功开机后，点击**连接**中的**Visual Studio Code**，即可通过 VSCode 连接使用。


## 二、在CANNLab云开发环境上用VSCode连接Jupyter Server运行ipynb

通过在VSCode中安装jupyter相关插件，选择内核，能够直接在IDE中运行notebook，但有时这种方式会找不到内核；
更为推荐的方式是在服务器终端启动一个Jupyter Server，然后在VSCode中连接该Server。

核心步骤为：
1. 在服务器终端中安装Jupyter相关库
2. 在服务器终端中启动Jupyter notebook server
3. 复制终端输出的URL
4. 在VSCode的.ipynb中选择Jupyter Server，并粘贴这个URL
5. 使用期间保持Jupyter Server的终端运行

### 1、前置条件
请先确认已经进入服务器环境，并且当前terminal使用的是你希望运行代码的Python环境。
可以在终端中检查：
```bash
which python
python --version
```
也可以检查pip对应的是不是同一个python:
```bash
python -m pip --version
```
推荐使用`python -m pip`而不是`pip`来安装依赖，以确保使用的是当前terminal中的python环境，减少将依赖安装到另一个python环境中的问题。

### 2、安装必要的库
在terminal中执行:
```bash
python -m pip install --upgrade pip
python -m pip install ipykernel notebook jupyter
```

### 3、安装库的作用
#### ipykernel
ipykernel是Python的jupyter内核。
jupyter本身负责管理notebook页面、连接、消息通信等，真正执行python代码的是内核，对于python notebook来说，这个内核就是ipykernel。
如果没有安装ipykernel，可能会出现以下问题：
- notebook能打开，但是没有可用的python内核。
- VSCode已经连接到了Jupyter Server，但是代码单元无法正常运行。
- 运行代码时提示找不到kernel或kernel启动失败。
#### notebook
notebook提供jupyter notebook命令，也就是后面要启动的Notebook Server。
这个server会在服务器上启动一个本地服务，监听某个地址和端口，例如：
```text
http://127.0.0.1:8888
```
VSCode后续通过这个地址连接到Jupyter Server。
如果没有安装notebook，可能会出现 jupyter: command not found。

#### jupyter
jupyter是jupyter相关工具的总入口包。安装它的作用是确保jupyter这个命令和相关基础组件可用。很多jupyter子命令都会通过这个总入口来调用，例如：
```bash
python -m jupyter kernelspec list
```
简单来说：
- ipykernel负责执行python代码
- notebook负责启动Notebook Server
- jupyter提供jupyter命令入口和基础组件

### 4、启动Jupyter Server
在服务器终端中执行：
```bash
python -m notebook --ip=127.0.0.1 --port=8889 --no-browser
```
参数解释：
- --ip=127.0.0.1：只允许本机访问这个Jupyter Server。
- --port=8889：指定监听的端口号为8889。
- --no-browser：不自动打开浏览器。
启动成功后，终端会打印出类似下面格式的信息：
```text
To access the server, open this file in a browser:
    file:///xxx.html
Or copy and paste one of these URLs:
    http://127.0.0.1:8889/xxxtoken=xxx
```
复制 http://127.0.0.1:8889/xxxtoken=xxx 这一整段URL。

### 5、在VSCode的ipynb文件中连接Jupyter Server
打开VSCode中的.ipynb文件后，按照下面的步骤操作：
1. 点击notebook右上角的内核选择按钮，通常显示为`Select Kernel`、`选择内核`或当前kernel名称。
2. 选择`Existing Jupyter Server`或`现有Jupyter服务器`。
3. 当VSCode弹窗要求输入Jupyter Server的URL时，粘贴之前复制的完整URL。
4. VSCode连接成功后，再选择该Jupyter Server提供的python kernel。
5. 运行notebook中的代码单元。

### 6、使用过程中不要关闭terminal
启动Jupyter Server的terminal必须保持运行。
不要做这些操作：
- 关闭启动了Jupyter Server的terminal
- 按Ctrl+C中断Jupyter Server的运行
- 断开会导致进程中止的SSH session

### 7、推荐操作流程总结
第一次使用时：
```bash
python -m pip install --upgrade pip
python -m pip install ipykernel notebook jupyter
python -m notebook --ip=127.0.0.1 --port=8889 --no-browser
```
然后：
1. 复制terminal中打印出来的完整URL
2. 打开VSCode中的ipynb文件
3. 点击`选择内核`
4. 点击`现有Jupyter服务器`
5. 粘贴完整URL
6. 选择该Server下的python kernel
7. 保持terminal不关闭，运行notebook中的代码单元

后续使用时，只需要启动server：
```bash
python -m notebook --ip=127.0.0.1 --port=8889 --no-browser
```
再把URL复制粘贴到VSCode即可。

# FastGelu 自定义算子

## 任务说明
本目录提交 FastGelu 自定义算子实现代码。

## 目录结构
- op_host：Host 侧 tiling 与算子注册代码
- op_kernel：Ascend C Kernel 实现
- CMakeLists.txt：工程构建配置

## 算子公式
y = x / (1 + exp(-1.702 * x))

## 提交人
曹文卓
##邮箱
1877963619@qq.com

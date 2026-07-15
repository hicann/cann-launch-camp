#!/bin/bash

# 清除可能存在的性能文件
rm -rf prof/
# 创建性能文件存放目录
mkdir -p prof/
chmod 700 prof/
# 设置自定义算子so路径并执行调用代码
source ${HOME}/vendors/custom/bin/set_env.bash;msprof op --output=prof test/execute_op

echo "--- 性能采集完成 ---"
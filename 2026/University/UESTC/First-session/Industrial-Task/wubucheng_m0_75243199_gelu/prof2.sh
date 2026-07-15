#!/bin/bash

rm -rf prof2

# 创建性能文件存放目录
!mkdir -p prof2
chmod 700 prof2/

# 设置错误即停止
set -e

echo "--- 1. 编译并部署算子 ---"
cd custom_op/
rm -rf build_out
bash build.sh
./build_out/custom_opp*.run --install-path=${HOME}/
cd - > /dev/null

echo "--- 2. 编译调用代码 ---"
# 注意：确保 $ASCEND_TOOLKIT_HOME 环境变量已在环境中设置
g++ -I$ASCEND_TOOLKIT_HOME/include \
    -I${HOME}/vendors/custom/op_api/include \
    -L$ASCEND_TOOLKIT_HOME/lib64 \
    -L${HOME}/vendors/custom/op_api/lib \
    test.cpp \
    -lcust_opapi -lnnopbase -lacl_rt \
    -o test/execute_op

echo "--- 3. 仿真 ---"
# 设置自定义算子so路径并执行调用代码
source ${HOME}/vendors/custom/bin/set_env.bash;msprof op simulator --soc-version=Ascend910B1 --output=./prof2 --core-id=0 ./test/execute_op

echo "--- 全部完成 ---"
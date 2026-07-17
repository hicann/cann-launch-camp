#清空可能存在的build目录
rm -rf build
#创建build目录
mkdir build
#进入build目录
cd build

#激活CANN8.5.0环境
source ~/Ascend/cann-8.5.0/set_env.sh 
cmake ..
#编译
make
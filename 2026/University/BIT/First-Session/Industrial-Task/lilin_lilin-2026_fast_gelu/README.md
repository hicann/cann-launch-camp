FastGelu 昇腾自定义算子 README
1. 工程简介
基于昇腾CANN AscendC实现FastGelu激活自定义算子，分层拆分为Host侧Tiling逻辑与Device核函数，支持ACLN标准接口调用，适配昇腾NPU加速推理。

2. 目录结构

fast_gelu_op/
├── CMakeLists.txt       # 工程总编译配置
├── op_host              # Host侧Tiling、算子注册、ACLN接口生成
│   ├── CMakeLists.txt
│   ├── fast_gelu.cpp            # Tiling实现逻辑
│   ├── fast_gelu_tiling.h       # Tiling头文件
│   └── tiling_key_fast_gelu.h   # 切分参数定义
└── op_kernel            # AscendC Device核函数实现
    ├── CMakeLists.txt
    └── fast_gelu.cpp            # 硬件计算核

3. 文件说明
根目录
- `CMakeLists.txt`：加载CANN编译工具链，批量编译host与kernel模块，打包算子库。

op_host（主机侧）
- `fast_gelu_tiling.h` / `tiling_key_fast_gelu.h`：定义算子切分参数、Tiling入参结构体
- `fast_gelu.cpp`：Tiling切分逻辑，自动生成aclnn调用接口

op_kernel（设备核）
- `fast_gelu.cpp`：AscendC向量计算代码，NPU上执行FastGelu数学运算

4. 编译步骤
bash
1. 加载CANN环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
2. 构建编译目录
mkdir build && cd build
cmake ..
make -j

编译产物输出至build目录，包含Tiling库、Kernel核库、ACLN接口库。

5. 使用方式
1. 编译生成算子动态库部署至设备；
2. 通过`aclnnFastGelu`标准接口调用自定义算子；
3. 支持推理框架直接调用加速Gelu计算。

6. 环境依赖
- CANN 6.x及以上版本
- CMake ≥3.16
- 目标芯片：Ascend910B（可修改CMake配置切换）
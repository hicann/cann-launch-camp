# LessEqual 昇腾Ascend C算子开发作业
## 一、课题说明
本作业基于华为CANN框架，使用Ascend C语言开发LessEqual小于等于比较算子。算子实现两个输入张量逐元素对比运算，满足输入A小于等于输入B时输出True，否则输出False。本算子支持FP32、FP16两种浮点数据类型，功能与精度严格对标TensorFlow官方接口`tf.math.less_equal`，保证运算结果完全一致。

项目代码分为两端实现：op_host目录实现主机侧算子注册、参数校验与Tiling分块配置；op_kernel目录实现昇腾设备侧核心计算逻辑，符合标准Ascend C算子开发规范。

## 二、项目目录结构
```
wangxinyue_wwwwxxxyy_less_equal/
├── README.md
└── src/
    ├── CMakeLists.txt
    ├── op_host/
    └── op_kernel/
```
## 三、运行环境
硬件环境：昇腾910 AI处理器
软件环境：CANN开发套件、Ascend C编译工具链、CMake编译工具、TensorFlow环境

## 四、编译与运行步骤
依次在终端逐条执行下面命令：
```
cd wangxinyue_wwwwxxxyy_less_equal/src
mkdir build
cd build
cmake ..
make -j
```
执行以上命令即可完成算子工程完整编译，生成可部署的算子内核文件。

## 五、精度验证说明
本次开发以TensorFlow的`tf.math.less_equal`运算结果为真值基准，分别构造FP32、FP16不同维度测试张量，包含元素小于、等于、大于三种测试场景。

将同一组测试数据分别输入自研Ascend C算子与TensorFlow官方接口，逐元素比对输出布尔张量结果。经过多组测试验证，自研LessEqual算子输出结果与TensorFlow基准结果完全一致，精度无误差，功能正常可用。

## 六、算子功能参数
输入：两个维度匹配的浮点张量

输出：同维度布尔张量

支持数据类型：FP32、FP16

运算逻辑：逐元素执行 A ≤ B 逻辑判断
# LessEqual 算子

## 功能描述
实现逐元素比较两个输入张量，输出布尔值张量，对应 TensorFlow 的 `tf.math.less_equal`。

## 支持的数据类型
- FP16
- FP32

## 精度对齐
与 TensorFlow 严格对齐，误差小于 1e-5。

## 编译与运行
```bash
mkdir build && cd build
cmake ..
make

##目录结构
· op_host/：宿主端代码
· op_kernel/：核函数实现（含 tiling 配置）
· CMakeLists.txt：构建脚本

##作者

刘芯雨 / Merx_
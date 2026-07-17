# LessEqual算子快速入门

## 1. 环境准备

### 1.1 硬件要求

- 昇腾AI处理器：Ascend 910B
- 内存：32GB以上
- 存储：至少10GB可用空间

### 1.2 软件要求

| 软件 | 版本 | 说明 |
|------|------|------|
| Linux | Ubuntu 22.04/CentOS 8 | 操作系统 |
| CANN | 6.0.RC1+ | Ascend计算架构 |
| Ascend C Compiler | 随CANN安装 | 算子编译器 |
| CMake | 3.16+ | 构建工具 |

### 1.3 环境变量配置

```bash
# 设置CANN路径（根据实际安装路径调整）
export CANN_PATH=/usr/local/Ascend/cann/latest
export ASCEND_TOOLKIT_HOME=$CANN_PATH
export PATH=$CANN_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CANN_PATH/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$CANN_PATH/python/site-packages:$PYTHONPATH
```

## 2. 获取源码

```bash
# 克隆代码仓
git clone <repository-url>

# 进入项目目录
cd Industrial-Task/zhangyiming_zYiming_less_equal
```

## 3. 构建算子

### 3.1 创建构建目录

```bash
mkdir build && cd build
```

### 3.2 配置CMake

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_COMPUTE_UNIT=ascend910b
```

### 3.3 编译

```bash
# 使用多核编译（推荐）
make -j$(nproc)
```

### 3.4 验证构建结果

```bash
# 检查生成的动态库
ls -la *.so
```

预期输出：
```
custom.so          # 算子包动态库
cust_optiling.so   # Tiling动态库
cust_opapi.so      # ACLNN接口动态库
```

## 4. 算子调用示例

### 4.1 ACLNN C++调用示例

```cpp
#include <iostream>
#include "aclnn/aclnn.h"

int main() {
    // 初始化ACL
    aclInit(nullptr);

    // 1. 创建输入张量（float16类型）
    float data1[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float data2[] = {2.0f, 2.0f, 2.0f, 2.0f};
    
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    
    // 创建张量描述
    aclTensorDescriptor desc1, desc2;
    aclCreateTensorDescriptor(&desc1, ACL_FORMAT_ND, ACL_FLOAT16, 1, (int64_t[]){4});
    aclCreateTensorDescriptor(&desc2, ACL_FORMAT_ND, ACL_FLOAT16, 1, (int64_t[]){4});
    
    // 创建张量
    aclCreateTensor(&x1, desc1, data1, sizeof(data1));
    aclCreateTensor(&x2, desc2, data2, sizeof(data2));

    // 2. 调用LessEqual算子
    aclTensor* y = nullptr;
    aclnnStatus status = aclnnLessEqual(x1, x2, &y);
    
    if (status != ACLNN_STATUS_SUCCESS) {
        std::cerr << "算子调用失败" << std::endl;
        return -1;
    }

    // 3. 获取输出结果
    int8_t result[4];
    aclGetTensorBuffer(y, (void**)&result, nullptr);
    
    // 打印结果：预期 [1, 1, 0, 0] 即 [True, True, False, False]
    std::cout << "输出结果: ";
    for (int i = 0; i < 4; ++i) {
        std::cout << static_cast<int>(result[i]) << " ";
    }
    std::cout << std::endl;

    // 4. 释放资源
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(y);
    aclDestroyTensorDescriptor(desc1);
    aclDestroyTensorDescriptor(desc2);
    
    aclFinalize();
    
    return 0;
}
```

### 4.2 编译测试程序

```bash
g++ -o test_less_equal test_less_equal.cpp \
  -I$CANN_PATH/include \
  -L$CANN_PATH/lib64 \
  -lascendcl -laclnn
```

### 4.3 运行测试程序

```bash
./test_less_equal
```

预期输出：
```
输出结果: 1 1 0 0 
```

## 5. 广播功能测试

### 5.1 矩阵与向量广播

```cpp
// x1: shape=[2,2], dtype=float32
float matrix_data[] = {1.0f, 2.0f, 3.0f, 4.0f};

// x2: shape=[2], dtype=float32
float vector_data[] = {2.0f, 3.0f};

// 创建张量描述
aclCreateTensorDescriptor(&desc1, ACL_FORMAT_ND, ACL_FLOAT, 2, (int64_t[]){2, 2});
aclCreateTensorDescriptor(&desc2, ACL_FORMAT_ND, ACL_FLOAT, 1, (int64_t[]){2});

// x2将广播为[[2,3],[2,3]]
// 输出结果: [[True, True], [False, False]]
```

### 5.2 标量与张量广播

```cpp
// x1: shape=[3], dtype=int32
int32_t tensor_data[] = {1, 5, 3};

// x2: shape=[], dtype=int32（标量）
int32_t scalar_data = 3;

// x2将广播为[3, 3, 3]
// 输出结果: [True, False, True]
```

## 6. 精度验证

### 6.1 与TensorFlow对比

```python
import tensorflow as tf
import numpy as np

# 生成测试数据
np.random.seed(42)
x1_np = np.random.rand(100, 100).astype(np.float32)
x2_np = np.random.rand(100, 100).astype(np.float32)

# TensorFlow计算
x1_tf = tf.convert_to_tensor(x1_np)
x2_tf = tf.convert_to_tensor(x2_np)
y_tf = tf.math.less_equal(x1_tf, x2_tf)

# Ascend NPU计算（通过ACLNN）
# ... 调用本算子 ...

# 对比结果
# 要求：np.allclose(y_npu, y_tf.numpy()) == True
```

### 6.2 边界值测试

```python
# 测试边界值
x1 = tf.constant([np.finfo(np.float32).min, np.finfo(np.float32).max], dtype=tf.float32)
x2 = tf.constant([0.0, 0.0], dtype=tf.float32)

y = tf.math.less_equal(x1, x2)
# 预期: [True, False]
```

## 7. 常见问题

### Q1: 构建失败，提示找不到ASC包

**解决方案**：确保正确设置了CANN环境变量，并检查CMake配置。

```bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann/latest
```

### Q2: 算子调用失败，提示数据类型不支持

**解决方案**：检查输入张量数据类型，确保为以下支持类型之一：float16、float32、int32、int8。

### Q3: 广播操作结果不正确

**解决方案**：检查输入张量形状是否满足广播规则。广播规则：从最右侧维度开始逐维比较，维度大小相等或其中一个为1时可广播。

### Q4: 非对齐维度报错

**解决方案**：本算子已支持非32整倍数的维度，无需额外处理。

## 8. 调试技巧

### 8.1 查看Tiling配置

在Host侧`TilingFunc`中添加调试信息，查看计算的Tiling参数：

```cpp
std::cout << "mode: " << tiling->mode << std::endl;
std::cout << "totalLen: " << tiling->totalLen << std::endl;
std::cout << "tileLen: " << tiling->tileLen << std::endl;
std::cout << "blockDim: " << tiling->blockDim << std::endl;
```

### 8.2 启用详细日志

```bash
export ASCEND_GLOBAL_LOG_LEVEL=3
```

### 8.3 使用Profiling工具

```bash
# 启动Profiling
ascend-profiler start

# 运行程序
./test_less_equal

# 停止Profiling并生成报告
ascend-profiler stop -o profile_report
```

## 9. 性能优化建议

### 9.1 数据类型选择

- **优先使用float16**：float16数据量小，计算速度快，适合大多数深度学习场景
- **int8适合量化场景**：数据量最小，但需要先转换为half再比较

### 9.2 张量形状优化

- **避免极端形状**：如N=1或N=10000的边界值
- **尽量使用对齐形状**：虽然算子支持非对齐，但对齐形状性能更好

### 9.3 批量处理

- **合并小张量**：多个小张量运算可以合并为一个大张量，减少算子调用开销

## 10. 技术支持

- 参考文档：[LessEqual算子任务说明](LessEqual.md)
- CANN官方文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/60RC1/infacldevg/opdev/atlasopdev_16_0000.html
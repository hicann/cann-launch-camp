# Addcmul 算子

## 课题说明
基于 Ascend C 实现 Addcmul 算子，支持 910B 芯片。

## 计算公式
y = input_data + x1 × x2 × value

## 支持特性
- 数据类型：float16、float32、int8、int32
- NumPy 风格广播语义
- 非 32 字节对齐维度

## 编译与运行

```bash
# 在 CANN 环境中
mkdir -p build && cd build
cmake .. -DCMAKE_CXX_COMPILER=g++
make -j4
```

## 目录结构
```
op_host/addcmul.cpp         # Host 侧 Tiling 实现
op_kernel/addcmul.cpp       # Kernel 侧核函数实现
op_kernel/addcmul_tiling.h  # Tiling 结构体定义
op_kernel/tiling_key_addcmul.h  # Tiling Key 定义
CMakeLists.txt              # 构建配置
```
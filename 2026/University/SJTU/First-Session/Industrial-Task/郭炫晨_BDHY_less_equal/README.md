# LessEqual 算子实现

## 算子功能
元素级比较运算 x1 <= x2，返回布尔值。

## 支持数据类型
- float16 (DT_FLOAT16)
- float32 (DT_FLOAT)
- int32 (DT_INT32)
- int8 (DT_INT8)

## 文件结构
- op_kernel/less_equal_tiling.h - Tiling结构体定义
- op_kernel/less_equal.cpp - Kernel核函数实现
- op_host/less_equal.cpp - Host侧Tiling实现与算子注册

## 运行芯片
Ascend 910B
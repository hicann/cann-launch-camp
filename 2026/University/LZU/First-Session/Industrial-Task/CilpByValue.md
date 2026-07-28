# 题目内容

**描述**

## 一、赛题背景

ClipByValue是数值裁剪算子，用于将张量中的元素限制在指定范围内。广泛应用于梯度裁剪、数值稳定性处理、激活函数限制等深度学习场景，是防止数值溢出和梯度爆炸的重要工具。

本题要求基于TensorFlow原生tf.raw_ops.ClipByValue算子的核心业务逻辑，采用Ascend C编程语言进行算子原生开发，在昇腾NPU硬件上实现一款高性能、高精度的ClipByValue算子。

## 二、算子功能描述

实现的ClipByValue算子需将输入张量中的每个元素裁剪到[min, max]范围内。若元素小于min则输出min，若大于max则输出max，否则保持原值。

算子支持浮点和整数数据类型，输出类型与输入类型保持一致，需兼容非32整倍数的维度非对齐场景。

## 三、核心定义与约束

### 3.1 参考算子

TensorFlow原生算子：tf.raw_ops.ClipByValue

参考文档：https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/raw_ops/ClipByValue

### 3.2 输入输出与属性总览

| 类型 | 参数名 | 类型 | 维度形状 | 支持数据类型 | 数据格式 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| INPUT（必选） | x | tensor | (..., N4, N3, N2, N) | float16、float32、int32 | ND | 输入张量 |
| INPUT（必选） | clip_value_min | tensor | 标量或与x同形状 | float16、float32、int32 | ND | 裁剪下界 |
| INPUT（必选） | clip_value_max | tensor | 标量或与x同形状 | float16、float32、int32 | ND | 裁剪上界 |
| OUTPUT（输出） | y | tensor | (..., N4, N3, N2, N) | 与输入类型相同 | ND | 裁剪结果张量 |

### 3.3 关键输入约束

- 维度取值范围（均为正整数）：
- N ∈ [1, 10000]
- N2 ∈ [1, 10000]
- N3 ∈ [1, 2000]
- N4 ∈ [1, 500]
- 输入为任意多维张量，最终维度可拆解为(..., N4, N3, N2, N)，前序...为任意合法批次维度
- 非对齐场景兼容：N、N2、N3、N4 均可能为非32的整倍数，算子需适配内存/数据非 32 字节对齐的场景
- 数值取值约束：clip_value_min ≤ clip_value_max
- 数据类型约束：x、clip_value_min、clip_value_max的数据类型必须相同

### 3.4 数学公式与计算规则

- **基础公式：**

y = clamp(x, clip_value_min, clip_value_max)

- **计算规则：**
- 若 x < clip_value_min，则 y = clip_value_min
- 若 x > clip_value_max，则 y = clip_value_max
- 否则 y = x
- **精度要求：** 裁剪运算要求完全准确，无误差容忍

### 3.5 输出要求

- **形状约束：** 输出张量的形状与输入张量x形状完全一致
- **类型约束：** 输出数据类型与输入数据类型保持一致
- **数值范围：** 输出值在[clip_value_min, clip_value_max]范围内

## 四、规则要求

- **广播规则：** 支持clip_value_min和clip_value_max为标量或与x同形状的张量
- **性能要求：** 在保证正确性的前提下，优化计算性能，充分利用NPU硬件特性

## 五、示例说明

### 示例 1：基础裁剪

```
输入x：tensor ([-5.0, 0.0, 5.0, 10.0])，dtype=float32，shape=[4]
输入clip_value_min：tensor ([-2.0])，dtype=float32，shape=[]
输入clip_value_max：tensor ([6.0])，dtype=float32，shape=[]

输出y：tensor ([-2.0, 0.0, 5.0, 6.0])，dtype=float32，shape=[4]
```

结果解释：-5.0<-2.0裁剪为-2.0，10.0>6.0裁剪为6.0，其他保持原值

### 示例 2：整数类型裁剪

```
输入x：tensor ([1, 5, 10, 15])，dtype=int32，shape=[4]
输入clip_value_min：tensor ([3])，dtype=int32，shape=[]
输入clip_value_max：tensor ([12])，dtype=int32，shape=[]

输出y：tensor ([3, 5, 10, 12])，dtype=int32，shape=[4]
```

结果解释：1<3裁剪为3，15>12裁剪为12

## 六、测试用例覆盖范围

- **数据类型覆盖：** float16、float32、int32所有支持类型
- **维度场景覆盖：**
- 基础维度：1维、2维、3维、4维
- 非对齐场景：N、N2、N3、N4为非32整倍数的内存非对齐场景
- **边界场景覆盖：**
- 极限维度：N=1、N=10000等边界值
- 边界裁剪：所有元素都需要裁剪的情况
- **精度场景覆盖：**
- 正常数值范围：常规裁剪验证
- 完全准确：裁剪运算无误差容忍
# FastGelu Ascend C 算子提交说明

## 目录结构

```text
caoxiangyang_2301_79503900_fast_gelu/
|-- README.md
`-- code/
    |-- CMakeLists.txt
    |-- op_host/
    |   |-- CMakeLists.txt
    |   `-- fast_gelu.cpp
    `-- op_kernel/
        |-- CMakeLists.txt
        |-- fast_gelu.cpp
        |-- fast_gelu_tiling.h
        `-- tiling_key_fast_gelu.h
```

## 实现内容

本提交基于 Ascend C 实现 FastGelu 自定义算子，输出 shape 和 dtype 均与输入保持一致，支持 `float16` 和 `float32`，数据格式为 `ND`。

题目公式为：

```text
y = x * exp(0.851 * (x - abs(x))) / (1 + exp(-1.702 * abs(x)))
```

kernel 中使用等价且更稳定的一指数形式：

```text
e = exp(-1.702 * abs(x))
y = (max(x, 0) + min(x, 0) * e) / (1 + e)
```

### 等价公式推导

令：

```text
beta = 1.702
0.851 = beta / 2
e = exp(-beta * abs(x))
```

按 `x` 的正负分两种情况讨论。

当 `x >= 0` 时：

```text
abs(x) = x
x - abs(x) = 0
exp(0.851 * (x - abs(x))) = exp(0) = 1
e = exp(-1.702 * x)
```

原公式可化简为：

```text
y = x / (1 + e)
```

此时：

```text
max(x, 0) = x
min(x, 0) = 0
```

代入统一表达式：

```text
(max(x, 0) + min(x, 0) * e) / (1 + e)
= (x + 0 * e) / (1 + e)
= x / (1 + e)
```

与原公式一致。

当 `x < 0` 时：

```text
abs(x) = -x
x - abs(x) = x - (-x) = 2x
```

分子指数项为：

```text
exp(0.851 * (x - abs(x)))
= exp(0.851 * 2x)
= exp(1.702 * x)
```

又因为：

```text
-1.702 * abs(x)
= -1.702 * (-x)
= 1.702 * x
```

所以：

```text
e = exp(-1.702 * abs(x)) = exp(1.702 * x)
```

原公式可化简为：

```text
y = x * e / (1 + e)
```

此时：

```text
max(x, 0) = 0
min(x, 0) = x
```

代入统一表达式：

```text
(max(x, 0) + min(x, 0) * e) / (1 + e)
= (0 + x * e) / (1 + e)
= x * e / (1 + e)
```

也与原公式一致。

因此，无论 `x >= 0` 还是 `x < 0`，原公式都可以统一写成：

```text
e = exp(-1.702 * abs(x))
y = (max(x, 0) + min(x, 0) * e) / (1 + e)
```

该写法有两个目的：

- 指数输入始终小于等于 0，避免 `exp(large_positive)` 溢出。
- 减少指数运算等成本开销较大的运算次数。

## 关键设计

- Host 侧完成输出 shape/dtype 推导。
- Tiling 数据包含总元素数、单 tile 元素数和 32B block 对齐元素数。
- Kernel 侧按 32B block 多核切分，真实尾块由最后一个核处理。
- 对齐主循环使用 `DataCopy`，非 32B 尾块使用 `DataCopyPad`，兼容非对齐输入长度。
- 使用 `TBuf<VECCALC>` 保存 `exp(-1.702 * abs(x))` 中间结果。
- 当前版本使用单缓冲，避免双缓冲在小规模测试中引入额外 UB 和调度成本。



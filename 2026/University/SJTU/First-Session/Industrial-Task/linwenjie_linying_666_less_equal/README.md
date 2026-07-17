# LessEqual Ascend C 自定义算子

## 1. 项目简介

本项目实现了面向昇腾 AI 处理器的 Ascend C 自定义 `LessEqual` 算子。

算子对两个输入张量逐元素执行小于等于比较：

`y = (x1 <= x2)`

当对应位置满足 `x1 <= x2` 时输出 `true`，否则输出 `false`。

实现支持同形状输入、标量广播和一般张量广播，并针对不同数据规模设计了多种计算路径。

## 2. 作者信息

- 姓名：林文杰
- GitCode 账号：linying_666
- 任务名称：LessEqual

## 3. 目标平台

- 目标处理器：Ascend 910B
- 开发语言：Ascend C / C++
- 构建工具：CMake
- 验证环境：CANN 9.0.0

## 4. 输入与输出

### 输入

| 名称 | 说明 |
| --- | --- |
| `x1` | 第一个输入张量 |
| `x2` | 第二个输入张量 |

两个输入张量应具有相同数据类型，并满足广播规则。

支持的数据类型：

- `float16`
- `float32`
- `int32`
- `int8`

支持的数据格式：

- `ND`

### 输出

| 名称 | 数据类型 | 说明 |
| --- | --- | --- |
| `y` | `bool` | `x1 <= x2` 的逐元素比较结果 |

输出形状为两个输入张量广播后的形状。

## 5. 广播能力

当前实现支持：

1. 两个输入形状完全相同；
2. `x1` 为标量；
3. `x2` 为标量；
4. 一般张量广播；
5. 最高 32 维张量。

## 6. 实现说明

### Host 侧

Host 侧负责：

- 算子原型注册；
- 输出形状推导；
- 输出数据类型推导；
- 广播形状与步长计算；
- UB 空间评估；
- Tile 大小选择；
- AI Core 数量与任务划分；
- TilingKey 计算路径选择。

### Kernel 侧

Kernel 侧包含：

- Tiny 同形状路径；
- Tiny 标量广播路径；
- 小规模同形状路径；
- 中等规模标量广播路径；
- 通用同形状路径；
- 通用广播路径；
- 多核并行和自适应分块处理。

## 7. 目录结构

~~~text
.
├── README.md
├── TEST_RESULTS.md
├── build.sh
└── src
    ├── CMakeLists.txt
    ├── op_host
    │   ├── CMakeLists.txt
    │   └── less_equal.cpp
    └── op_kernel
        ├── CMakeLists.txt
        ├── less_equal.cpp
        ├── less_equal_tiling.h
        └── tiling_key_less_equal.h
~~~

## 8. 构建方法

进入本项目目录后执行：

~~~bash
chmod +x build.sh
./build.sh
~~~

脚本会尝试自动加载 CANN 环境，并在当前目录下创建 `build` 目录。

也可以提前手动加载环境：

~~~bash
source ~/Ascend/cann-9.0.0/set_env.sh
./build.sh
~~~

需要保留已有构建目录进行增量构建时：

~~~bash
CLEAN_BUILD=0 ./build.sh
~~~

## 9. 测试结果

课程评测的 5 个测试点全部通过：

| 测试点 | 结果 | 输出错误占比 | 用时 |
| --- | --- | ---: | ---: |
| 1 | Pass | 0.00% | 3.30 μs |
| 2 | Pass | 0.00% | 6.36 μs |
| 3 | Pass | 0.00% | 13.44 μs |
| 4 | Pass | 0.00% | 3.76 μs |
| 5 | Pass | 0.00% | 18.34 μs |

汇总结果：

- 通过测试点：5 / 5
- 最大输出错误占比：0.00%
- 平均用时：9.04 μs

完整记录见 [TEST_RESULTS.md](TEST_RESULTS.md)。

## 10. 说明

本目录仅包含 LessEqual 算子结营作业所需的源码、构建脚本和验证结果。

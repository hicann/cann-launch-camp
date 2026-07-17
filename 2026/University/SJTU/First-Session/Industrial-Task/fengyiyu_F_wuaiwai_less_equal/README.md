# LessEqual Ascend C 自定义算子

## 项目简介

本项目基于 Ascend C 实现 LessEqual 自定义算子，对两个输入张量逐元素执行
`x1 <= x2` 比较，并输出 `bool` 类型结果。实现面向 `ascend910b`，支持相同
shape 的快速路径以及 NumPy/TensorFlow 风格的广播路径。

## 算子信息

| 项目 | 说明 |
| --- | --- |
| 算子名称 | `LessEqual` |
| 输入 | `x1`、`x2`，数据类型必须相同 |
| 输入类型 | `float16`、`float32`、`int32`、`int8` |
| 输出 | `y`，数据类型为 `bool` |
| 数据格式 | `ND` |
| 目标硬件 | `ascend910b` |
| 广播语义 | NumPy/TensorFlow 标准广播规则 |

数学定义：

```text
y[i] = (x1[i] <= x2[i])
```

## 目录结构

```text
fengyiyu_F_wuaiwai_less_equal/
├── README.md
├── build.sh
└── code/
    ├── CMakeLists.txt
    ├── op_host/
    │   ├── CMakeLists.txt
    │   └── less_equal.cpp
    └── op_kernel/
        ├── CMakeLists.txt
        ├── less_equal.cpp
        ├── less_equal_tiling.h
        └── tiling_key_less_equal.h
```

主要文件说明：

- `code/op_host/less_equal.cpp`：算子注册、shape/dtype 推导和 tiling 计算。
- `code/op_kernel/less_equal.cpp`：Ascend C Kernel，实现普通比较和广播比较。
- `code/op_kernel/less_equal_tiling.h`：Host 与 Kernel 共享的 tiling 数据结构。
- `code/op_kernel/tiling_key_less_equal.h`：不同输入类型的模板参数配置。
- `build.sh`：在 Linux CANN 环境中配置并编译算子工程。

## 实现说明

- 相同 shape 的输入使用连续访存快速路径，并按 AI Vector Core 数量分块。
- 不同 shape 的输入先在 Host 侧校验广播兼容性、计算步长并合并连续维度，
  Kernel 侧按广播后的行处理数据。
- `float16` 和 `float32` 直接比较；`int8` 无损转换为 `half` 后比较；
  `int32` 使用 `min(x1, x2) == x1` 实现精确的小于等于比较。
- 尾块按 256 个元素向上对齐计算，仅将真实元素写回 GM，以兼容非对齐 shape。
- 输出以 `int8` 的 `0/1` 表示 `bool` 的 `False/True`。

## 平台测验结果

本实现已完成在线平台测验，结果如下：

| 排名 | 平台账号 | 测验时间 | 得分 |
| ---: | --- | --- | ---: |
| 8 | `waiwai1019` | 2026/07/11 10:48:35 | 74.88 |

测试点五项耗时为：`2.82 μs`、`6.30 μs`、
`12.90 μs`、`2.98 μs`、`17.26 μs`。

## 构建环境

- Linux
- CANN Toolkit（包含 Ascend C 编译工具链和 `ASCConfig.cmake`）
- CMake 3.16 或更高版本
- 支持 `ascend910b` 的昇腾开发环境

默认的 CANN 环境脚本路径为：

```text
/usr/local/Ascend/ascend-toolkit/set_env.sh
```

如果安装位置不同，可以通过 `CANN_ENV_SCRIPT` 指定实际路径。

## 构建方法

进入本目录后执行：

```bash
bash build.sh
```

自定义 CANN 环境脚本或并行编译任务数：

```bash
CANN_ENV_SCRIPT=/path/to/set_env.sh BUILD_JOBS=8 bash build.sh
```

脚本等价于以下手动步骤：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S code -B code/build
cmake --build code/build --parallel 8
```

构建产物位于 `code/build/`。重新构建前如需清理缓存，可手动删除该目录。

## 验证建议

在具备 Ascend 910B 和 CANN 的环境中，建议至少覆盖以下场景，并将输出与
TensorFlow `tf.math.less_equal` 或 NumPy `less_equal` 逐元素比较：

1. `float16`、`float32`、`int32`、`int8` 四种输入类型；
2. 相同 shape 的一维、多维及非 32 字节对齐输入；
3. 标量与张量、向量与矩阵、高维张量之间的广播；
4. 两个输入相等、数据类型最小值和最大值等边界值；
5. 空张量以及题目约束内的大尺寸输入。

LessEqual 的比较结果要求完全一致，建议验证条件为：

```python
assert np.array_equal(actual, np.less_equal(x1, x2))
```

# LessEqual：一次证据驱动的 Ascend C 算子优化实践

> 作者：niziyi（GitCode：`m0_73993426`）
>
> 平台：Ascend 910B / CANN 9.0.1
>
> 最终版本：V35B
> 最终成绩：79.23 分，第 1 名

## 项目简介

本项目使用 Ascend C 实现 TensorFlow `tf.math.less_equal` 的核心语义：

```text
y[i] = (x1[i] <= x2[i])
```

实现支持 `float16`、`float32`、`int32`、`int8` 四种输入类型，输出为 `bool`，覆盖 NumPy/TensorFlow 广播、空张量和非 32 字节对齐场景。优化过程并非一次性写出“快代码”，而是在封闭评测中持续建立假设、拆分变量、验证回归，最终形成通用正确路径与热点专用路径并存的实现。

## 最终性能

V35B 使用完全相同的代码进行了两次评测。第二次评测获得最终成绩：

| 测试点 | 首次 V35B | 重复评测（最终） |
|---|---:|---:|
| TP1 | 2.08 μs | **2.14 μs** |
| TP2 | 5.30 μs | **4.52 μs** |
| TP3 | 11.38 μs | **10.16 μs** |
| TP4 | 2.40 μs | **2.36 μs** |
| TP5 | 16.60 μs | **15.78 μs** |
| 分数 | 73.38 | **79.23** |

两次结果的差异也构成了本项目的重要结论：微秒级封闭评测不能只看一次提交，必须结合代码路径、单变量实验和同字节重复测量判断优化是否真实。

## 最终架构

Host 侧首先完成广播合法性检查、相邻广播维合并、stride 生成和运行模式识别，再根据数据类型、形状、规模和每核工作量选择 Kernel 调度二进制。

### 广播运行模式

| 模式 | 典型场景 | 核心策略 |
|---|---|---|
| `LINEAR` | 同形状或合并后的一维输出 | 连续分块，减少索引元数据 |
| `ROW` | 一般高维广播 | 按完整内层行处理，里程表推进外层坐标 |
| `REUSE_X1` | `[1,N]` 对 `[M,N]` | x1 的列 Tile 驻留 UB，跨行复用 |
| `REUSE_X2` | `[M,N]` 对 `[1,N]` | x2 的列 Tile 驻留 UB，跨行复用 |

### 编译期调度模式

| TilingKey | 用途 |
|---|---|
| `SCH_NORMAL` | 单缓冲通用路径，保护小任务固定开销 |
| `SCH_LINEAR_DB_ONES` | 多 Tile LINEAR 双缓冲，并预生成全 1 Tensor |
| `SCH_FAST_LIGHT` | 严格同形状 INT8、129–4096 元素的轻量单段路径 |
| `SCH_STATIC_HALF` | 严格同形状 FP16、129–16384 元素的静态 UB 路径 |

四种调度值使用 2-bit UINT 模板字段。Host 只选择与场景匹配的二进制，避免通用广播逻辑、双缓冲状态和热点专用代码相互拖累。

### 数据类型路径

| 类型 | 比较实现 |
|---|---|
| `float16` | 原生 `Compare(..., LE)`；热点范围使用静态 FP16 Kernel |
| `float32` | 原生 `Compare(..., LE)` |
| `int8` | 精确 Cast 到 half 后比较；小型同形状场景使用 FAST_LIGHT |
| `int32` | `Max(x1, x2) == x2`，避免有损浮点转换 |

比较结果先生成 bit mask，再经 `Select` 与 `Cast` 写出 `uint8_t` 表示的 bool。

## 目录结构

```text
.
├── README.md
├── code/
│   ├── CMakeLists.txt
│   ├── op_host/
│   │   ├── CMakeLists.txt
│   │   └── less_equal.cpp
│   └── op_kernel/
│       ├── CMakeLists.txt
│       ├── less_equal.cpp
│       ├── less_equal_tiling.h
│       └── tiling_key_less_equal.h
└── docs/
    ├── OPTIMIZATION_JOURNEY.md
    ├── TECHNICAL_GUIDE.md
    ├── MAINTENANCE_GUIDE.md
    ├── RL_EXPLORATION.md
    └── CLOSED_BENCHMARK_REFLECTION.md
```

## 构建

先加载 CANN 环境，再从 `code/` 配置构建。示例：

```bash
source /home/nzy/Ascend/cann-9.0.1/set_env.sh
cmake -S code -B build \
  -DASC_DIR=/home/nzy/Ascend/cann-9.0.1/lib64/cmake
cmake --build build -j8
```

项目的 `CMakeLists.txt` 将目标计算单元固定为 `ascend910b`。完整构建应包含 Host tiling、四种 dtype 的 OPC Kernel、Kernel 打包和 `libcust_opapi.so`。

## 文档导航

- [优化历程：从正确性基线到 V35B](./docs/OPTIMIZATION_JOURNEY.md)
- [最终实现的技术说明](./docs/TECHNICAL_GUIDE.md)
- [冻结版本与维护指南](./docs/MAINTENANCE_GUIDE.md)
- [自动化黑盒搜索与 RL 探索](./docs/RL_EXPLORATION.md)
- [关于封闭评测、教学与真实工作负载的思考](./docs/CLOSED_BENCHMARK_REFLECTION.md)

## 一点总结

这个项目最有价值的部分，不只是最后一次排名。真正可复用的经验是：先守住正确性，再用最小实验回答一个问题；保护已经赢下的路径；面对噪声时相信可解释的证据，而不是相信一次漂亮的数字。

## 作者

- 姓名：niziyi
- 邮箱：3311750378@qq.com
- GitCode：`m0_73993426`

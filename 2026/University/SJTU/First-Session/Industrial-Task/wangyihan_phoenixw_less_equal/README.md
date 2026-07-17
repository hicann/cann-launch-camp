# LessEqual Ascend C 算子实现

## 功能说明

本项目基于 Ascend C 实现 TensorFlow `tf.math.less_equal` 的核心语义，对两个输入张量逐元素执行：

```text
y = x1 <= x2
```

输出类型固定为 `bool`，设备内存中使用 `0/1` 表示 `false/true`。

支持：

- `float16`、`float32`、`int32`、`int8`
- ND 格式和任意维张量
- NumPy/TensorFlow 风格广播
- 非 32 字节对齐长度
- 空张量及空维广播
- 多核并行和向量化计算

两个输入的数据类型必须相同；不能广播的 shape 会在 Host 侧返回失败。

## 算法实现

### 同形状路径

输入连续时采用 `CopyIn -> Compute -> CopyOut` 流水：

```text
GM(x1/x2) -> UB -> 向量比较 -> bool 结果 -> GM(y)
```

不同数据类型的计算策略如下：

| 输入类型 | 向量计算策略 |
|---|---|
| `float16` | `Compare(LE)` |
| `float32` | `Compare(LE)` |
| `int32` | `Min(x1, x2)` 后执行 `Compare(x1, min, EQ)` |
| `int8` | 向量 `Cast` 为 `float16` 后执行 `Compare(LE)` |

`Compare` 输出压缩 bitmask，随后通过向量 `Select + Cast` 生成逐字节 bool 输出。无法组成完整向量 repeat 的尾部使用精确标量处理。

### 广播路径

Host 侧从右向左推导广播 shape 和 stride，不实际扩展输入张量。相邻的连续维度或共同广播维度会被合并，以减少 Kernel 侧 rank。

Kernel 按输出最后一维分块：

- 连续输入使用 `DataCopyPad` 搬入 UB；
- 广播维为 1 时搬入一个标量，并使用 `Duplicate` 在 UB 中展开；
- 每核仅在起始位置计算一次多维坐标，后续行通过 stride 增量和进位更新地址。

### 模板化与性能优化

- 使用 TilingKey 按 dtype、同形状/广播、小张量/向量路径生成专用 Kernel；
- 输入和输出队列使用深度为 2 的双缓冲，支持 DMA 与 Vector 流水重叠；
- 根据 UB 容量、dtype 和临时缓冲占用动态计算 tile；
- 小张量使用单核低延迟路径，避免不必要的 UB、DMA 和向量掩码转换；
- 广播按行内 tile 分核，避免长单行张量只使用一个 AI Core；
- TilingData 仅保留输出 shape 和两输入 stride，减少冗余数据和 Scalar 压力。

## 项目结构

```text
.
├── README.md
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
```

## 构建方式

首先加载 CANN 环境，路径请按实际安装位置调整：

```bash
source /path/to/Ascend/cann/set_env.sh
```

编译算子：

```bash
cd src
cmake -S . -B build
cmake --build build -j4
```

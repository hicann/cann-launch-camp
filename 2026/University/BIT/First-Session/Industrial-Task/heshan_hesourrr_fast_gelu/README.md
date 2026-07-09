# FastGeluCustom Ascend C 自定义算子

## 1. 项目简介

本项目在昇腾 NPU 上使用 Ascend C 实现自定义 `FastGelu` 算子，面向 `float16` 和 `float32` 输入，输出 shape 与 dtype 均与输入保持一致。算子采用逐元素计算方式，支持 ND 格式输入，可覆盖 1D、2D、3D、4D 及更高维张量。

当前实现重点兼顾：

- 输出正确性；
- `float16` 精度；
- 中大规模张量吞吐；
- 小规模张量启动开销控制；
- 代码结构简洁，便于提交和维护。

## 2. 算子信息

| 项目 | 内容 |
|---|---|
| 算子名称 | `FastGelu` |
| 输入 | `x` |
| 输出 | `y` |
| 输入 dtype | `float16`, `float32` |
| 输出 dtype | 与输入一致 |
| 输入 format | `ND` |
| 输出 format | `ND` |
| 目标硬件 | `ascend910b` |
| workspace | `0` |

## 3. 数学公式

题目给出的 FastGelu 形式为：

```text
y = x * exp(0.851 * (x - abs(x))) / (1 + exp(-1.702 * abs(x)))
```

本实现采用等价化简形式：

```text
y = x / (1 + exp(-1.702 * x))
```

该形式可以减少算子内部的向量计算链，核心计算只需要：

```text
tmp = -1.702 * x
tmp = exp(tmp)
tmp = tmp + 1
y   = x / tmp
```

与稳定原式相比，该版本少用了 `Abs`、`Sub`、第二次 `Exp` 以及额外 `Mul`，因此计算链更短，适合追求运行速率的测试场景。

## 4. 目录结构

```text
FastGeluCustom/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── CMakeLists.txt
│   └── fast_gelu.cpp
└── op_kernel/
    ├── CMakeLists.txt
    ├── fast_gelu.cpp
    ├── fast_gelu_tiling.h
    └── tiling_key_fast_gelu.h
```

各文件作用：

| 文件 | 作用 |
|---|---|
| `CMakeLists.txt` | 工程总入口，设置 `ascend910b`，添加 Host 与 Kernel 子目录 |
| `op_host/fast_gelu.cpp` | Host 侧算子注册、shape 推导、dtype 推导、tiling 计算 |
| `op_kernel/fast_gelu.cpp` | Kernel 侧 Ascend C 计算逻辑 |
| `op_kernel/fast_gelu_tiling.h` | Host 与 Kernel 共享的 tiling 数据结构 |
| `op_kernel/tiling_key_fast_gelu.h` | dtype 模板 tiling key，生成 `float16` / `float32` 两套 Kernel |

## 5. 实现特点

### 5.1 模板 TilingKey 区分 dtype

Host 侧通过：

```cpp
ASCENDC_TPL_SEL_PARAM(context, DT_X);
```

Kernel 侧通过：

```cpp
template <typename DT_X>
__global__ __aicore__ void fast_gelu(...)
```

让平台在编译期生成 `float16` 和 `float32` 两个 Kernel 实例。这样可以避免运行期 dtype 分支，使 `if constexpr` 在编译期裁剪无关路径。

### 5.2 half 输入使用 float 中间计算

`float16` 分支先将输入 cast 到 `float32`：

```cpp
Cast(xf, xLocal, RoundMode::CAST_NONE, len);
```

之后在 float 中完成 `Muls`、`Exp`、`Adds`、`Div`，最后 cast 回 half。这样比直接 half 计算更稳，尤其对隐藏点中的小数值、负数和精度边界更有利。

### 5.3 按最小元素数控制启动核数

Host 侧使用：

```cpp
MIN_ELEMENTS_PER_CORE = 64
```

当张量较小时，不会盲目启动所有 AIV 核，减少小规模输入下的调度和流水初始化开销。

### 5.4 blockLength 按 32B 对齐

Host 侧根据 dtype 计算：

```cpp
alignNum = 32 / typeSize
blockLength = AlignUp(ceil(length / blockDim), alignNum)
```

这样每个核处理的起始位置更规整，有利于 GM 到 UB 的搬运，并且兼容非 32B 对齐的尾块。

### 5.5 Double Buffer 与预取下一 tile

Kernel 使用：

```cpp
BUFFER_NUM = 2
```

并在处理当前 tile 前预取下一 tile：

```cpp
CopyIn(nextOffset, nextLen);
```

整体流程近似为：

```text
预取第 0 块
循环：预取下一块 -> 计算当前块 -> 写回当前块
```

这种方式可以降低 DMA 搬运等待对 Vector 计算的影响。

### 5.6 不同 dtype 使用不同 TILE_LENGTH

Kernel 根据 dtype 在编译期选择 tile 大小：

```cpp
TILE_LENGTH = IsSameType<T, half>::value ? 10240 : 6144;
```

`half` 每个元素 2 字节，可以使用更大的 tile；`float` 每个元素 4 字节，tile 略小，避免 UB 压力过高。

## 6. 编译方法

进入工程根目录后执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

rm -rf build
mkdir build
cd build
cmake ..
make -j8
```

如果平台 CANN 路径不同，先查找环境脚本：

```bash
find /usr/local/Ascend -name set_env.sh
```

然后替换 `source` 路径。

## 7. 编译成功检查

编译完成后可检查生成文件：

```bash
find . -name "*.so" -o -name "*.run" -o -name "*.json"
```

比赛平台通常只需要工程编译通过，然后直接提交评测。

## 8. 常见问题

### 8.1 `Could not find ASC`

CANN 环境变量没有加载。执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

然后重新清理 build 编译。

### 8.2 `fast_gelu_x_tilingkey does not name a template`

说明 Kernel 入口和 `tiling_key_fast_gelu.h` 的模板机制不匹配。当前版本要求保留：

```cpp
template <typename DT_X>
__global__ __aicore__ void fast_gelu(...)
```

同时保留 `tiling_key_fast_gelu.h` 中的 `ASCENDC_TPL_ARGS_DECL` 和 `ASCENDC_TPL_SEL`。

### 8.3 输出结果异常

优先检查：

1. 是否清理了旧的 `build`；
2. 是否使用了当前版本所有文件；
3. `fast_gelu_tiling.h` 中的结构体是否与 Host / Kernel 完全一致；
4. `op_host/fast_gelu.cpp` 是否调用了 `ASCENDC_TPL_SEL_PARAM(context, DT_X)`。

建议每次替换文件后执行：

```bash
rm -rf build
mkdir build
cd build
cmake ..
make -j8
```

## 9. 版本备注

该版本偏向性能优化：使用等价化简公式减少计算链，并对 `float16` 使用 float 中间计算保证精度。若测试数据包含极大负数且平台严格检查 `Inf/NaN` 行为，可考虑将 Kernel 的 `Compute` 部分切换回题目给出的稳定原式。

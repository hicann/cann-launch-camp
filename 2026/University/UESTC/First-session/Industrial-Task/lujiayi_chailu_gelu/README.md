# GELU 自定义算子说明文档

## 1. 项目简介

本工程实现华为昇腾 Ascend C 自定义 GELU 算子，算子名为 `Gelu`，支持 `float16` 和 `float32` 两种输入输出数据类型，输入输出格式为 `ND`。

GELU 的参考计算公式为：

```text
GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

本版本基于 `v28_hgelu_fexact_4096`，采用按数据类型分流的实现策略：

```text
float16 路径：使用 AscendC 高阶 Gelu API，并显式提供 sharedTmpBuffer。
float32 路径：使用 exact erf 公式展开，保证与 PyTorch 默认 GELU 对齐。
```

该设计的目标是在保证正确性的前提下，尽量提升小规模 `float16` 测试点的速度，同时保持 `float32` 测试点的精度稳定。

## 2. 工程目录结构

```text
.
├── CMakeLists.txt
├── README.md
├── op_host
│   ├── CMakeLists.txt
│   └── gelu.cpp
└── op_kernel
    ├── CMakeLists.txt
    ├── gelu.cpp
    ├── gelu_tiling.h
    └── tiling_key_gelu.h
```

各文件作用如下：

```text
op_host/gelu.cpp
    Host 侧代码，负责算子原型注册、shape 推导、dtype 推导、tiling 参数计算和 blockDim 设置。

op_kernel/gelu.cpp
    Kernel 侧代码，负责 Global Memory 到 Unified Buffer 的搬运、GELU 计算和结果写回。

op_kernel/gelu_tiling.h
    TilingData 结构体定义，保存 totalLength、tileLength 和 tmpSize。

op_kernel/tiling_key_gelu.h
    TilingKey 模板定义，用于区分 float16 和 float32 两种 kernel 实例。

CMakeLists.txt
    算子工程构建入口。
```

## 3. 支持范围

```text
算子名称：Gelu
输入：input_x
输出：output
支持 dtype：float16、float32
支持 format：ND
目标平台：ascend910b
```

输出 shape 与输入 shape 完全一致，输出 dtype 与输入 dtype 完全一致。

## 4. 核心实现思路

### 4.1 Host 侧 tiling

Host 侧首先获取输入张量元素总数 `totalLength` 和输入数据类型，然后通过 `ASCENDC_TPL_SEL_PARAM` 选择对应 dtype 的模板实例。

关键参数：

```cpp
static constexpr uint32_t TILE_LENGTH = 4096;
```

`TILE_LENGTH = 4096` 是根据多轮平台实测得到的当前较优 tile 粒度。该粒度可以减少高阶数学 API 的调用次数，同时避免过大的 UB 压力。

Host 侧还会根据总长度估计需要使用的 AIV core 数：

```text
needCore = ceil(totalLength / TILE_LENGTH)
usedCore = min(needCore, numCoresAiv)
```

这样可以在小 shape 下减少空跑 core，在大 shape 下尽量利用 AIV 并行能力。

### 4.2 Kernel 侧分核和分块

Kernel 侧根据 `totalLength`、`tileLength` 和实际 `blockNum` 计算每个 core 负责的 tile 范围。

每个 core 处理自己的连续数据区间，主循环按 tile 处理：

```text
CopyIn  ->  Compute  ->  CopyOut
```

完整 tile 使用普通 `DataCopy`，非完整尾块使用 `DataCopyPad`，保证非 32 字节对齐场景下也能正确搬运。

### 4.3 float16 计算路径

当输入类型为 `float16` 时，kernel 使用 AscendC 高阶 `Gelu` API：

```cpp
AscendC::Gelu<T, false, false>(yLocal, xLocal, tmpLocal, calLength);
```

其中 `tmpLocal` 来自 `TBuf<VECCALC>`，作为官方高阶 API 的 shared temporary buffer。尾块计算长度会按 32B 对齐粒度向上取整，写回时仍只写真实有效长度。

### 4.4 float32 计算路径

当输入类型为 `float32` 时，kernel 使用 exact erf 展开公式：

```text
x / sqrt(2)
erf(x / sqrt(2))
1 + erf(x / sqrt(2))
0.5 * (1 + erf(x / sqrt(2)))
x * 0.5 * (1 + erf(x / sqrt(2)))
```

该路径避免使用官方 `Gelu` 的近似计算结果，优先保证 `float32` 精度与 PyTorch 默认 GELU 对齐。

## 5. 构建与提交方式

在比赛平台中，将本目录下的全部内容复制到提交目录，确保提交目录至少包含：

```text
CMakeLists.txt
README.md
op_host/
op_kernel/
```

平台会自动执行 CMake 和 Ascend C 编译流程。若在本地环境编译，可参考：

```bash
mkdir -p build
cd build
cmake ..
make -j
```

实际命令以比赛平台提供的构建脚本为准。

## 6. 参数说明

当前主要可调参数位于：

```text
op_host/gelu.cpp
```

```cpp
static constexpr uint32_t TILE_LENGTH = 4096;
uint32_t selectedTmpSize = 131072U;
```

参数含义：

```text
TILE_LENGTH
    每次 kernel 内处理的元素个数。当前固定为 4096。

selectedTmpSize
    float16 路径中 AscendC::Gelu 高阶 API 使用的 sharedTmpBuffer 字节数。
```

当前版本建议保持上述参数不变。

## 7. 优化点总结

本版本主要包含以下优化：

```text
1. 使用 TilingKey 模板区分 float16 / float32，避免运行时类型判断。
2. 小 shape 动态减少 blockDim，减少空核开销。
3. tileLength 固定为实测较优的 4096。
4. 完整 tile 使用 DataCopy，尾块使用 DataCopyPad。
5. float16 使用官方高阶 Gelu API，提高小规模 case 性能。
6. float32 使用 exact erf 公式，保证 PyTorch 默认 GELU 精度。
7. 使用 TBuf 管理临时计算空间，避免不必要的 Global Memory 中转。
```

## 8. 注意事项

```text
1. 本工程不包含 printf、cout、DumpTensor 等调试输出代码。
2. 不建议在未重新验证的情况下修改 TILE_LENGTH。
3. 不建议将 float32 路径直接替换为官方 Gelu API，否则可能出现精度不匹配。
4. sharedTmpBuffer 不能与输入或输出 LocalTensor 地址重叠。
5. 尾块写回只写真实 validLength，避免 padded 数据污染输出。
```

## 9. 版本说明

```text
版本：v28_hgelu_fexact_4096
基线：v16_tile4096_reuse_y_1tmp
主要变化：float16 路径切换为 AscendC 高阶 Gelu API；float32 路径保留 exact erf 公式。
```

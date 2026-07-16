# LessEqual 算子（Ascend C）

对标 `tf.math.less_equal`：逐元素比较 `x1 <= x2`，输出 `bool`。在昇腾 910B (ascend910b) 上用 Ascend C 原生开发。

> 源码位于 `src/` 子目录；下文「构建与测试」中的 `bash build.sh` 等命令请在 `src/` 下执行。

## 功能

- **输入 dtype**：float16 / float32 / int32 / int8（两输入 dtype 必须一致）
- **输出**：bool（1 字节 int8 存储）
- **形状**：任意多维；支持非 32 字节对齐；支持空张量
- **广播**：完整 NumPy 广播语义（标量×张量、向量×矩阵、任意多维不同形状、双向广播）
- **精度**：比较结果完全精确（含各 dtype 极值边界）

## 实现要点

计算路径复刻 CANN 内置 LessEqual：`Compare(CMPMODE::LE) → Select(0/1) → 输出 bool`。

- **两条执行路径**（Host 侧 Tiling 判定）：
  - *Fast path*：x1、x2 同形状，1D 展平，多核 + UB 分块 + Double Buffer 向量化。
  - *Broadcast path*：形状不同，按输出「行」（最后一维）处理；Host 预计算广播 stride（被广播维=0），Kernel 用混合进制计数器推进行索引（无除法/取模）；最后一维广播时读单元素并填充整行。
- **dtype 适配**：
  - float16 / float32：原生 `Compare`。
  - int8：`Compare` 不支持 → 先 `Cast` 到 half 再比较。
  - int32：硬件 `Compare` 仅支持 EQ → 用恒等式 `(a<=b) == (min(a,b)==a)`，`Min` + `Compare(EQ)` 对 int32 均精确。
  - Select 统一用 half 常量产出 0/1，再 `Cast` 到 int8 输出（`half→int8` 有直接通路）。
- **count 约束**：`Compare/Select` 要求 `count*sizeof` 为 256B 整数倍，故按 256 元素对齐（输出只写回有效长度）。

## 目录

```
op_host/less_equal.cpp        # InferShape(广播) / InferDataType(bool) / TilingFunc
op_kernel/less_equal.cpp      # KernelLessEqual<T> 模板核函数
op_kernel/less_equal_tiling.h # TilingData 结构
op_kernel/tiling_key_less_equal.h # DT_X1 四类型 TilingKey
test/less_equal_test.cpp      # aclnn 二段式接口调用 + CPU golden 对拍（33 用例）
build.sh                      # 编译 → 编译测试 → 运行 一键脚本
CMakeLists.txt / op_host / op_kernel 下的 CMakeLists.txt 为工程原始脚手架，未改动
```

## 构建与测试（昇腾服务器）

```bash
source ~/Ascend/ascend-toolkit/set_env.sh
bash build.sh
```

脚本流程（不修改任何 CMakeLists，使用 SHARED 构建产物）：
1. `cmake + make + make binary` 编译算子与 kernel 二进制；
2. 直接 `g++` 编译测试程序，链接 `build/libcust_opapi.so`；
3. 设置 `ASCEND_CUSTOM_OPP_PATH=build/tmp/vendors/custom` 后运行 33 个用例
   （多 dtype / 多维 / 非对齐 / 空张量 / 广播 / 边界极值）。

当前结果：**PASS=33 FAIL=0**。

> 注意：默认 `make` 不编译 kernel 二进制，必须执行 `make binary`（build.sh 已包含）。
> SHARED 构建在 `build/tmp/vendors/custom` 下生成完整算子包（op_impl + kernel + op_info），
> 无需额外 `.run` 安装，测试时通过 `ASCEND_CUSTOM_OPP_PATH` 指向该目录即可。

## 性能优化

用 `msprof op` 采集 device 侧 kernel 时间（`test/less_equal_prof.cpp` 为采集入口）。
相比初版的优化点：

1. **常量 0/1 提到 Init**：Select 用的 `1.0/0.0` half 常量只在 `Init` 中 `Duplicate` 一次，
   移出逐 tile 循环，减少每 tile 两条向量指令。
2. **消除多余 Cast**：fp32/int32 的掩码用 **half** 常量做 Select（half→int8 有直接 Cast 通路），
   去掉了此前 float→half→int8 的双次 Cast 与额外 buffer。
3. **小张量核数自适应**：`blockDim = min(核数, ceil(total/256))`，小张量不再无谓拉起 40 核、
   也避免每核不足 256 时被向量粒度放大的冗余计算。
4. **广播行缓存**：当某操作数跨「行」不变（外层 stride 全 0）且整行装得下一个 tile 时，
   该操作数一行只从 GM 读一次并在本核内跨行复用，消除逐行重复搬运（广播主要瓶颈）。

小张量（延迟敏感）device 时间约提升 2×（如一维 fp32 2048：5.44μs → ~2.5μs；
一维 int32 4096：6.60μs → ~3.2μs），大张量亦有 10%+ 改善，
广播场景约 3.5×（1024²vs1024：41.7μs → ~11.8μs）。

> 完整的踩坑、报错定位与调优过程见 [OPTIMIZATION_JOURNEY.md](OPTIMIZATION_JOURNEY.md)。

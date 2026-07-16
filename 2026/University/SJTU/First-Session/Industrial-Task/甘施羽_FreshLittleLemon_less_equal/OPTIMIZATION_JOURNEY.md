# LessEqual 算子开发与调优心路历程

> 记录用 Ascend C 在昇腾 910B3 上实现 `tf.math.less_equal` 算子的完整过程：从踩坑、
> 报错定位，到用 profiling 驱动的性能调优。所有结论均在 CANN 8.5.2 服务器上实测验证。

---

## 一、背景与目标

- **题目**：逐元素比较 `x1 <= x2`，输出 bool。支持 float16/float32/int32/int8、任意多维、
  非 32 字节对齐、完整 NumPy 广播。
- **参考**：CANN 内置 LessEqual 计算图 `Compare(CMPMODE::LE) → Select(0/1) → 输出`。
- **评分**：竞赛按 5 个测试点的 **device 侧 kernel 时间**（μs）打分。

初版思路：直调（非 DAG 框架）复刻 `Compare → Select → Cast` 三段式，两条路径
（等形状 fast path / 广播 path），dtype 用模板区分。

---

## 二、报错调试心路（正确性阶段）

这一阶段的核心教训是：**Ascend C 的很多约束不写在文档里，要去读 toolkit 头文件的
`ASCENDC_ASSERT` 和 impl 源码**。按报错顺序复盘：

### 1. `AlignUp` 函数名冲突
- **现象**：`call to 'AlignUp' is ambiguous`。
- **原因**：toolkit 的 `kernel_utils_ceil_oom_que.h` 已有全局 `AlignUp`，与我的 helper 撞名。
- **修复**：所有 helper 加 `Le` 前缀（`LeAlignUp/LeCeilDiv/LeMin`）。
- **教训**：kernel 命名空间里全局符号很多，自定义工具函数一律加算子前缀。

### 2. `Cast<half,half>` 与 `Cast<float,int8>` 无匹配函数
- **现象**：`no matching function for call to 'CastIntrinsicsImpl'`。
- **原因一**：运行时 `if (IsSameType...)` 两个分支**都会被编译**，导致 int8 特化里
  生成了非法的 `Cast<half,half>`。→ 改用 `if constexpr` 编译期裁剪。
- **原因二**：硬件**不支持 float→int8 直接 Cast**，需经 half 中转。
- **教训**：模板核里凡是按类型分流，一律用 `if constexpr`；Cast 的合法通路要查
  `kernel_operator_vec_vconv_impl.h` 里的重载。

### 3. `Duplicate` 不支持 int8
- **现象**：`static assertion failed ... Duplicate ... half/bfloat16/int16/uint16/int32/uint32/float`。
- **原因**：广播维填充整行用了 `Duplicate`，但它不支持 int8。
- **修复**：int8 走标量 `SetValue` 循环填充（`FillVal` 里 `if constexpr` 分流）。

### 4. 小张量结果错乱（最隐蔽的一个）
- **现象**：`n<64` 的用例大面积 mismatch，大张量却对。单核实验发现凡 `n` 非某对齐值就错。
- **定位**：读 `kernel_operator_vec_cmpsel_intf_impl.h` 第 180 行——
  `Compare` level-2 要求 **`count * sizeof(T) % 256 == 0`**。我原先只对齐到 32。
- **修复**：`cnt` 对齐到 256 元素（各 dtype 都满足），输出只写回有效 `n` 个，尾部垃圾无害。
- **教训**：vector API 的 count 有 256B 对齐硬约束，这是"小张量专属 bug"的常见根因。

### 5. 广播路径读到脏数据（流水线竞争）
- **现象**：广播用例前几个元素就错。
- **原因**：在 `EnQue` **之前**就 `GetValue` 读单元素——数据还没搬到。
- **修复**：严格遵循 `Alloc → CopyIn → EnQue → DeQue → (此后才能)GetValue` 的顺序。
- **教训**：`GetValue` 只能在 `DeQue` 之后调用，这是 TQue 流水线的硬性时序。

### 6. int32 比较结果错（只有 int32 挂）
- **现象**：fp16/fp32/int8 全过，唯独 int32 的 LE 结果错。
- **定位**：读 `kernel_operator_vec_cmp_impl.h` 第 127 行——
  **int32 硬件 Compare 只支持 `CMPMODE::EQ`，不支持 LE**。
- **修复**：用恒等式 `(a<=b) ⇔ (min(a,b)==a)`，`Min`+`Compare(EQ)` 对 int32 都精确。
- **教训**：不同 dtype 的同一 API 支持的模式不同，别假设 LE 对所有类型可用。

### 7. UB 溢出（大 int32 崩溃）
- **现象**：`large_i32_9999` 在 Synchronize 崩。
- **原因**：int32 分支多用了一个 `minBuf`，我的 tiling `perElem` 估算没算进去，
  `tileLength` 偏大导致 UB 溢出。
- **修复**：精确核算每元素 UB 占用（含所有临时 buffer），`perElem` 留足余量。
- **教训**：tiling 的 UB 预算必须逐一枚举 kernel 里所有 `InitBuffer`，宁可保守。

**正确性阶段结论**：33 个用例（多 dtype × 多维 × 非对齐 × 空张量 × 广播 × 边界极值）全过。

---

## 三、性能调优心路（profiling 驱动）

原则：**不猜，用 `msprof op` 采集 device 时间和 PipeUtilization 定位瓶颈，改一处测一处。**

### Round 0：建立基线

用 `msprof op --output=... <app>` 采 `OpBasicInfo.csv` 的 `Task Duration(us)`。
初版 device 时间：

| 代表形状 | 初版 |
|---|---|
| 一维 fp32 2048 | 5.44μs |
| 一维 int32 4096 | 6.60μs |
| 大 fp32 1024² | 12.34μs |
| 大 fp16 2048² | 19.82μs |
| 广播 1024²vs1024 | 41.66μs |

排行榜对比发现：**小张量（测试点 1、4）差距最大（~3×）**，广播是绝对最慢项。

### Round 1：小张量与通用开销（三处改动）

1. **常量提到 Init**：Select 用的 `1.0/0.0` half 常量，之前**每个 tile** 都 `Duplicate`
   一次，移到 `Init` 只填一次。省掉每 tile 两条向量指令。
2. **消除多余 Cast**：初版为了"Select 位宽匹配 Compare 源"，fp32/int32 走了
   `float→half→int8` 双次 Cast。实测发现**误判**——int32 当初错是因为 LE→EQ（见报错6），
   与 Select 位宽无关。统一改用 **half 常量做 Select**（`half→int8` 一次 Cast 搞定），
   还省了两个 buffer。
3. **小张量核数自适应**：`blockDim = min(核数, ⌈total/256⌉)`。小张量不再无谓拉起 40 核，
   也避免每核不足 256 元素时被向量粒度放大的冗余计算。

结果：

| 形状 | 初版 | Round1 |
|---|---|---|
| 一维 fp32 2048 | 5.44 | **2.52** |
| 一维 int32 4096 | 6.60 | **3.22** |
| 大 fp32 | 12.34 | **10.88** |
| 大 fp16 | 19.82 | **17.56** |

小张量 ~2× 提升，一举追平/接近榜首。

### Round 2：用 PipeUtilization 找下一个瓶颈

采 `PipeUtilization.csv` 的 aiv 各流水占比（block0）：

| 形状 | vec_ratio | scalar_ratio | mte2_ratio | 结论 |
|---|---|---|---|---|
| t2 中等 fp16 | 0.06 | **0.61** | 0.41 | scalar/setup 受限，算法本身已紧 |
| t5 大张量 | 0.27 | 0.44 | **0.90** | **访存带宽受限**，接近硬件极限 |
| bc 广播 | 0.06 | 0.30 | **0.95** | **冗余搬运**——29μs 全耗在 MTE2 |

结论清晰：
- t5 已是带宽墙，不值得再投入；
- **广播是唯一有大空间的**——`[1024,1024] vs [1024]` 把那 1024 长的小操作数
  **逐行重复从 GM 搬了 1024 次**。

### Round 3：广播行缓存（最大单点收益）

洞察：广播时若某操作数在"行"维度不变（外层 stride 全 0），它**整行只需从 GM 读一次**，
本核内跨行复用即可。

实现 `ProcessBroadcastCached`：
- 判定条件 `OuterAllZero(stride)`（外层 stride 全 0）且整行装得下一个 tile；
- 不变操作数 `Alloc→CopyIn→EnQue→DeQue` 一次后**持有不 Free**，跨行复用；
- 变化操作数逐行加载；输出照常 `Compare→Select→Cast→CopyOut`；
- 不满足条件时回退到通用逐行路径（保证正确性覆盖）。

结果：**广播 41.66μs → 11.82μs（3.5×）**，其余用例无回归。

---

## 四、最终成绩

| 代表形状 | 初版 | 最终 | 榜首 |
|---|---|---|---|
| 一维 fp32 2048（≈测点1） | 5.44 | **2.52** | 2.02 |
| 一维 int32 4096（≈测点4） | 6.60 | **3.22** | 2.06 |
| 中 fp16（≈测点2） | 6.92 | **6.28** | 4.34 |
| 大 fp32（≈测点3） | 12.34 | **10.88** | 10.60 |
| 大 fp16（≈测点5） | 19.82 | **17.56** | 16.88 |
| 广播 | 41.66 | **11.82** | — |

正确性始终 **33/33 全过**。

---

## 五、方法论沉淀

1. **报错先读 toolkit 源码**：Ascend C 的类型/模式/对齐约束多藏在头文件的
   `ASCENDC_ASSERT`、impl 的 `SupportType<>` 和分支里，比查文档快且准。
2. **模板核用 `if constexpr`**：运行时 `if` 会编译所有分支，按 dtype 分流必用编译期裁剪。
3. **TQue 时序铁律**：`GetValue` 只能在 `DeQue` 之后；`Alloc→CopyIn→EnQue→DeQue`。
4. **count 的 256B 对齐**：小张量结果错的头号嫌疑；对齐 count、只写回有效长度。
5. **dtype 差异**：int32 Compare 只支持 EQ；float→int8 无直接 Cast；Duplicate 不支持 int8。
   遇到就用等价变换（min+eq / half 中转 / 标量填充）绕过。
6. **调优不猜、看 profiling**：
   - `OpBasicInfo.csv` → 总 device 时间与占用核数；
   - `PipeUtilization.csv` → vec/scalar/mte2/mte3 占比，一眼看出是算力、
     scalar setup 还是访存受限；
   - **带宽受限（mte2≈0.9）的用例不必再优化**，把精力投到冗余搬运/低占用的用例。
7. **改一处测一处**：每次只动一个变量，跑正确性 + profiling 回归，避免相互掩盖。

---

## 附：如何复现

```bash
source ~/Ascend/ascend-toolkit/set_env.sh
cd lessequal-claudecode && bash build.sh          # 编译 + 33 用例正确性

# 性能采集（device 时间）
B=$PWD/build; export ASCEND_CUSTOM_OPP_PATH=$B/tmp/vendors/custom
g++ -std=c++17 -O2 -o $B/le_prof test/less_equal_prof.cpp \
  -I$ASCEND_HOME_PATH/include -I$ASCEND_HOME_PATH/include/acl \
  -I$ASCEND_HOME_PATH/include/aclnn -I$B/autogen \
  -L$ASCEND_HOME_PATH/lib64 -L$B -lascendcl -lnnopbase -lcust_opapi -ldl \
  -Wl,-rpath,$ASCEND_HOME_PATH/lib64 -Wl,-rpath,$B
mkdir -p ~/prof && chmod 750 ~/prof
msprof op --output=~/prof "$B/le_prof t1"          # t1..t5, bc
```

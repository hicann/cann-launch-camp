# GELU 算子(Ascend C / Atlas A2 910B)——设计与实现说明

昇腾 910B(Atlas A2)上的 GELU 激活算子 Ascend C 实现。当前为经过完整实验筛选后的**最优稳定版本(6/6 全过)**。本文档面向后续维护者(人与 AI),说明算法、数据切分、双数据类型分路、精度保证、构建部署,并附上**已验证走不通的方向**,避免重复踩坑。

---

## 1. 功能与精度要求

GELU 定义:

```
GELU(x) = x · Φ(x) = x · 0.5 · (1 + erf(x / √2))
```

支持两种输入 dtype,逐元素精度要求:

| dtype | 相对误差 | 绝对误差 |
| :--- | :--- | :--- |
| float32 | < 1e-4 | < 1e-4 |
| float16 | < 1e-3 | < 1e-3 |

输入为任意 shape 的 ND 张量(动态 shape),输出 shape/dtype 与输入一致。

---

## 2. 目录结构

```
code/
├── op_host/
│   └── gelu.cpp            # Host: Tiling 计算 / InferShape / InferDataType / 算子注册
└── op_kernel/
    ├── gelu.cpp            # Kernel: 核函数实现(双 dtype 分路)
    ├── gelu_tiling.h       # Host↔Kernel 共享的 Tiling 结构体
    └── tiling_key_gelu.h   # dtype 模板选择 key
```

---

## 3. 核心算法

### 3.1 float32 路径:sigmoid 五次多项式近似

利用恒等式 `0.5·(1 + tanh(z)) = sigmoid(2z)`,把 GELU 写成:

```
GELU(x) = x · sigmoid(g(x)),  g(x) = D1·x + D3·x³ + D5·x⁵
```

系数经 minimax 拟合,FP32 下最大绝对误差 ~2.6e-5(< 1e-4,约 4× 余量):

```
D1 = 1.59501576856
D3 = 0.074011292044
D5 = -0.00070303357684
```

在 `x²` 上用霍纳法求值,**共 8 条向量指令**(1e-4 精度下的指令下限):

```
tmp = x·x              // Mul   → x²
y   = D5·tmp           // Muls
y   = y + D3           // Adds
y   = y·tmp            // Mul
y   = y + D1           // Adds  → P(x²)
y   = y·x              // Mul   → g
y   = sigmoid(y)       // Sigmoid → Φ
y   = x·y              // Mul   → GELU
```

> **为什么不用 native `Gelu<float>`**:native 的 float 快速路径是 tanh 近似,误差 ~1e-3 量级,过不了 1e-4;高精度模式又太慢。自研五次多项式是"精度达标 + 指令最少"的平衡点。

### 3.2 float16 路径:native `Gelu<half, false, false>`

half 精度要求宽松(1e-3),直接调用框架高阶 API `AscendC::Gelu<half, false, false>`,由框架自动申请临时区。

> **为什么 half 用 native 而非自研**:实测 native `Gelu<half,false,false>` 比"自研 fp32 Erf(Cast→Erf→Cast)"更快(演进中 81.9 → 82.9 即此改动),且精度达标。

---

## 4. 数据切分(Tiling)

见 `op_host/gelu.cpp` 的 `TilingFunc` 与 `op_kernel/gelu_tiling.h`。

### 4.1 泛化 Tiling(负载均衡)

- 按 **32B 对齐块**在核间均分总数据;余数块由前 `tailBlockNum` 个"大核"每核多担 1 块。
- **关键点**:所有核统一循环 `finalTileNum` 次;再把每个核的块数在这些片间尽量均摊(前 `remTiles` 片各多担 1 块)。这样**每一轮各核负载几乎相等**,消除"尾轮只有大核在转、其余核空闲"的空转损耗。
- 全部分片参数由 **host 侧预计算**,通过结构体传入 kernel,kernel 不做标量重算。
- 数学保证:每片块数 ≤ `tileBlockNum`(单缓冲物理上限),不会溢出 UB。

### 4.2 分档选核(消除固定开销)

小张量以固定开销(kernel 拉起 + 多核同步)为主,盲目用满 40 核反而更慢:

- **half**:`≤1024` 单核 / `≤8192` 每核约 1024 元素 / `>8192` 满核。
- **float**:`≤1024` 单核 / 其余 `ceil(n/1024)` 上限满核(平滑扩核,消除阈值处"8 核→40 核"的性能悬崖)。

### 4.3 UB 缓冲预算

- 自有缓冲固定占 **50% UB**,另一半留给高阶 API(Sigmoid / Gelu)自动申请临时区。
- 每元素字节:half `8B`(双缓冲 in/out),float `20B`(双缓冲 in/out + fp32 sigmoid 工作区)。
- **host / kernel 预算必须一致**(均为双缓冲):host 侧 `ownBytesPerElem` 与 kernel 侧 `kQueDepth=2` 对应,否则 tile 尺寸错配会导致性能回退。

---

## 5. 执行路径(Kernel)

```
Process()
 ├─ realCoreDataNum == 0 → 直接返回(该核无数据)
 ├─ useTBufPath (float 且 tileNum ≤ 1) → ProcessTBufSerial()  // 单 tile: TBuf 直通, 手动同步
 └─ else                              → ProcessTiled()        // 多 tile / 全部 half: TQue 双缓冲
```

- **float 多 tile / 所有 half** 走 `ProcessTiled`(TQue `inQueueX/outQueueY` 双缓冲)——搬运与计算重叠,访存受限用例的最优解。
- **float 单 tile** 走 `ProcessTBufSerial`(TBuf,无队列 Alloc/Free/EnQue/DeQue 开销)。
  - 注意:TBuf 下高阶 `DataCopy`/向量 API **不会自动插入 MTE↔V 流水同步**,必须手动用 `SetFlag/WaitFlag` 补齐(MTE2→V、V→MTE3 的 RAW,V→MTE2、MTE3→V 的 WAR),否则会 99.9% 结果错误。
- 搬运按 32B 对齐分流:对齐用 `DataCopy`,非对齐(末核尾块 / 标量)用 `DataCopyPad`,依据 `length` 安全裁剪。

---

## 6. 构建与部署

标准 Ascend C 自定义算子工程流程(CANN 环境,已验证 CANN 9.0.0 可编译):

```bash
# 配置 CANN 环境变量后, 在算子工程根目录执行工程既有编译脚本 / CMake 流程
# 产物为自定义算子 run 包, 安装后即可在推理/评测中调用
```

> **编译注意**:模板类中对成员模板(如 `inQueueX.AllocTensor<DT_INPUT_X>()`)且模板实参为依赖类型时,新版 clang 强制要求 `template` 关键字消歧,本代码已全部加上 `.template`。

---

## 7. 已验证走不通的方向(重要:勿重复尝试)

| 方向 | 结果 | 结论 |
| :--- | :--- | :--- |
| native `Gelu<float>` 快速路径 | float 误差 ~1e-3 量级,过不了 1e-4 | float 必须自研多项式 |
| float 三次多项式(降指令) | 误差 ~7e-4 > 1e-4 | float 至少需五次多项式 |
| half 自研 fp32 Erf(Cast→Erf→Cast) | 能过但更慢 | half 用 native `Gelu<false,false>` 最快 |
| **float 多 tile 串行 TBuf(去 TQue 同步)** | 补同步后无重叠,全线更慢 | 多 tile 访存受限,必须靠 TQue 双缓冲重叠 |
| **BUFFER_NUM=3 三缓冲** | tile 挤小,test6 反而更慢 | 双缓冲即最优 |
| UB 占比 50%→65% / 自适应分档 | 大张量回退 | UB 占比保持 50% |
| half 选核 8192/核(更少核) | 无改善 | 现有分档选核即可 |
| RegBase / VF 融合 | 910B(A2) 不支持 | 硬件不支持 |

**唯一稳定涨分的方向 = 降低计算指令数(Erf→tanh→sigmoid、half 用 native)。** 该方向已到指令下限。

---

## 8. 关键不变量(改动时务必保持)

1. **float 五次系数 D1/D3/D5** 与五次结构不可降阶(降阶过不了 1e-4)。
2. **half 用 native `Gelu<half,false,false>`**。
3. **UB 自有缓冲占比 50%**,且 **host `ownBytesPerElem` 与 kernel `kQueDepth` 必须一致**(均双缓冲)。
4. 非对齐搬运必须用 `DataCopyPad` 并以 `length` 做末尾裁剪。
5. Tiling 分片参数全部 host 侧预计算,kernel 不重算标量。
6. TBuf 直通路径必须手动补齐流水同步(`SetFlag/WaitFlag`)。
7. 模板类中依赖类型的成员模板调用需加 `.template` 关键字。
8. **元素总数 / 每核元素数 / GM 偏移相关的量必须用 `uint64_t`,不可收窄回 `uint32_t`**:`GeluTilingData` 的 `length/bigCoreDataNum/smallCoreDataNum/smallCoreBaseOffset`、host 的 `inputNum/inputLength/totalBlockNum/*CoreDataNum` 及 kernel 的 `globalOffset/coreDataNum/realCoreDataNum/offset/remain` 均为 64 位。否则超大张量(FP16 >~21 亿 / FP32 >~10 亿元素)会 32 位溢出,导致分核/偏移错误、静默数据损坏。循环计数、每片块数(受 UB/核数约束、恒小于 2^32)保持 `uint32_t`。

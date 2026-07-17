# GELU 自定义算子（AscendC / CANN）

基于华为昇腾 CANN 软件栈与 AscendC 编程语言实现的 **GELU（Gaussian Error Linear Unit，高斯误差线性单元）** 自定义算子。

- 支持芯片：`Ascend910B`（在 `CMakeLists.txt` 中通过 `ASCEND_COMPUTE_UNIT` 指定）
- 支持数据类型：`float16`、`float32`
- 支持数据格式：`ND`
- 算子包名：`custom`（由 `npu_op_package` 注册）

---

## 1. 算子说明

GELU 是一种常用于 Transformer 类模型的激活函数。其精确数学定义为：

```
gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
```

其中 `erf` 为高斯误差函数。**本算子采用 5 阶近似实现（Hendrycks tanh 近似的高阶 exp 形式）**，而非调用 `Erf` 指令：

```
gelu(x) ≈ x / (1 + exp(-scale * (x + c1 * x^3 + c3 * x^5)))

其中：
  scale = 2 * sqrt(2/π) ≈ 1.59576912
  c1    = 0.044715
  c3    = 0.001072
```

该近似等价于常见的 `0.5 * x * (1 + tanh(sqrt(2/π) * (x + c1*x^3 + c3*x^5)))`，但改写为 `exp` 形式；相比 3 阶近似额外**保留 `x^5` 项**以获得更高精度。

**为何用近似而非精确 erf：**
- AscendC 的 `Erf` 属于慢指令，是性能瓶颈；改用 `Mul / Muls / Add / Adds / Exp / Div` 组合后，指令吞吐更友好。
- 5 阶近似在常用数值范围内与精确 `erf` 结果非常接近，足以满足 Transformer 类模型的精度需求，且速度更快。

计算流程（Device 侧，共 11 步）：

```
Step 1:  z = x^2,  tmp = x^2          // 保存 x^2 用于后续算 x^5
Step 2:  z = z * x        = x^3
Step 3:  t = tmp * z       = x^5       // x^2 * x^3
Step 4:  z = c1 * x^3
Step 5:  t = c3 * x^5
Step 6:  z = c1*x^3 + c3*x^5
Step 7:  z = x + z                   // x + c1*x^3 + c3*x^5
Step 8:  z = -scale * z              // -1.59576912 * z
Step 9:  z = exp(z)
Step 10: z = z + 1                   // 1 + exp(z)
Step 11: z = x / (1 + exp(z))        // 得到 gelu(x)
```

---

## 2. 目录结构

```
.
├── CMakeLists.txt            # 顶层构建脚本（find_package(ASC) + 算子包入口）
├── op_host/                  # Host 侧：算子定义 + Tiling 策略
│   ├── CMakeLists.txt        # Host 侧源码编译/代码生成配置
│   └── gelu.cpp              # OpDef 定义、InferShape/InferDataType、TilingFunc
└── op_kernel/                # Device 侧（AICore Kernel）：核函数实现
    ├── CMakeLists.txt        # Kernel 源码与库配置
    ├── gelu.cpp              # KernelGelu 类 + gelu 核函数入口
    ├── gelu_tiling.h         # Host 与 Kernel 之间的 Tiling 数据结构协议
    └── tiling_key_gelu.h     # TilingKey 模板（按输入数据类型实例化）
```

---

## 3. 环境依赖

- 昇腾 AI 处理器：`Ascend910B`
- 驱动与固件：与 CANN 版本匹配
- CANN 开发套件（含 `ASC` CMake 包、`ascendc` 头文件与工具链）
- CMake：`>= 3.16.0`
- 构建需在已配置好 CANN 环境（含 `ASC` 包的 `find_package` 路径）的机器上进行

---

## 4. 编译构建

在本项目根目录执行：

```bash
# 1. 创建构建目录
mkdir -p build && cd build

# 2. 生成构建系统（CMake 会自动 find_package(ASC)）
cmake ..

# 3. 编译
make -j
```

构建产物（算子包安装文件）会输出到 `CMAKE_BINARY_DIR`（即构建目录）下，包含：

- `cust_optiling`：Host 侧 Tiling 动态库
- `cust_opapi`：单算子 aclnn 调用动态库（由 `ascend_autogen` 自动生成）
- `ascendc_kernels`：Device 侧 Kernel 动态库

安装算子包后即可在应用中通过 aclnn 单算子接口（`aclnnGelu` 系列）调用该算子。

---

## 5. 算子接口

| 项目       | 说明                                                         |
| ---------- | ------------------------------------------------------------ |
| 算子名     | `Gelu`                                                       |
| 输入       | `input_x`：shape 任意，`float16`/`float32`，格式 `ND`        |
| 输出       | `output`：与输入同 shape、同 dtype，格式 `ND`                |
| 计算单元   | `AICore`（`ascend910b`）                                     |
| Shape/Type 推导 | 当前 `InferShape` / `InferDataType` 直接透传（保持输入输出一致） |

---

## 6. 实现要点

### 6.1 Host 侧 Tiling 策略（`op_host/gelu.cpp`）

采用**两层切分（大核/小核模型）**，不依赖固定核数：

1. **对齐粒度**：`alignElems = 32 / typeLen`（fp32 → 8，fp16 → 16），作为尾块 32B 对齐的元素单位。
2. **核数反推**：按 256B 对齐反推实际使用的核心数 `coreNum`（`usedCoreNum`），而非盲目使用 `totalCoreNum`，保证每个核负载 ≥ 256B 对齐粒度，提升 UB 利用率。
3. **核间不均分**：元素总数 `totalElems` 均分到 `coreNum` 个核，余数 `tailBlockNum` 由前 `tailBlockNum` 个**大核**各多拿 1 个元素（`bigCoreDataNum = base+1`，小核为 `base`）。
4. **核内切分**：每核再按 `tileDataNum`（依据总数据量分级）切成 `tileNum` 轮，最后一轮为**尾块**。分级系数（单位：`DEFAULT_SLOT = 32/typeLen`）：

   | 总字节数 `totalDataScale` | `tileDataNum`      |
   | ------------------------- | ------------------ |
   | `< 128`                   | `DEFAULT_SLOT*320` |
   | `< 512`                   | `DEFAULT_SLOT*1040`|
   | `< 2048`                  | `DEFAULT_SLOT*1040`|
   | `≥ 2048`                  | `DEFAULT_SLOT*1020`|

5. **32B 对齐**：尾块 `tailDataNum` 在 Host 侧完成向上 32B 对齐，Kernel 侧零开销直接使用。
6. **双缓冲**：`bufferNum = 2`，使用 `TQue` 2 槽队列实现流水。

> 注：`SetBlockDim(coreNum)` 在核间分配（大核/小核）计算完成后设置。

### 6.2 Device 侧 Kernel（`op_kernel/gelu.cpp`）

- `KernelGelu<T>` 模板类按数据类型 `T`（`float16` / `float32`）实例化。
- 大核 / 小核根据 `blockIdx` 与 `tailBlockNum` 自动分发，分别计算各自偏移 `gmOffset`。
- GM 视图额外预留 `tileDataNum` 余量，避免 `DataCopy` 边界截断。
- 主循环：`满块迭代`（每轮 `tileDataNum` 个）+ `尾块迭代`（已对齐的 `tailDataNum` 个）。
- 计算流水：`CopyIn → Compute → CopyOut`，通过双缓冲队列（VECIN / VECOUT）重叠搬运与计算。
- **临时缓冲**：`tmpBuf`（`TBuf`）用于暂存 `x^2`，以在 Step 3 计算 `x^5 = x^2 * x^3`，避免重复计算平方。

### 6.3 Tiling 数据结构协议（`op_kernel/gelu_tiling.h`）

`GeluTilingData` 是 Host 与 Kernel 之间的数据契约，字段含义见头文件注释，关键字段：

| 字段             | 含义                                               |
| ---------------- | -------------------------------------------------- |
| `smallCoreDataNum` | 小核分到的总元素数                               |
| `bigCoreDataNum`   | 大核分到的总元素数（= smallCoreDataNum + 1）      |
| `finalBigTileNum`  | 大核迭代轮数（含尾块）                           |
| `finalSmallTileNum`| 小核迭代轮数（含尾块）                           |
| `tileDataNum`      | 单轮最大元素数（受 UB 容量约束，按数据量分级）   |
| `smallTailDataNum` | 小核尾块元素数（已 32B 对齐）                    |
| `bigTailDataNum`   | 大核尾块元素数（已 32B 对齐）                    |
| `tailBlockNum`     | 大核个数（= totalElems % coreNum，为 0 时无大核）|
| `bufferNum`        | TQue 槽数（本项目固定 2）                        |
| `usedCoreNum`      | 实际使用的核心数（256B 对齐反推）                |

---

## 7. 自定义与扩展

- **更换芯片**：修改根 `CMakeLists.txt` 中的 `set(ASCEND_COMPUTE_UNIT ascend910b)` 及 `op_host/gelu.cpp` 中 `AddConfig("ascend910b")`。
- **新增数据类型**：在 `op_host/gelu.cpp` 的 `OpDef` 与 `op_kernel/tiling_key_gelu.h` 的 `ASCENDC_TPL_DATATYPE_DECL/SEL` 中补充对应 dtype。
- **调整单轮粒度**：修改 `op_host/gelu.cpp` 中 `tileDataNum` 的分级系数（第 5 步）。
- **调整缓冲槽数**：修改 `bufferNum`（`op_host/gelu.cpp` 中的赋值与 `op_kernel/gelu.cpp` 的 `BUFFER_NUM` 需保持一致）。
- **切换近似精度**：如需调整近似阶数，可修改 `op_kernel/gelu.cpp` 中的系数 `C1` / `C3` 与计算步骤（如去除 `x^5` 项即退化为 3 阶近似）。

---

## 8. 参考

- 华为昇腾 CANN 文档：算子开发（AscendC）
- GELU 原始论文：*Gaussian Error Linear Units (GELUs)*, Hendrycks & Gimpel, 2016（含 tanh 近似形式）

# LessEqual Ascend C 自定义算子实现

## 1. 项目简介

本项目实现 CANNJudge LessEqual 自定义算子，基于 Ascend C Kernel 直调方式完成 Host 侧 tiling 与 Kernel 侧向量化比较计算。算子对两个输入张量执行逐元素 `x1 <= x2` 比较，输出布尔类型结果张量，行为与 TensorFlow `tf.math.less_equal` 完全对齐。

- 支持输入数据类型：`float16`、`float32`、`int32`、`int8`
- 输出数据类型：`bool`
- 支持数据格式：`ND`
- 支持最高 16 维张量
- 支持 NumPy/TensorFlow 标准广播语义（含标量广播、向量与矩阵广播、不同形状高维广播）
- 兼容非 32 字节对齐的尾块场景与空张量场景
- 比较运算完全准确，无误差容忍

目标 SOC：`ascend910b`。

---

## 2. 计算公式

基础公式：

```text
y_i = (x1_i <= x2_i)
```

- 若 `x1_i <= x2_i`，则 `y_i = True`
- 若 `x1_i > x2_i`，则 `y_i = False`

输出形状为两个输入按 NumPy 广播规则融合后的形状，输出始终为 `bool` 类型。

---

## 3. 文件结构

```text
code/
├── CMakeLists.txt                  # 顶层构建配置
├── op_host/
│   ├── CMakeLists.txt
│   └── less_equal.cpp              # Host 侧：算子注册、shape/type 推导、tiling 计算
└── op_kernel/
    ├── CMakeLists.txt
    ├── less_equal_tiling.h         # TilingData 结构定义
    ├── tiling_key_less_equal.h     # TilingKey 模板参数定义
    └── less_equal.cpp              # Kernel 侧：GM->UB 搬运、向量化比较、UB->GM 写回
```

各文件职责：

- `op_host/less_equal.cpp`：算子注册（`OpDef`）、输入输出类型声明、`InferShape` / `InferDataType`、`TilingFunc`（计算广播输出形状、各输入步长、`blockDim` 切分，并写入 `LessEqualTilingData`）。
- `op_kernel/less_equal_tiling.h`：`LessEqualTilingData` 结构体，含 `length`、`blockLength`、`rank`、`isBroadcast`、`scalarBroadcast` 以及 16 维的 `outDims` / `x1Strides` / `x2Strides`。
- `op_kernel/tiling_key_less_equal.h`：以 `DT_X1` 为模板参数，按 `float16/float/int32/int8` 四种 dtype 选择 kernel 实例。
- `op_kernel/less_equal.cpp`：`KernelLessEqual` 模板类，三段式 `CopyIn/Compute/CopyOut`，按 tiling 标志走连续、标量广播、通用广播三条计算路径。

---

## 4. Host 侧设计

### 4.1 TilingData 结构

```cpp
constexpr uint32_t LESS_EQUAL_MAX_DIMS = 16;

struct LessEqualTilingData {
    uint32_t length;          // 广播后输出总元素数
    uint32_t blockLength;     // 每个核处理的基础块长度
    uint32_t rank;            // 输出维度数
    uint32_t isBroadcast;     // 是否需要广播
    uint32_t scalarBroadcast; // 0=否, 1=x2为标量, 2=x1为标量
    uint32_t outDims[LESS_EQUAL_MAX_DIMS];   // 输出各维大小
    uint32_t x1Strides[LESS_EQUAL_MAX_DIMS]; // x1 各维步长（广播维为0）
    uint32_t x2Strides[LESS_EQUAL_MAX_DIMS]; // x2 各维步长（广播维为0）
};
```

通过 `REGISTER_TILING_DEFAULT(LessEqualTilingData)` + `GET_TILING_DATA_WITH_STRUCT(LessEqualTilingData, tiling_data, tiling)` 注册，框架自动保证 Host 与 Kernel 侧 `sizeof(LessEqualTilingData)` 一致，无需任何手动 padding。

### 4.2 Tiling 计算流程

1. 通过 `GetRequiredInputShape(0/1)` 获取 `StorageShape`，再取 `GetOriginShape()`，得到 x1、x2 的原始形状。
2. 计算 `outRank = max(rank1, rank2)`，若超过 `LESS_EQUAL_MAX_DIMS` 直接返回 `GRAPH_FAILED`。
3. 从右向左逐维对齐，按 NumPy 规则判定是否可广播（维度相等或其中之一为 1），不可广播则返回失败；同时累乘得到 `outputLength`，溢出 `uint32_t` 时返回失败。
4. 从右向左计算 x1、x2 的行主序步长，广播维（自身为 1 而输出维非 1）步长置 0。
5. 检测标量广播：若一端整体步长为 1、另一端整体步长等于 `outputLength`，则标记 `scalarBroadcast = 1`（x2 为标量）或 `2`（x1 为标量），Kernel 走快路径。
6. 按 `minElementsPerCore = 4096` 估算所需核数，与平台 `GetCoreNumAiv()` 取小，设置 `blockDim` 与 `blockLength`。

---

## 5. Kernel 侧设计

三段式流水：

```text
CopyIn  : GM -> UB（DataCopyPad 兼容非对齐尾块）
Compute : UB 向量化比较
CopyOut : UB -> GM（DataCopyPad 写回 bool）
```

`TQue` 采用 `BUFFER_NUM = 2` 双缓冲，`TILE_LENGTH = 1024`。

### 5.1 三条计算路径

Kernel 根据 tiling 标志选择路径：

- **连续路径（`isBroadcast == 0`）**：x1、x2 与输出形状完全一致，按一维连续布局直接逐元素比较。
- **标量广播路径（`scalarBroadcast != 0`）**：一端为标量，使用 `CompareScalar` 指令直接与标量比较，避免广播搬运。
- **通用广播路径（`isBroadcast == 1` 且非标量）**：按输出线性下标反推各维坐标，结合 `x1Strides` / `x2Strides` 计算x1、x2 的偏移，逐 tile 计算并写回。

### 5.2 各 dtype 的向量化实现

- **float16 / float32**：直接使用 `AscendC::Compare` 的 `LE` 模式。
- **int32**：先 `Min(x1, x2)` 取较小值，再用 `Compare(EQ)` 判断 `min == x1`，等价于 `x1 <= x2`（避免整数比较指令缺失）。
- **int8**：先 `Cast` 到 `half`，再走 `Compare(LE)`，最后写回 `bool`。

### 5.3 非对齐与空张量

- 搬入与写回均使用 `DataCopyPad`，按 32 字节对齐补零，兼容任意非对齐尾块。
- `length == 0` 时 Kernel 直接返回，不触发任何计算，安全处理空张量。

---

## 6. 关键参数

Kernel 侧：

```cpp
constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_LENGTH = 1024;
constexpr uint32_t COMPARE_MASK_BYTES = 128;
```

Host 侧：

```cpp
constexpr uint32_t LESS_EQUAL_MAX_DIMS = 16;
constexpr uint32_t minElementsPerCore = 4096;
```

---

## 7. 构建与运行

### 7.1 环境依赖

- CANN 8.5.0 及以上
- Ascend C 编译工具链
- 目标 SOC：`ascend910b`

### 7.2 编译算子

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 按实际 CANN 安装路径
cd code
mkdir -p build_out && cd build_out
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

构建产物位于 `build_out/`，包含 `libcust_opapi.so`、`libcust_optiling.so`、`ascendc_kernels.so` 及 `ascend910b` 目录下的算子 kernel 与 op-info。

### 7.3 调用算子

编译完成后，设置自定义算子路径并调用 aclnn 接口：

```bash
export ASCEND_CUSTOM_OPP_PATH=$(pwd)/build_out/tmp/vendors/custom
export LD_LIBRARY_PATH=$(pwd)/build_out/lib:$(pwd)/build_out/op_host:$LD_LIBRARY_PATH
```

调用接口：

```cpp
aclnnLessEqualGetWorkspaceSize(x1, x2, y, &workspaceSize, &executor);
aclnnLessEqual(workspace, workspaceSize, executor, stream);
```

---

## 8. 测试场景

测试覆盖题目要求的全部场景：

- **数据类型**：`float16`、`float32`、`int32`、`int8` 全覆盖
- **维度场景**：1D、2D、3D、4D 及高维 ND
- **非对齐场景**：33、31、15、7 等非 32 整倍数尾块
- **空张量**：零元素输入
- **广播场景**：
  - 标量与张量广播（`scalarBroadcast == 1 / 2` 两条快路径均覆盖）
  - 向量与矩阵广播（如 `(2,3)` vs `(3)`、`(1,4)` vs `(3,1)`）
  - 不同形状高维广播（如 `(2,1,4)` vs `(3,4)` → `(2,3,4)`）
- **边界数值**：负数比较、全相等、`-0/+0`、接近数据类型边界的值
- **大规模**：4097 元素 float16 1D（多核 + 非对齐）

精度要求：比较运算完全准确，无误差容忍（输出与 `tf.math.less_equal` 逐元素一致）。

### 自检结果

- 基础功能测试：9/9 通过（`float32` / `float16` / `int32` / `int8`，含非对齐 33、31，含 3 种广播类型）
- 边界与广播测试：13/13 通过（含空张量、3D、标量广播双路径、高维广播、int8 非对齐、大规模 4097、负数比较）
- 随机 fuzz 测试：300 组随机 shape/dtype/广播组合，逐组与 NumPy 参考实现逐元素比对，全部一致

---

## 9. 设计要点总结

1. **struct-by-const-ref + 框架自动 sizeof 对齐**：`LessEqualTilingData` 不做任何手动 padding，通过 `REGISTER_TILING_DEFAULT` / `GET_TILING_DATA_WITH_STRUCT` 让框架保证 Host 与 Kernel 侧结构体大小一致，从根本上避免 tiling capacity 校验失败导致的空指针问题。
2. **三路径分流**：连续、标量广播、通用广播分别走快路径与通用路径，标量广播用 `CompareScalar` 省去广播搬运开销。
3. **int32 走 `Min + Compare(EQ)`**：规避整数比较指令的语义限制，等价实现 `x1 <= x2`。
4. **int8 先 Cast 到 half**：复用浮点 `Compare(LE)` 路径，保证精度。
5. **全程 `DataCopyPad`**：搬入写回均支持非 32B 对齐尾块，兼容任意非对齐输入与空张量。
6. **16 维上限 + 溢出保护**：`outRank > 16` 或 `outputLength` 超过 `uint32_t` 上限时 `TilingFunc` 直接返回 `GRAPH_FAILED`，保证极端输入下的安全性。

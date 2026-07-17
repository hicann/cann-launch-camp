# LessEqual Ascend C 自定义算子

## 1. 项目简介

本工程使用 Ascend C 在 Ascend 910B 上实现 TensorFlow `tf.math.less_equal` 的核心语义：

```text
y = (x1 <= x2)
```

算子逐元素比较输入张量 `x1` 和 `x2`，输出 `bool` 张量。工程支持：

- 输入类型：`float16`、`float32`、`int32`、`int8`
- 输出类型：`bool`
- 数据格式：`ND`
- NumPy/TensorFlow 风格广播
- 标量与张量广播
- 向量与高维张量广播
- 非 32 字节对齐数据
- 空张量和零维标量
- 最多 8 维张量
- 大张量的 64 位长度、stride 和 GM 偏移

核心目标是在保证比较结果完全正确的前提下，使用多核、UB 分块、`DataCopyPad`、Double Buffer 和 Vector API 提升性能。

## 2. 工程结构

```text
LessEqual_project/
├── README.md
└── code/
    ├── CMakeLists.txt
    ├── CMakePresets.json
    ├── build.sh
    ├── op_host/
    │   ├── CMakeLists.txt
    │   └── less_equal.cpp
    ├── op_kernel/
    │   ├── CMakeLists.txt
    │   ├── less_equal_custom.cpp
    │   ├── less_equal_custom_tiling.h
    │   └── tiling_key_less_equal_custom.h
    └── test_less_equal.cpp
```

其中：

- `op_host` 负责算子注册、Shape/DType 推导、广播校验、多核切分及 UB tiling。
- `op_kernel` 负责 GM 与 UB 之间的数据搬运、广播映射和向量比较。
- `less_equal_custom_tiling.h` 定义 Host 下发到 Kernel 的 tiling 数据。
- `tiling_key_less_equal_custom.h` 定义四种输入数据类型的模板实例。

## 3. 算子接口

| 类型 | 名称 | 支持类型 | 格式 | 说明 |
|---|---|---|---|---|
| 输入 | `x1` | float16/float32/int32/int8 | ND | 第一个输入张量 |
| 输入 | `x2` | float16/float32/int32/int8 | ND | 第二个输入张量，类型必须与 `x1` 相同 |
| 输出 | `y` | bool | ND | 广播后的逐元素比较结果 |

广播从最右侧维度开始。两个对应维度相等，或者其中一个为 `1` 时可以广播；否则 Shape 推导和 Tiling 均返回失败。

空维广播没有简单使用 `max(d1, d2)`：维度 `0` 与维度 `1` 广播后的结果应为 `0`。

## 4. `op_host` 设计

### 4.1 Shape 与类型推导

`InferShape` 完成以下工作：

1. 检查两个输入 Shape 和输出 Shape 指针。
2. 检查输入秩不超过 `MAX_DIMS = 8`。
3. 将两个输入 Shape 右对齐。
4. 对每一维执行广播兼容性校验。
5. 设置广播后的输出 Shape。

`InferDataType` 始终将输出类型设置为 `ge::DT_BOOL`。

Shape 推导与 Tiling 使用相同的广播规则，避免图编译阶段成功、运行阶段才失败的不一致行为。

### 4.2 Tiling 数据

Host 向 Kernel 下发：

- 输出总元素数 `totalLength`
- 对齐后的 `x1Shape`、`x2Shape` 和 `outShape`
- 两个输入的连续存储 stride
- 内部维度数 `dims`
- 输入 dtype
- 实际启用的 Vector Core 数量 `blockDim`
- 单次 UB 处理元素数 `ubChunkSize`

`totalLength`、stride 和 Kernel GM 偏移均使用 `uint64_t`，避免多维乘积超过 32 位后产生静默回绕和错误地址。

零维标量在 Tiling 内部归一化成一维 `[1]`，但输出 Shape 仍保持标量语义。这可以复用线性 Kernel，同时避免 `dims - 1` 越界。

### 4.3 多核切分

Host 根据总元素数动态设置核数：

- 小张量使用较少的核，避免启动开销超过计算收益。
- 大张量最多使用平台提供的全部 AIV Core。
- 目标是每核至少处理约 4096 个元素。

Kernel 再按 64 位元素数量将任务尽量均匀分配到各核，尾核处理剩余数据。

### 4.4 UB 分块

Kernel 最坏情况下需要：

- 两个 Double Buffer 输入队列
- 一个 Double Buffer 输出队列
- 输入类型工作区
- 两个 half 工作区
- 一个 float/int32 工作区

Host 使用实际 UB 容量的安全比例计算 `ubChunkSize`，并保证：

- 输入搬运字节数不超过 `DataCopyParams::blockLen` 的 `uint16_t` 上限。
- chunk 按 32 字节对应的元素数对齐。
- Kernel 端再次执行上限保护，防止异常 tiling 导致截断。

### 4.5 CMake 改进

`op_host/CMakeLists.txt` 显式列出 `less_equal.cpp`，不再使用 `file(GLOB)` 搜索 Host 源码，使新增或删除文件时的构建行为更明确。

自动生成的 aclnn 源文件仍由构建系统管理。

## 5. `op_kernel` 设计

### 5.1 两条执行路径

Kernel 在运行时判断输入是否需要广播：

1. **线性快速路径**：两个输入 Shape 均等于输出 Shape，直接将张量展平成一维并进行多核分块。
2. **广播路径**：按输出最后一维逐行处理，通过 64 位 stride 将输出坐标映射到两个输入的 GM 偏移。

广播路径只在行和维度层面执行 Scalar 索引计算，元素比较由 Vector API 批量完成。

### 5.2 数据搬运

输入和输出统一使用 `DataCopyPad`：

- 支持非 32 字节对齐的首尾数据。
- 避免手工处理 GM 尾部时出现越界。
- `blockLen` 在转换成 `uint16_t` 前已经由 Host 和 Kernel 双重限制。

输入和输出队列使用两个 buffer，使 MTE2、Vector 和 MTE3 指令具备流水并行条件。

广播标量搬入 UB 后，通过 `EnQue/DeQue` 建立 MTE2 到 Vector 的依赖，确保 Vector 指令读取标量前搬运已完成。

### 5.3 广播填充与 CANN 9.0 API 兼容

Ascend 910B 对应的 CANN 9.0 `Duplicate` 不支持 int8。处理方式为：

```text
int8 标量 → Cast 到 half → half Duplicate → Cast 回 int8
```

half、float 和 int32 可直接使用对应类型的 `Duplicate`。

这避免了对 int8 调用不支持的 API，并保持广播后的主体计算为向量化实现。

### 5.4 各数据类型比较实现

#### float16

使用 `Max/Sub/Abs/Mins/Muls/Adds/Cast` 组合生成 `0/1` 结果。所有操作在 UB 内批量执行。

#### float32

使用 float32 Vector API 完成比较逻辑，再经 float32 → half → uint8 的 Cast 链输出 bool 存储值。

#### int8

输入先转换到 half，在 half 上执行向量比较逻辑，最后转换为 uint8。

int8 的完整取值范围可由 half 精确表示，因此不会引入比较精度损失。

#### int32

早期实现直接通过差值推导比较结果，在 `INT_MIN` 与 `INT_MAX` 异号比较时会发生 32 位回绕。

最终实现采用符号分离：

1. 将 `x1`、`x2` 是否为负转换成 `neg1/neg2` 的 `0/1` 标记。
2. 同号元素使用 `x2 - x1` 判断，此时差值不会溢出。
3. 异号元素直接由 `x1` 的符号确定结果。
4. 使用向量乘法作为掩码，将同号结果与异号结果合并。

CANN 9.0 的 dav-c220 实现不支持 int32 `Abs`。符号差只可能为 `-1/0/1`，因此使用：

```cpp
AscendC::Mul(x2, x2, x2, count);
```

平方得到 `1/0/1`，与该场景的绝对值等价，同时是 Ascend 910B 支持的 int32 Vector API。

## 6. 改进过程

### 6.1 初始版本问题

初始 Kernel 的核心比较使用逐元素 `LocalTensor::GetValue/SetValue` 循环。该方式具有以下问题：

- Scalar 指令比例高。
- Vector Core 利用率低。
- 大张量耗时随元素数量线性增加，无法充分使用 SIMD。
- 虽然声明了 Double Buffer，但逐元素计算成为主要瓶颈。

### 6.2 第一阶段：向量化与流水

完成以下改进：

- 将逐元素比较替换为 Ascend C Vector API。
- 增加无广播线性快路径。
- 广播路径改为按行向量处理。
- 输入输出统一使用 `DataCopyPad`。
- 输入输出队列启用 Double Buffer。
- 根据实际 UB 容量动态计算 chunk。

### 6.3 第二阶段：代码审查修复

根据代码审查修复：

- `InferShape` 缺少广播兼容性检查。
- `InferShape` 秩超过 8 时可能写越界。
- `InferShape` 输入/输出 Shape 空指针风险。
- 零维标量导致 `dims - 1` 越界。
- `totalLength` 的 32 位乘积溢出。
- stride 和 GM 偏移的 32 位溢出。
- `DataCopyParams::blockLen` 强制转成 `uint16_t` 时截断。
- 未使用的 `outStride` 字段。
- Host 源码使用 `file(GLOB)`。

### 6.4 第三阶段：CANN 9.0 编译兼容

实际编译暴露两项 API 支持差异：

1. `Duplicate<int8_t>` 在 Ascend 910B 上不受支持，改用 half 中转。
2. int32 `Abs` 在 dav-c220 实现中不受支持，改用 int32 `Mul` 平方。

后续修改应以 CANN 9.0 官方 API 的具体产品支持矩阵为准，不能只根据 `LocalTensor<T>` 可声明的类型推断某个 Vector API 支持相同类型。

### 6.5 第四阶段：边界精度修复

测试发现普通 int32 场景通过，但 `INT_MIN/INT_MAX` 组合失败。根因是异号减法回绕。最终使用“符号分类 + 同号差值 + 掩码合并”修复，避免边界值影响判断。

## 7. 测试情况

在 int32 边界修复前，端到端测试结果为：

```text
Passed: 19
Failed: 1
Total:  20
```

已通过场景包括：

- 四种输入 dtype 的基础比较
- float32 标量广播
- 2D/1D、3D 和混合维度广播
- 3D、4D 张量
- `N=17`、`N2=33` 等非对齐场景
- 相等值边界
- int8 最小值/最大值
- 单元素张量
- `N=10000` 大向量
- float16 二维及非对齐场景

唯一失败项为 `boundary_int32_minmax`，其实现已修复。之后发现的 int32 `Abs` 编译问题也已替换为受支持的 `Mul`。最终版本仍应在 CANN 9.0 环境重新执行完整测试，目标结果为：

```text
Passed: 20
Failed: 0
Total:  20
```

## 8. 编译与运行

### 8.1 编译工程

```bash
cd LessEqual_project/code
source ${HOME}/Ascend/cann-9.0.0/set_env.sh
bash build.sh
```

### 8.2 安装

```bash
cmake --install build_out --prefix=${HOME}/vendors/customize
cp -r build_out/tmp/vendors/custom/op_impl/* \
    ${HOME}/vendors/customize/op_impl/
```

### 8.3 编译测试程序

```bash
source ${HOME}/Ascend/cann-9.0.0/set_env.sh

export LD_LIBRARY_PATH=${HOME}/vendors/customize/lib:\
${HOME}/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:\
${ASCEND_HOME_PATH}/opp/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:\
${LD_LIBRARY_PATH}

export ASCEND_CUSTOM_OPP_PATH=${HOME}/vendors/customize

g++ -o test_less_equal test_less_equal.cpp \
    -I${ASCEND_HOME_PATH}/include \
    -I${ASCEND_HOME_PATH}/include/acl \
    -I${HOME}/vendors/customize/include \
    -L${ASCEND_HOME_PATH}/aarch64-linux/lib64 \
    -L${HOME}/vendors/customize/lib \
    -L${ASCEND_HOME_PATH}/opp/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64 \
    -lascendcl -lnnopbase -lcust_opapi -lcust_opmaster_rt2.0 \
    -lstdc++ -std=c++17 \
    -Wl,-rpath,${ASCEND_HOME_PATH}/aarch64-linux/lib64 \
    -Wl,-rpath,${HOME}/vendors/customize/lib \
    -Wl,-rpath,${ASCEND_HOME_PATH}/opp/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64

./test_less_equal
```

修改 Kernel 后应重新执行完整构建，避免复用旧的 Kernel 二进制。

## 9. 性能验证建议

功能测试全部通过后，使用 `msprof op` 采集性能数据，重点关注：

- `aiv_time`
- `aiv_scalar_ratio`
- `aiv_vec_ratio`
- `aiv_mte2_ratio`
- `aiv_mte3_ratio`
- 各核耗时差异
- GM 搬运有效带宽

建议分别测量：

- 无广播的大张量线性路径
- 标量广播
- 最后维度广播
- 高维广播
- 四种 dtype
- 对齐和非对齐 shape

若 Scalar 比例仍然较高，应重点分析广播路径中的 64 位坐标和 stride 计算；若 MTE 比例较高，应进一步调整 chunk 大小并检查流水是否充分重叠。

## 10. 已知限制

- 最大支持 8 维输入。
- 两个输入 dtype 必须相同。
- 仅配置 Ascend 910B。
- 广播路径按输出最后一维逐行处理；复杂高维广播的 Scalar 索引成本高于无广播路径。
- 当前性能结论需要以真实 NPU 上的 `msprof` 数据为准。
- 浮点 NaN、正负无穷和正负零应补充专门测试，以确认与 TensorFlow 的精确语义完全一致。

## 11. 参考资料

- TensorFlow `tf.math.less_equal`
- CANN 9.0 Ascend C API 文档
- Ascend C `DataCopyPad`、`Duplicate`、`Cast`、`Mul` 等基础 API
- Ascend C 算子性能优化与 Tiling 设计指南


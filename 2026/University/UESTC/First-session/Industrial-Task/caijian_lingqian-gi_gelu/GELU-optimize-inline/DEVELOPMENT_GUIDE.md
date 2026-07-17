# GELU 自定义算子 — 本地开发、调试与优化指南

本文档记录在本地环境（CANN 8.5.0 / Ascend910B4）下开发、编译、调试和优化 AscendC 自定义算子的完整流程，包括环境配置、Profiling 采集、性能分析和优化策略。

---

## 目录

1. [环境配置](#1-环境配置)
2. [编译算子](#2-编译算子)
3. [安装算子包](#3-安装算子包)
4. [运行测试](#4-运行测试)
5. [Profiling 性能分析](#5-profiling-性能分析)
6. [结果解读与瓶颈定位](#6-结果解读与瓶颈定位)
7. [优化实践记录](#7-优化实践记录)
8. [踩坑记录](#8-踩坑记录)

---

## 1. 环境配置

### 1.1 当前环境

| 组件 | 版本/路径 |
|------|----------|
| CANN | 8.5.0 (`/usr/local/Ascend/cann-8.5.0`) |
| NPU | Ascend910B4 |
| SoC | `ascend910b` |
| 编译器 | GCC 11.4.0 |
| CMake | 4.2+ |
| Python | 3.11 |
| msprof | `/usr/local/Ascend/cann-8.5.0/bin/msprof` |

### 1.2 环境变量

必须设置的环境变量：

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0
export ASCEND_OPP_PATH=/usr/local/Ascend/cann-8.5.0/opp
export ASCEND_CUSTOM_OPP_PATH=$ASCEND_HOME_PATH/opp/vendors/custom
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH
```

> **注意**：`$ASCEND_HOME_PATH/lib64` 是可运行二进制所需的。不要将 `aarch64-linux/devlib/linux/aarch64` 加入 `LD_LIBRARY_PATH`——这会导致 `aclInit` 返回错误码 500000（详见 [踩坑记录](#81-ld_library_path-中的-devlib-与-aclinit-冲突)）。

### 1.3 验证环境

```bash
# 检查 CANN 版本
cat $ASCEND_HOME_PATH/version.cfg

# 检查 NPU 状态
npu-smi info

# 检查 msprof 工具
which msprof
msprof op --help
```

---

## 2. 编译算子

### 2.1 一键编译

```bash
cd /opt/atomgit/GELU
bash build.sh
```

`build.sh` 执行流程：

1. **CMake 配置** — 查找 CANN 工具链，设置 `ascend910b` 目标
2. **Host 编译** — 编译 `op_host/` → 生成 `libcust_optiling.so`、`libcust_opsproto_rt2.0.so`、`libcust_opapi.so`
3. **Kernel 编译** — 编译 `op_kernel/` → 为每个 tilingKey 生成 `.o` 内核二进制（位于 `build_out/op_kernel/ascendc_kernels/binary/ascend910b/gelu/`）
4. **打包** — 生成安装包 `build_out/custom_opp_ubuntu_aarch64.run`

### 2.2 编译产物结构

```
build_out/
├── custom_opp_ubuntu_aarch64.run     # 可安装的算子包
├── autogen/
│   ├── libascend_all_ops.so          # 算子注册 + tiling + kernel 全量包
│   └── aclnn_gelu.cpp                # 自动生成的 aclnn 接口
├── op_host/
│   └── libcust_opapi.so              # aclnn 接口库
├── op_kernel/
│   └── ascendc_kernels/binary/ascend910b/gelu/
│       ├── Gelu_<hash1>.o            # float32 kernel 二进制
│       ├── Gelu_<hash1>.json         # float32 kernel 元数据
│       ├── Gelu_<hash2>.o            # float16 kernel 二进制
│       └── Gelu_<hash2>.json         # float16 kernel 元数据
└── packages/vendors/custom/          # 打包后的 vendor 目录
```

### 2.3 常见编译问题

**问题：`Permission denied: '/usr/local/Ascend/cann-8.5.0/opp/vendors/config.ini'`**

解决方法（vendors 目录权限被安装脚本修改，导致普通用户不可读）：

```bash
sudo chmod a+r /usr/local/Ascend/cann-8.5.0/opp/vendors/config.ini
```

**问题：Kernel compilation error 且无明显错误信息**

检查 `ASCEND_HOME_PATH` 是否正确设置，以及 Python 环境是否可访问 CANN 的 Python 工具链：

```bash
echo $ASCEND_HOME_PATH
python3 -c "import asc_op_compile_base"   # 验证 Python 工具链可用
```

---

## 3. 安装算子包

每次编译后都需要重新安装算子包：

```bash
cd /opt/atomgit/GELU

# 创建 build 软链接（某些测试程序硬编码了 build/ 路径）
ln -sf build_out build

# 安装算子包
sudo ASCEND_OPP_PATH=$ASCEND_OPP_PATH bash build_out/custom_opp_ubuntu_aarch64.run

# 修复权限（安装脚本可能重置 vendor 权限）
sudo chmod a+rx $ASCEND_HOME_PATH/opp/vendors/
sudo chmod -R a+rX $ASCEND_HOME_PATH/opp/vendors/custom/
```

验证安装：

```bash
ls -la $ASCEND_CUSTOM_OPP_PATH/op_api/lib/libcust_opapi.so
ls -la $ASCEND_CUSTOM_OPP_PATH/op_impl/ai_core/tbe/kernel/ascend910b/gelu/*.o
```

---

## 4. 运行测试

### 4.1 背景：libascendcl.so 的三方依赖冲突

在 CANN 8.5.0 上运行自定义算子测试程序面临一个**关键矛盾**：

```
lib64/libascendcl.so  →  DT_NEEDED  →  lib64/libruntime.so
                                         ↓ DT_NEEDED
                                      devlib/libascend_hal.so
```

- `libascendcl.so` 在 `lib64/` 中，但**传递依赖**了 `devlib/` 中的 `libascend_hal.so`
- 如果将 `devlib` 加入 `LD_LIBRARY_PATH` 再启动进程 → `aclInit()` 返回 **500000**（驱动冲突）
- 如果不加 `devlib` → `libascendcl.so` 启动时加载失败（`libascend_hal.so` 找不到）
- 在运行时 `setenv("LD_LIBRARY_PATH", ...)` 再 `dlopen` → **无效**（glibc 的 `ld.so` 只在启动时缓存 `LD_LIBRARY_PATH`）

**关键区别**：`devlib/` 目录下也有 `libascendcl.so`，但它是一个**桩库**（stub），仅用于编译链接，运行时执行会报错。

### 4.2 推荐方案：纯 dlopen 方式

推荐使用 `dlopen` 方式构建测试程序，避免在进程启动时加载任何 lib64 中传递依赖 devlib 的库。

**编译**（不链接任何 CANN 库）：

```bash
gcc -o prof_gelu prof_gelu.c -lm -ldl
```

**加载顺序**（至关重要）：

```
1. dlopen("libascend_all_ops.so", RTLD_LAZY | RTLD_GLOBAL)
   └── 安全：该库只依赖 lib64 中的基础库，无 devlib 传递依赖

2. aclInit(NULL)
   └── 此时 LD_LIBRARY_PATH 中不含 devlib，不会冲突

3. dlopen("libascend_hal.so", RTLD_LAZY | RTLD_GLOBAL)
   └── aclInit 完成后，可以安全加载 devlib 库

4. dlopen("libascendcl.so", RTLD_LAZY | RTLD_GLOBAL)
   └── 此时 libascend_hal.so 已在符号表中，传递依赖可解析

5. dlsym 获取 aclrtMalloc / aclrtMemcpy / aclopExecuteV2 等函数指针

6. 调用 aclrtSetDevice → aclrtCreateContext → aclopExecuteV2
```

**完整代码示例**（`prof_gelu.c` 的核心结构）：

```c
// 1. 只加载算子注册库（安全）
dlopen("libascend_all_ops.so", RTLD_LAZY | RTLD_GLOBAL);

// 2. 初始化 ACL（无 devlib 冲突）
aclInit(NULL);
aclrtSetDevice(0);
aclrtCreateContext(&ctx, 0);
aclrtCreateStream(&stream);

// 3. 加载 devlib 库（aclInit 之后安全）
dlopen("libascend_hal.so", RTLD_LAZY | RTLD_GLOBAL);
dlopen("libascendcl.so", RTLD_LAZY | RTLD_GLOBAL);
// ... dlsym 获取函数指针

// 4. 运行算子
aclopExecuteV2("Gelu", ...);
```

### 4.3 不推荐的方案：静态链接 + devlib 在 LD_LIBRARY_PATH

```bash
# ❌ 不推荐：会导致 aclInit 返回 500000
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/aarch64-linux/devlib/linux/aarch64
./test_program
```

---

## 5. Profiling 性能分析

### 5.1 使用模拟器模式（板上模式不可用）

本环境不支持板上 Profiling（`msprof op` 报错 "Device profiling is not supported"），但**模拟器模式**可用：

```bash
msprof op simulator --soc-version=Ascend910B4 \
    --config=<json_config> \
    --aic-metrics="PipeUtilization" \
    --output=<output_dir>
```

### 5.2 JSON 配置文件编写

参考官方文档格式，为每个 kernel 编写一个 JSON 文件：

```json
{
    "kernel_name": "Gelu_f5fab30ba526d74aff03e6498ad42e59_0",
    "kernel_path": "/path/to/Gelu_<hash>.o",
    "blockdim": 24,
    "mode": "ca",
    "device_id": 0,
    "magic": "RT_DEV_BINARY_MAGIC_ELF_AIVEC",
    "test_cases": [
        {
            "case_name": "Gelu_float32_65536",
            "param_desc": [
                {
                    "param_type": "input",
                    "type": "float32",
                    "shape": [65536],
                    "data_path": "/path/to/input.bin",
                    "name": "input_x"
                },
                {
                    "param_type": "output",
                    "type": "float32",
                    "shape": [65536],
                    "name": "output"
                },
                {
                    "param_type": "tiling",
                    "tiling_data_size": 44,
                    "tiling_data_path": "/path/to/tiling.bin"
                }
            ]
        }
    ]
}
```

**参数字段说明**：

| 字段 | 说明 | 必填 |
|------|------|------|
| `kernel_name` | kernel 函数名（`.o` 中的符号名，带 `_0` 后缀） | 是 |
| `kernel_path` | `.o` 文件绝对路径 | 是 |
| `blockdim` | 使用的 AI Core 数（910B4 最大 24） | 是 |
| `mode` | `ca` 表示性能仿真模式 | 是 |
| `magic` | Vector 算子用 `RT_DEV_BINARY_MAGIC_ELF_AIVEC` | 是 |
| `param_type` | `input` / `output` / `tiling` / `workspace` | 是 |
| `tiling_data_size` | tiling 数据大小（字节），可通过 `.json` 元数据的 `opParaSize` 获得 | tiling 时必填 |
| `tiling_data_path` | tiling 二进制文件路径 | tiling 时必填 |

### 5.3 生成 tiling 数据

Tiling 数据是 Host 侧 `TilingFunc` 计算出的分块参数，以二进制文件形式传递给模拟器。GELU 算子的 `GeluTilingData` 结构体：

```c
struct GeluTilingData {
    uint32_t dim0;                   // 元素总数
    uint32_t coreNum;                // 实际使用的核数
    uint32_t blockFormer;            // 每核处理元素数（512 元素对齐）
    uint32_t blockNum;               // 总 block 数
    uint32_t ubFormer;               // 每 UB 块元素数（256B 对齐）
    uint32_t ubLoopOfFormerBlock;    // 首 block 的 UB 完整循环次数
    uint32_t ubTailOfFormerBlock;    // 首 block 的 UB 尾段元素数
    uint32_t ubLoopOfTailBlock;      // 末 block 的 UB 完整循环次数
    uint32_t ubTailOfTailBlock;      // 末 block 的 UB 尾段元素数
};
```

生成 tiling 数据的 Python 脚本示例（`gen_tiling.py`）：

```python
import struct

def gen_tiling(total_length, dtype_is_fp16):
    elem_size = 2 if dtype_is_fp16 else 4
    total_bytes = total_length * elem_size

    # 常量设置（需与 op_host/gelu.cpp 中的 TilingFunc 一致）
    ELEM_ALIGN = 512
    UB_ALIGN_BYTES = 256
    MIN_CORE_BYTES = 4096
    MAX_CORE_NUM = 24
    UB_SIZE = 192 * 1024

    align_elements = ELEM_ALIGN // elem_size
    ub_align_elements = UB_ALIGN_BYTES // elem_size

    # 动态核数
    core_num = min(max((total_bytes + MIN_CORE_BYTES - 1) // MIN_CORE_BYTES, 1), MAX_CORE_NUM)

    # 每核元素数（512 对齐）
    block_former = ((total_length + core_num - 1) // core_num + align_elements - 1) // align_elements * align_elements
    if block_former < align_elements:
        block_former = align_elements

    # 总 block 数
    block_num = (total_length + block_former - 1) // block_former

    # UB 切分
    buffer_divisor = 20 if dtype_is_fp16 else 24
    max_ub_elements = UB_SIZE * 95 // 100 // buffer_divisor
    ub_former = (max_ub_elements // ub_align_elements) * ub_align_elements
    ub_former = min(max(ub_former, ub_align_elements), block_former)

    # UB 循环
    ub_loop_of_former_block = block_former // ub_former
    ub_tail_of_former_block = block_former % ub_former

    tail_block_elements = total_length - (block_num - 1) * block_former
    ub_loop_of_tail_block = tail_block_elements // ub_former
    ub_tail_of_tail_block = tail_block_elements % ub_former

    return struct.pack('<IIIIIIIII',
        total_length, core_num, block_former, block_num,
        ub_former, ub_loop_of_former_block, ub_tail_of_former_block,
        ub_loop_of_tail_block, ub_tail_of_tail_block)
```

> **注意**：`opParaSize` 可能大于结构体大小（36 字节），多出的字节用 `\x00` 填充到 44 字节。

### 5.4 Profiling 运行命令

```bash
export ASCEND_CUSTOM_OPP_PATH=$ASCEND_HOME_PATH/opp/vendors/custom
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:\
$ASCEND_HOME_PATH/aarch64-linux/simulator/Ascend910B4/lib:\
$ASCEND_HOME_PATH/aarch64-linux/devlib/linux/aarch64

msprof op simulator \
    --config=/path/to/prof_config.json \
    --aic-metrics="PipeUtilization" \
    --output=/path/to/prof_out
```

### 5.5 输出产物

```
prof_out/OPPROF_<timestamp>/
├── dump/
│   ├── aicore_binary.o
│   ├── object_dump.txt
│   └── pc_start_addr.txt
└── simulator/
    ├── core0.veccore0/
    │   ├── core0.veccore0_instr_exe.csv   # 指令级流水线统计
    │   ├── core0.veccore0_code_exe.csv    # 代码热点（需 -g 编译）
    │   └── trace.json                      # 时序流水线（Chrome tracing）
    ├── core1.veccore0/
    ├── ...
    ├── trace.json                          # 全局时序流水线
    └── visualize_data.bin                  # 可视化数据
```

---

## 6. 结果解读与瓶颈定位

### 6.1 核心执行概况

Profiling 输出会显示每个核的执行时间：

```
core_name           duration_time(us)   running_time(us)    
core0.veccore0      1.11                0.78                
core1.veccore0      1.11                0.78                
```

- `running_time(us)`：实际计算时间
- `duration_time(us)`：包含调度等待的总时间
- 各核时间应基本一致（差异 >10% 表示负载不均衡）

### 6.2 指令流水线分解

使用 `instr_exe.csv` 分析各流水线占比：

```bash
# 按流水线统计指令数和周期数
awk -F',' 'NR>1 {pipes[$3]+=$4; cycles[$3]+=$5; time[$3]+=$6} \
    END {for (p in pipes) \
        printf "%-20s count=%-8d cycles=%-10d time_us=%.3f\n", \
        p, pipes[p], cycles[p], time[p]}' \
    simulator/core0.veccore0/core0.veccore0_instr_exe.csv
```

### 6.3 瓶颈识别方法

| 特征 | 瓶颈类型 | 优化方向 |
|------|---------|---------|
| SCALAR 占比 > 80% | 控制流/地址计算瓶颈 | 减少函数调用、优化 DMA 参数设置、增大 tile 粒度 |
| VECTOR 占比 > 50% | 计算瓶颈 | 优化算法、减少精度转换、使用更高效的近似 |
| MTE2 占比 > 30% | 访存/带宽瓶颈 | 减少 DMA 次数、合并小搬运、增大 UB buffer |
| 核间时间差异 >10% | 负载不均衡 | 优化核数分配、调整 tiling 策略 |

### 6.4 最耗时指令分析

```bash
# 按周期降序排列指令
sort -t, -k5 -rn simulator/core0.veccore0/core0.veccore0_instr_exe.csv | head -20
```

常见高耗时指令类型：

| 指令 | 含义 | 可能原因 |
|------|------|---------|
| `STI_XN_IMM` | 存储立即数到寄存器 | 大量常量加载（struct 初始化、地址偏移） |
| `LDP_XI_XJ_XN` | 加载寄存器对 | GlobalTensor 地址计算、函数参数传递 |
| `STP_XI_XJ_XN` | 存储寄存器对 | 函数调用边框保存/恢复 |
| `MOV_OUT_TO_UB` | OUT→UB 数据搬运 | VECOUT 到 VECIN/VECCALC 的路由 |

---

## 7. 优化实践记录

### 7.1 已尝试的优化及效果

#### 优化 A：去除 TQue，改用直连 VECIN/VECOUT Buffer

**结果**：❌ 无改善（指令数 108 → 111）

**分析**：SCALAR 瓶颈来自函数调用边框和 `this` 指针开销，而非 TQue 管理。

#### 优化 B：对齐数据用 DataCopy 替代 DataCopyPad

**结果**：❌ 基本无改善（指令数 108 → 108）

#### 优化 C：全内联 — 消除函数调用链和 this 指针（✅ 推荐）

**思路**：将 `KernelGelu` 类的所有方法（Init / Process / CopyIn / CopyOut / Compute / ComputeTanh5）全部内联到 kernel 入口函数 `gelu()` 中，使用局部变量替代成员变量。

**结果**：✅ **1.81x 加速**

| 指标 | v1 原始（类+方法） | v4 全内联 | 改善 |
|------|-------------------|-----------|------|
| 总指令数 | 108 | **53** | **−51%** |
| SCALAR 周期 | 8,186 | **1,104** | **−86%** |
| VECTOR 周期 | 64 | **32** | **−50%** |
| 每核执行时间 | 0.76 μs | **0.42 μs** | **1.81x** |
| 总吞吐量（24核） | 84 G元素/s | **152 G元素/s** | **1.81x** |

**根因**：AscendC 编译器对类成员函数的 `__aicore__ inline` **并没有完全内联**。`this` 指针的解引用、成员变量的间接访问、多级函数调用链的边框保存/恢复产生了大量 SCALAR 指令。将逻辑直接写入 kernel 入口函数后，编译器能全局优化整个数据流。

### 7.2 建议的下一步优化方向

#### 方向 1：增大 Tiling 每核处理量

当前 `blockFormer=2,816` 元素/核。增大 `MIN_CORE_BYTES` 可让每核处理更多数据，摊薄 SCALAR 固定开销。

```cpp
// op_host/gelu.cpp
constexpr uint32_t MIN_CORE_BYTES = 8192;  // 从 4096 翻倍
```

#### 方向 2：编译加 `-g` 生成代码热点映射

```cmake
ascendc_compile_options(ascendc_kernels_${RUN_MODE} PRIVATE
    -g
    -O2
)
```

#### 方向 3：在真实 NPU 上板上评测

```bash
msprof op --output=./prof_out --ai-core=on \
    --aic-metrics="PipeUtilization,Memory" \
    ./test_program blockdim 24
```

#### 方向 4：使用多项式近似替代 Tanh

```
GELU(x) ≈ x · σ(1.702x)   // sigmoid 近似，避免 tanh
```

---

## 8. 踩坑记录

### 8.1 `LD_LIBRARY_PATH` 中的 devlib 与 `aclInit` 冲突

**现象**：

```
aclInit(NULL) → 500000
```

**原因**：`devlib/linux/aarch64/libascend_hal.so` 在加载时会初始化驱动层，与 `aclInit` 的驱动初始化冲突。如果在进程启动时通过 `LD_LIBRARY_PATH` 或 `RPATH` 加载了 `libascend_hal.so`，则 `aclInit` 会返回 500000。

**解决方案**：
1. 进程启动时不将 devlib 加入 `LD_LIBRARY_PATH`
2. 使用纯 `dlopen` 方式加载库（见 [4.2 节](#42-推荐方案纯-dlopen-方式)）
3. 加载顺序：`libascend_all_ops.so` → `aclInit` → 再 dlopen devlib 库

### 8.2 `setenv("LD_LIBRARY_PATH", ...)` 不影响 `dlopen`

**现象**：运行时修改 `LD_LIBRARY_PATH` 后 `dlopen` 仍然找不到库。

**原因**：glibc 的 `ld.so` 只在进程启动时读取并缓存 `LD_LIBRARY_PATH`。运行时的 `setenv` 不影响已缓存的搜索路径。

**解决方案**：使用绝对路径调用 `dlopen`，并在之前手动加载传递依赖的库（使用 `dlopen` 而非依赖 `LD_LIBRARY_PATH`）。

### 8.3 `libascendcl.so` 在 devlib 中为桩库

**现象**：使用 `devlib/linux/aarch64/libascendcl.so` 时运行报错 "stub library cannot be used for execution"。

**原因**：devlib 中的 `libascendcl.so` 仅包含符号表，用于编译链接，无实际执行逻辑。运行时必须使用 `lib64/libascendcl.so`。

**检查方法**：

```bash
# devlib 版：只依赖基础系统库（stub）
readelf -d /usr/local/Ascend/cann-8.5.0/aarch64-linux/devlib/linux/aarch64/libascendcl.so | grep NEEDED
# 输出：libstdc++.so.6, libm.so.6, libgcc_s.so.1, ...

# lib64 版：依赖完整的 CANN 库
readelf -d /usr/local/Ascend/cann-8.5.0/lib64/libascendcl.so | grep NEEDED
# 输出：libmsprofiler.so, libascend_dump.so, libruntime.so, ...
```

### 8.4 Kernel 编译时 vendors/config.ini 权限不足

**现象**：

```
PermissionError: [Errno 13] Permission denied: '/usr/local/Ascend/cann-8.5.0/opp/vendors/config.ini'
```

**原因**：安装算子包后，vendors 目录权限被重置，`config.ini` 对普通用户不可读。

**修复**：

```bash
sudo chmod a+r /usr/local/Ascend/cann-8.5.0/opp/vendors/config.ini
```

### 8.5 模拟器 Profiling 的局限性

1. **指令计数不代表真实运行时间**：模拟器以固定 tick 周期计算，与真实 NPU 的时钟频率/流水线调度不同
2. **SCALAR 开销在模拟器上被放大**：真实硬件中，SCALAR 指令与 VECTOR 指令可并行执行，模拟器可能高估 SCALAR 的占比
3. **板上 Profiling 不可用**：本环境不支持 `msprof op` 板上模式，只能使用模拟器
4. **需要加 `-g` 编译选项**：否则无法生成代码热点映射（`code_exe.csv` 为空）

---

## 附录

### A. 快速参考命令

```bash
# 完整编译+安装+Profiling 流程
cd /opt/atomgit/GELU

# 1. 编译
bash build.sh

# 2. 安装
ln -sf build_out build
sudo ASCEND_OPP_PATH=$ASCEND_OPP_PATH bash build_out/custom_opp_ubuntu_aarch64.run
sudo chmod a+rx $ASCEND_HOME_PATH/opp/vendors/
sudo chmod -R a+rX $ASCEND_HOME_PATH/opp/vendors/custom/

# 3. 生成 tiling 数据
python3 gen_tiling.py

# 4. Profiling
export ASCEND_CUSTOM_OPP_PATH=$ASCEND_HOME_PATH/opp/vendors/custom
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:\
$ASCEND_HOME_PATH/aarch64-linux/simulator/Ascend910B4/lib:\
$ASCEND_HOME_PATH/aarch64-linux/devlib/linux/aarch64

msprof op simulator \
    --config=prof_config_f32.json \
    --aic-metrics="PipeUtilization" \
    --output=prof_out

# 5. 分析结果
find prof_out -name "*.csv" -exec head -30 {} \;
```

### B. 相关文件清单

| 文件 | 用途 |
|------|------|
| `op_host/gelu.cpp` | Host 侧 Tiling 函数 + 算子注册 |
| `op_kernel/gelu.cpp` | Device 侧 Kernel（计算核心） |
| `op_kernel/gelu_tiling.h` | Tiling 数据结构体 |
| `op_kernel/tiling_key_gelu.h` | TilingKey 模板定义 |
| `prof_gelu.c` | Profiling 测试程序（纯 dlopen 方式） |
| `prof_config_f32.json` | float32 Profiling JSON 配置 |
| `gen_tiling.py` | Tiling 数据生成脚本 |
| `prof_data/` | Profiling 输入数据目录 |
| `build.sh` | 编译脚本 |
| `run_gelu.c` | 旧版测试程序（aclopExecuteV2） |
| `run_gelu_aclnn.c` | 旧版测试程序（aclnnGelu） |

# LessEqual 算子 — Ascend C 自定义算子工程

## 项目概述

基于 Ascend C 编程语言实现的 **LessEqual** 算子，在昇腾 910B NPU 上执行逐元素比较 `x1 <= x2`，返回 `bool` 类型结果张量。

### 算子规格

| 项目 | 说明 |
|------|------|
| **算子名称** | LessEqual |
| **参考算子** | `tf.math.less_equal` |
| **输入类型** | float16, float32, int32, int8 |
| **输出类型** | bool (uint8) |
| **广播支持** | NumPy 风格广播语义 |
| **非对齐支持** | 支持非 32 字节对齐场景 |

## 项目结构

```
less_equal/
├── README.md                    # 本文件
├── CMakeLists.txt               # NPU bisheng 编译配置
├── less_equal_custom.asc        # 全功能算子 (Kernel + Host + Main)
├── cpu_debug/                   # CPU 仿真调试
│   ├── CMakeLists.txt           # GCC + tikicpulib 编译
│   ├── less_equal_test.cpp      # CPU 仿真测试程序
│   ├── data_utils.h             # 数据读写工具
│   └── build/                   # 编译输出
├── op_kernel/                   # 核函数实现 (独立 .h + .cpp)
│   ├── less_equal_kernel.h
│   └── less_equal_kernel.cpp
├── scripts/                     # Python 测试脚本
│   ├── gen_data.py              # 测试数据生成
│   └── verify_result.py         # 结果验证
└── build/                       # NPU 编译输出
```

## 快速开始

### 1. CPU 仿真调试（本地验证，无需 NPU 硬件）

```bash
cd cpu_debug
mkdir -p build && cd build
cmake .. -Dsoc_version=Ascend910B4
make -j
./less_equal_test
```

输出示例：
```
[INFO]  ==============================================
[INFO]    LessEqual CPU Debug Tests
[INFO]    Cores=8  Total=16384  Block=2048  Tile=128
[INFO]  ==============================================
[INFO]  [PASS] float32: all 16384 correct
[INFO]  [PASS] float16: all 16384 correct
[INFO]  [PASS] int32: all 16384 correct
[INFO]  [PASS] int8: all 16384 correct
[INFO]    ALL TESTS PASSED!
```

### 2. NPU 编译（需昇腾硬件运行）

```bash
# 设置环境
source $ASCEND_TOOLKIT_HOME/set_env.sh

# 编译
mkdir -p build && cd build
cmake .. && make -j

# 运行（需要 NPU 硬件）
./demo
```

预期输出：
```
Output first 10: 1 1 1 1 1 1 1 1 1 1
Match: 16384 / 16384
[Success] Case accuracy verification passed.
```

### 3. 生成测试数据

```bash
# 生成 float32 数据
python3 scripts/gen_data.py --dtype float32 --shape 8,2048

# 生成广播测试数据
python3 scripts/gen_data.py --dtype float32 --shape 2,3,4 --broadcast

# 验证结果
python3 scripts/verify_result.py output.bin golden.bin
```

## 核心实现

### 核函数架构

```
less_equal_custom()            ← 核函数入口 (__global__ __aicore__)
  ├── dtype dispatch           ← 按 4 种类型分发
  └── KernelLessEqual<T>      ← 模板算子类 (Vector 范式)
        ├── Init()             ← 多核数据划分 + Pipeline 初始化
        ├── Process()          ← 流水线主循环
        │     ├── CopyIn()     ← GM → LM 数据搬运
        │     ├── Compute()    ← 比较运算 (x1 <= x2)
        │     └── CopyOut()    ← LM → GM 结果搬运
```

### 关键 API

| API | 用途 |
|-----|------|
| `AscendC::Compare` (CMPMODE::LE) | 矢量比较 (float/half) |
| `AscendC::DataCopy` | Global ↔ Local Memory 数据搬运 |
| `AscendC::TPipe` / `TQue` | Pipeline 流水线管理 |
| `ICPU_RUN_KF` | CPU 仿真核函数调用入口 |

## VSCode 快捷编译

打开 `less_equal_custom.asc` 或 `cpu_debug/less_equal_test.cpp` 后：
- `Ctrl+Shift+B` — 一键编译（在当前文件目录下 cmake + make）

## 环境要求

- CANN Toolkit 8.5.2+
- bisheng 编译器 (含于 CANN)
- GCC 9.4+ (CPU 调试)
- CMake 3.16+
- Python 3.7+ (测试脚本)

## 芯片类型

昇腾 910B (dav-2201)

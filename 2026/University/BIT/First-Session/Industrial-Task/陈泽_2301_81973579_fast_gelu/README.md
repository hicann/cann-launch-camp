# FastGelu 算子课题成果

北京理工大学 CANN 创新实训营 · 第一期产业课题

## 作者信息

- 姓名：陈泽
- GitCode 账号：2301_81973579
- CANNJudge 题目：[FastGelu](https://cannjudge.cn/bit/public/public)

## 项目简介

基于 Ascend C 在昇腾 910B 上实现 FastGelu 激活函数算子，支持 `float16` / `float32`，输出与 MindSpore `ops.fast_gelu` 对齐。

计算公式：

```
y = x * exp(0.851 * (x - |x|)) / (1 + exp(-1.702 * |x|))
```

## 目录结构

```
陈泽_2301_81973579_fast_gelu/
├── README.md           # 本说明
├── build.sh            # Linux 编译脚本
├── build.bat           # Windows 编译脚本
└── code/               # Ascend C 算子工程（CANNJudge 同款结构）
    ├── CMakeLists.txt
    ├── op_host/
    │   ├── CMakeLists.txt
    │   └── fast_gelu.cpp
    └── op_kernel/
        ├── CMakeLists.txt
        ├── fast_gelu.cpp
        ├── fast_gelu_tiling.h
        └── tiling_key_fast_gelu.h
```

## 实现说明

| 模块 | 说明 |
|------|------|
| `op_host/fast_gelu.cpp` | 泛化 Tiling：多核负载均衡、UB 分块、形状/类型推导 |
| `op_kernel/fast_gelu.cpp` | 双缓冲流水线 + `AscendC::FasterGelu` 高阶 API |
| `op_kernel/fast_gelu_tiling.h` | Tiling 参数结构体 |

## 环境要求

- CANN Toolkit **8.5.0**（与 OJ 一致）
- 昇腾 **910B**（Atlas A2 训练系列）
- CMake >= 3.16

## 编译方法

### Linux / 昇腾开发环境

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
chmod +x build.sh
./build.sh
```

### Windows（仅整理代码，实际编译需在 Linux + CANN 环境）

双击或在 cmd 中执行 `build.bat`（会提示需在 Linux 环境编译）。

### 手动编译

```bash
cd code
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

## CANNJudge 在线判题

1. 打开 https://cannjudge.cn/bit/public/public
2. 将 `code/op_host/fast_gelu.cpp`、`code/op_kernel/fast_gelu.cpp`、`code/op_kernel/fast_gelu_tiling.h` 三个文件内容粘贴到 OJ 编辑器
3. 点击提交，等待 5 个测试用例判题

## 精度要求

- float32：绝对/相对误差 < 1e-4
- float16：绝对/相对误差 < 1e-3

## 参考

- [MindSpore fast_gelu](https://www.mindspore.cn/docs/zh-CN/r1.8/api_python/ops/mindspore.ops.fast_gelu.html)
- [CANN FasterGelu API](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0772.html)

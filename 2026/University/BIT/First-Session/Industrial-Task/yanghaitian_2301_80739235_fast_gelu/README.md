# FastGelu Ascend C 算子提交说明

提交人：yanghaitian  
GitCode 账号：2301_80739235  
任务：fast_gelu

## 1. 实现内容

本目录提交的是 FastGelu Ascend C 自定义算子工程源码，基于题目要求实现逐元素计算：

```text
y = x * exp(0.851 * (x - abs(x))) / (1 + exp(-1.702 * abs(x)))
```

支持能力：

- 输入/输出 dtype：`float16`、`float32`
- 输入/输出 format：`ND`
- 输出 shape 与输入 shape 一致
- 兼容非 32 字节对齐尾部数据
- 支持多核切分与双缓冲队列

## 2. 目录结构

```text
yanghaitian_2301_80739235_fast_gelu/
|-- README.md
|-- build.sh
`-- code/
    |-- CMakeLists.txt
    |-- op_host/
    |   |-- CMakeLists.txt
    |   `-- fast_gelu.cpp
    `-- op_kernel/
        |-- CMakeLists.txt
        |-- fast_gelu.cpp
        |-- fast_gelu_tiling.h
        `-- tiling_key_fast_gelu.h
```

## 3. 实现要点

- Host 侧根据输入总元素数、AI Vector Core 数量和 UB 大小计算 `blockDim` 与 `tileSize`。
- Kernel 侧使用 `TQue<VECIN, 2>` 和 `TQue<VECOUT, 2>`，通过 `pipe.InitBuffer(..., 2, ...)` 开启 double buffer。
- `float32` 路径使用 CANN AscendC `FasterGelu<float, false, false>`，避免 `highPerformance=true` 带来的精度风险。
- `float16` 路径使用 float 中间计算：`Cast half->float`、`Muls`、`Exp`、`Adds`、`Div`、`Cast float->half`，在保证精度的同时减少不必要的指数计算。
- 使用 `DataCopyPad` 处理非 32 字节对齐尾部，直接 `DataCopy` 处理 32 字节对齐分块。

## 4. 编译方式

在具备 CANN/AscendC 环境的机器上执行：

```bash
cd yanghaitian_2301_80739235_fast_gelu
bash build.sh
```

或手动执行：

```bash
cd code
mkdir -p build
cd build
cmake ..
make -j
```

## 5. 本地验证

本地使用 CANN 9.0.0 做过以下验证：

- `opbuild`/打包流程通过，生成 Ascend910B 的 `float16` 和 `float32` kernel。
- CPU 仿真覆盖 `float16` / `float32`。
- CPU 仿真覆盖长度 `0, 1, 4, 17, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1000`。
- 覆盖非 32 对齐长度和多核尾块切分场景。

最终性能与正确性以 CANNJudge / 真实 Ascend 910B NPU 评测结果为准。

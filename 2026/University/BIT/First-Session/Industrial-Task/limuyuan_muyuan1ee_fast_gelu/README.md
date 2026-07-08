## FastGelu Ascend C 自定义算子

基于 Ascend C 实现 FastGelu 激活算子，运行于 Ascend 910B。

### 算子公式

$$y = \frac{x \cdot e^{0.851(x-|x|)}}{1 + e^{-1.702|x|}}$$

由 `0.851 × 2 = 1.702`，上式可等价化简为 $y = x \cdot \mathrm{sigmoid}(1.702\,x)$，kernel 据此实现。

### 实现说明

- Host：按 dtype 与 UB 容量自适应切分 tile，多核均分负载。
- Kernel：`Muls → Sigmoid → Mul`，UB 双缓冲驱动 MTE2 / V / MTE3 流水。
- 非对齐：满 tile 走 `DataCopy`，仅末 tile 走 `DataCopyPad`。
- fp16 内部升 fp32 计算；空张量直接 no-op。

### 目录结构

```
├── CMakeLists.txt
├── op_host/        # Tiling + 算子注册 + Shape/DType 推导
│   └── fast_gelu.cpp
└── op_kernel/      # 核函数 + tiling 结构体 + 模板特化
    ├── fast_gelu.cpp
    ├── fast_gelu_tiling.h
    └── tiling_key_fast_gelu.h
```

### 精度与环境

- float32 / float16，任意非对齐 ND 维度。
- 精度：fp32 < 1e-4，fp16 < 1e-3（相对 / 绝对）。
- CANN 8.5.0 · Ascend 910B · CMake ≥ 3.16

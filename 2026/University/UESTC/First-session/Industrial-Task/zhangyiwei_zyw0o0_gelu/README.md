# Gelu Ascend C 自定义算子

本目录实现赛题 **GELU 算子**，目标芯片为 **Ascend 910B**。算子对输入张量逐元素计算 GELU 激活函数，输出张量与输入保持相同 shape、dtype 和 ND 数据格式。

## 功能说明

- 算子名称：`Gelu`
- 输入：`input_x`
- 输出：`output`
- 支持数据类型：`float16`、`float32`
- 支持数据格式：`ND`
- 支持非 32 整倍数元素数量，尾块通过 `DataCopyPad` 处理
- 输出 shape 与输入 shape 完全一致

GELU 的精确定义为：

```text
GELU(x) = x * Phi(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

其中 `Phi(x)` 是标准正态分布累积分布函数。

当前实现根据 dtype 选择计算路径：

- `float32`：使用分段裁剪加多项式/指数形式近似，兼顾吞吐与精度。
- `float16`：使用常见 tanh 近似形式进行原生 FP16 向量计算。

## 目录结构

```text
.
├── CMakeLists.txt
├── README.md
├── op_host
│   ├── CMakeLists.txt
│   └── gelu.cpp
├── op_kernel
│   ├── CMakeLists.txt
│   ├── gelu.cpp
│   ├── gelu_tiling.h
│   └── tiling_key_gelu.h
└── scripts
    ├── build.ps1
    ├── build.sh
    ├── clean.ps1
    └── clean.sh
```

## 环境要求

请在已安装 CANN/Ascend C 开发环境的机器上构建，并确保已加载 CANN 环境变量。Linux 环境通常执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

如果环境变量没有自动设置，可以在执行构建脚本时显式传入 CANN toolkit 路径。

## 构建方法

### Linux

```bash
cd 2026/University/UESTC/First-session/Industrial-Task/zhangyiwei_zyw0o0_gelu
bash scripts/build.sh
```

指定 CANN toolkit 路径：

```bash
bash scripts/build.sh --cann-path /usr/local/Ascend/ascend-toolkit/latest
```

### Windows PowerShell

```powershell
cd 2026\University\UESTC\First-session\Industrial-Task\zhangyiwei_zyw0o0_gelu
.\scripts\build.ps1
```

指定 CANN toolkit 路径：

```powershell
.\scripts\build.ps1 -CannPath "C:\Ascend\ascend-toolkit\latest"
```

默认构建目录为 `build`，可以通过参数修改：

```powershell
.\scripts\build.ps1 -BuildDir build-release -BuildType Release
```

构建完成后，CMake 会在构建目录下生成自定义算子包相关产物。

## 清理构建产物

Linux：

```bash
bash scripts/clean.sh
```

Windows PowerShell：

```powershell
.\scripts\clean.ps1
```

## 运行与验证建议

本目录只包含算子工程源码，未额外附带 Python/ACL 调用样例。建议在 CANN 自定义算子包构建完成后，按比赛平台或本地 ACL/ATB/PyTorch 适配环境加载该算子，并使用 `torch.nn.functional.gelu` 作为标杆进行对比。

建议覆盖以下测试场景：

- dtype：`float32`、`float16`
- shape：标量、1 维、2 维、3 维、4 维以及带 batch 的多维输入
- 非对齐：元素数量或末维不是 32 的整倍数
- 数值：0、小正数、中等正数、大正数、小负数、中等负数、大负数、标准正态随机输入
- 特殊值：`NaN`、`+Inf`、`-Inf`

参考精度阈值：

- `float32`：绝对误差不超过 `1e-5` 或相对误差不超过 `1e-4`
- `float16`：绝对误差不超过 `1e-2` 或相对误差不超过 `1e-3`

## 实现要点

- `op_host/gelu.cpp` 负责算子注册、shape/dtype 推导和 tiling 计算。
- `op_kernel/gelu.cpp` 负责 AICore kernel 实现。
- tiling 根据总元素数、AIV 核数、UB 大小和 dtype size 切分 block 与 UB tile。
- 对齐场景使用普通 `DataCopy`，非对齐尾块使用 `DataCopyPad`。
- 输出 dtype 由 host 侧直接透传输入 dtype。


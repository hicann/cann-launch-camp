# GELU AscendC 自定义算子项目

Ascend NPU (910B4) 上的 GELU 激活函数自定义算子，使用 AscendC 开发。

## 项目结构

```
GELU/
├── op_host/gelu.cpp           # Host 侧 Tiling 实现 + 算子注册
├── op_kernel/gelu.cpp         # Kernel 侧 AscendC 实现
├── op_kernel/gelu_tiling.h    # Tiling 数据结构
├── op_kernel/tiling_key_gelu.h # TilingKey 模板定义
├── CMakeLists.txt             # 构建配置 (ascend910b)
├── test_gelu.py               # 测试数据生成 + PyTorch 参考值
├── test_data/                 # 生成的测试数据和参考结果
├── run_gelu.c                 # NPU 执行测试程序（C 版本）
└── run_gelu_aclnn.c           # NPU 执行测试程序（aclnn 版本）
```

## 构建

```bash
source /usr/local/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/aarch64-linux/lib64

cd GELU && rm -rf build && mkdir build && cd build
cmake ..
make -j$(nproc)                          # 编译 Host 库 + ops info
make Gelu_ascend910b                     # 编译 Kernel 二进制
```

### 构建产物

| 产物 | 路径 | 说明 |
|------|------|------|
| `libcust_opapi.so` | `build/op_host/` | aclnnGelu 两段式接口库 |
| `libcust_opmaster_rt2.0.so` | `build/op_host/` | Tiling 函数库 |
| `Gelu_*.o` | `build/op_kernel/.../binary/ascend910b/gelu/` | NPU kernel 二进制 |
| `aic-ascend910b-ops-info.json` | `build/op_kernel/.../tbe/op_info_cfg/` | 算子注册信息 |

## 安装算子包

```bash
cd GELU
source /usr/local/Ascend/cann/set_env.sh

PACKAGE_DIR="build/packages/vendors/custom/op_impl/ai_core/tbe"
mkdir -p "$PACKAGE_DIR/config/ascend910b" "$PACKAGE_DIR/kernel/config"
mkdir -p build/packages/vendors/custom/op_api/lib

# Host 库
cp build/op_host/libcust_opapi.so build/packages/vendors/custom/op_api/lib/
cp build/op_host/libcust_opmaster_rt2.0.so "$PACKAGE_DIR/op_tiling/lib/linux/aarch64/"

# Kernel 二进制
cp build/op_kernel/ascendc_kernels/binary/ascend910b/gelu/*.o "$PACKAGE_DIR/kernel/ascend910b/gelu/"
cp build/op_kernel/ascendc_kernels/binary/ascend910b/gelu/*.json "$PACKAGE_DIR/kernel/ascend910b/gelu/"

# Ops info
cp build/op_kernel/ascendc_kernels/tbe/op_info_cfg/ai_core/ascend910b/aic-ascend910b-ops-info.json "$PACKAGE_DIR/config/ascend910b/"
cp build/op_kernel/ascendc_kernels/tbe/op_info_cfg/ai_core/ascend910b/aic-ascend910b-ops-info.json "$PACKAGE_DIR/kernel/config/"
cp build/op_kernel/ascendc_kernels/tbe/op_info_cfg/ai_core/npu_supported_ops.json "$PACKAGE_DIR/config/"

echo "custom_opp_compiler_version=9.1.0" > build/packages/vendors/custom/version.info
```

## 在 NPU 上运行

### 编译测试程序

```bash
source /usr/local/Ascend/cann/set_env.sh
g++ -o run_gelu run_gelu.c \
    -I$ASCEND_HOME_PATH/include \
    -L$ASCEND_HOME_PATH/lib64 \
    -lascendcl -lm
```

### 执行测试

```bash
export ASCEND_CUSTOM_OPP_PATH=/path/to/GELU/build/packages/vendors
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/aarch64-linux/lib64

./run_gelu
```

### 环境要求

| 组件 | 版本要求 |
|------|---------|
| NPU 驱动 | ≥ 25.5.1（与 Toolkit 版本配套） |
| CANN Toolkit | 9.1.0（已验证） |
| 操作系统 | Linux aarch64 |
| Python | 3.8+（用于测试数据生成） |
| PyTorch | ≥ 2.0（用于参考值计算） |

## 算子详情

### 输入/输出

| 参数 | 名称 | 数据类型 | 格式 | Shape |
|------|------|---------|------|-------|
| Input | input_x | float16, float32 | ND | 任意 1D |
| Output | output | float16, float32 | ND | 同输入 |

### 编译架构

- SoC: **ascend910b** (Ascend 910B4)
- 架构: **dav-2201**
- Core 类型: **AIV** (VectorCore)
- 编译工具: bisheng (clang 15.0.5)

### Tiling 策略

- 多核切分（对齐到 512 元素）
- UB buffer 切分（对齐到 256B，留 10% 余量）
- 支持 tiny（128）到 huge（262144）各种张量规模

## 测试数据

```bash
cd GELU && python3 test_gelu.py
```

| 用例 | 元素数 | 数据类型 | 输入文件 | 参考文件 |
|------|--------|---------|---------|---------|
| tiny | 128 | float32 | `tiny_input.bin` | `tiny_reference.bin` |
| small | 2,048 | float32 | `small_input.bin` | `small_reference.bin` |
| medium | 8,192 | float16 | `medium_input.bin` | `medium_reference.bin` |
| large | 65,536 | float32 | `large_input.bin` | `large_reference.bin` |
| huge | 262,144 | float16 | `huge_input.bin` | `huge_reference.bin` |

## 已知问题

### 自定义算子加载失败

- **现象**: `aclopExecuteV2("Gelu")` 返回 `100024 (OP_NOT_FOUND)`
- **原因**: `ASCEND_CUSTOM_OPP_PATH` 机制在本环境的 CANN 社区版 + 容器部署场景中未生效
- **环境限制**: 该环境运行在 Kubernetes 容器中，NPU 驱动由宿主机提供（驱动版本 25.5.1），CANN Toolkit（9.1.0）与驱动之间的版本配套关系可能不完整

### 排查工具

| 错误码 | 含义 | 排查方向 |
|--------|------|---------|
| 100024 | OP_NOT_FOUND | 算子注册路径、ASCEND_CUSTOM_OPP_PATH |
| 161001 | PARAM_NULLPTR | 参数为空（tensor、executor） |
| 561002 | TILING_ERROR | Tiling 函数执行异常 |

## 参考资料

- [cann-samples](https://gitcode.com/cann/cann-samples) — CANN 算子实战样例
- [cannbot-skills](../cannbot-skills/) — AscendC 算子开发技能包（含环境检查、代码审查等）
- Cann-samples README 中的 CANN Toolkit 版本表（已验证 9.1.0）

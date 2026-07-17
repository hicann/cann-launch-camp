# LessEqual 算子

基于昇腾 CANN 平台的逐元素比较算子，计算 `y = (x1 <= x2)`，输出布尔类型。支持多维广播和多核并行，目标平台为 Ascend 910B。

## 支持的数据类型

- **float16** → bool
- **float32** → bool
- **int32** → bool
- **int8** → bool

## 目录结构

```text
wangyuchen_TWT0508_less_equal/
├── CMakeLists.txt                  # 顶层 CMake 配置
├── build.sh                        # 一键构建与本地测试脚本
├── README.md                       # 本文件
├── op_host/
│   ├── CMakeLists.txt              # Host 侧构建配置
│   └── less_equal.cpp              # 算子注册、形状推导、Tiling 逻辑
├── op_kernel/
│   ├── CMakeLists.txt              # Kernel 侧构建配置
│   ├── less_equal.cpp              # AI Core 核函数实现
│   ├── less_equal_tiling.h         # Tiling 数据结构定义
│   └── tiling_key_less_equal.h     # 模板参数声明（数据类型分发）
└── test/
    ├── gen_data.py                 # 生成测试数据与 NumPy 期望值
    ├── verify_result.py            # 对比算子输出与 NumPy 结果
    └── run_test.sh                 # 本地测试脚本
```

## 构建方法

需要先安装 CANN 开发环境，然后执行：

```bash
cd wangyuchen_TWT0508_less_equal
bash build.sh
```

脚本会自动定位 CANN 安装路径、配置 CMake、编译，最后运行本地 NumPy 测试验证。

## 实现要点

### 整体流程

1. **Host 侧**：获取两个输入的形状 → 推导广播输出形状 → 计算各维步长 → 维度折叠（合并相邻的同类维度） → 计算每个核的任务量 → 将参数打包传递给 Kernel。
2. **Kernel 侧**：根据 Tiling 参数从全局内存搬运数据到 UB，执行逐元素比较，将结果写回。

### 两种运行模式

- **直接模式**：两个输入形状完全一致，采用连续 DMA 搬运，效率最高。
- **广播模式**：输入形状不同，Host 侧已通过维度折叠保证最后一维始终连续（或为标量），Kernel 按行处理即可。

### 维度折叠

广播场景下，相邻维度如果在两个输入上的行为一致（同为连续，或同为标量广播），则可以合并成一个维度。折叠后维度数减少，且最后一维一定是连续段，从而避免了逐元素的 scatter/gather 操作，广播路径也能走 DMA 连续搬运。

### 各数据类型的比较方式

- **half / float**：硬件直接支持 `Compare` 指令，使用 `CMPMODE::LE`。
- **int8**：先通过 `Cast` 转换为 half 再比较。int8 的值域为 [-128, 127]，均可被 half 精确表示，不会丢失精度。
- **int32**：采用 `min(a, b) == a` 等价于 `a <= b` 的方式，避免 int32 转 float 时丢失低位精度（float 尾数仅 24 位）。

### 性能优化

1. **双缓冲队列**：输入输出队列深度均为 2，DMA 搬运与计算可并行执行。
2. **预填充常量**：ones / zeros 数组在初始化时一次性填充，后续每次比较直接复用，省去了反复调用 `Duplicate` 的开销。
3. **维度折叠**：消除逐元素偏移量计算，广播场景同样走 DMA 连续搬运。
4. **UB 预算管理**：根据数据类型和 tile 大小估算 UB 用量，自动选取不超过硬件限制的最大 tile。
5. **行级任务分配**：广播模式下将前 N-1 维乘积视为"行数"，按行均分给各核，行内数据连续。

### 测试覆盖

测试涵盖 55 个场景，包括：

- 四种数据类型的基础 1D 测试
- 2D / 3D / 4D 高维张量
- 标量广播、向量广播、高维广播
- 极值边界（各类型的 min / max）
- 全等值、全零值
- 非 32 对齐元素个数
- 大规模数据（5K ~ 10K 元素）
- 广播 + 大规模混合场景

## 许可证

CANN 算子开发模板

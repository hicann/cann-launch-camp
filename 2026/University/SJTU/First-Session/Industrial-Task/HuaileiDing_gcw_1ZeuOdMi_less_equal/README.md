# LessEqual Ascend C 自定义算子

本工程面向 Ascend 910B，实现逐元素比较算子 `y = (x1 <= x2)`。输入 `x1`、`x2` 的数据类型必须相同，支持 `float16`、`float32`、`int32`、`int8`，输出类型为 `bool`。算子支持 ND 张量、0 元素张量、非 32 字节对齐尾块以及最多 16 维的 NumPy 风格广播。

## 目录结构

```text
.
├── README.md
├── src/
│   ├── CMakeLists.txt
│   ├── op_host/
│   │   ├── CMakeLists.txt
│   │   └── less_equal.cpp
│   └── op_kernel/
│       ├── CMakeLists.txt
│       ├── less_equal.cpp
│       ├── less_equal_tiling.h
│       └── tiling_key_less_equal.h
└── test/
    ├── generate_cases.py
    ├── aclnn_runner.cpp
    ├── run_npu_test.sh
    ├── test_reference.py
    ├── verify_tensorflow.py
    └── verify_result.py
```

## 实现说明

Host 侧完成输出形状和类型推导、广播合法性检查、连续 stride 计算以及多核任务切分。广播维度的输入 stride 被设为 0，Kernel 可据此将输出坐标映射回输入地址。

Kernel 每个 AI Core 处理一段连续输出，并按 1024 个元素分块：

- 相同形状走连续搬运路径，使用 `DataCopyPad` 兼容非对齐尾块；
- 满 1024 元素的连续块使用 Ascend C 向量比较与掩码展开路径；
- 标量广播在完整块上使用专用向量路径，一般广播使用通用坐标映射路径；
- 非完整尾块使用安全的标量回退路径，`float16` 标量比较先精确提升为 `float32`；
- 布尔结果按 `uint8_t` 的 0/1 写回。

## 构建

要求已安装支持 `npu_op_package` 系列 CMake 接口的 CANN Toolkit，并加载其环境变量。示例：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S src -B src/build
cmake --build src/build -j
```

目标产品由 `src/CMakeLists.txt` 中的 `ASCEND_COMPUTE_UNIT=ascend910b` 指定。实际安装或调用方式以所用 CANN 版本生成的算子包和 ACLNN 接口为准。

## 测试数据

测试脚本依赖 Python 3 和 NumPy：

```bash
python -m pip install numpy
python test/generate_cases.py --output-dir test/data
python -m unittest test.test_reference
```

`generate_cases.py` 会生成输入二进制、布尔真值和 `cases.json`。在 NPU 调用程序将各用例结果写成 `<case_name>_y.bin` 后，可统一校验：

```bash
python test/verify_result.py \
  --manifest test/data/cases.json \
  --output-dir test/output
```

校验采用逐字节完全一致比较，不设置误差容忍。

### TensorFlow 精度对标

安装 TensorFlow 后，可用题目指定的 `tf.math.less_equal` 重新计算全部真值并做完全一致比较：

```bash
python -m pip install tensorflow
python test/verify_tensorflow.py --manifest test/data/cases.json
```

脚本对 `float16`、`float32` 等全部内置用例逐元素比较，任何一个布尔值不同都会返回失败。

## 910B 实机冒烟测试

算子构建完成后，运行 ACLNN 测试程序：

```bash
bash test/run_npu_test.sh 0
```

参数是 AscendCL 设备 ID；若算力环境未将物理设备映射为逻辑设备 0，请传入 `npu-smi info` 对应的设备 ID。测试覆盖 37 元素非对齐连续输入和 `[2,2] <= [2]` 广播输入，预期输出两行 `PASS`。

已验证环境与结果：

- CANN 8.5.2；
- Atlas 训练系列 910B3；
- 四种数据类型的 Kernel 均编译成功；
- Python 广播/索引单元测试 5 项全部通过；
- ACLNN 实机冒烟测试 `float32_non_aligned`、`float32_broadcast` 均通过。

## 覆盖范围

内置数据覆盖四种输入类型、同形状比较、标量/向量/高维广播、非 32 字节对齐长度、空张量、数据类型边界、相等值以及浮点 `NaN`/无穷值。

## 当前限制

- 最大输出 rank 为 16；
- 输出元素总数和输入线性索引必须能用 `uint32_t` 表示；
- 两个输入必须具有相同数据类型；
- 动态未知维度需要在运行前具体化。

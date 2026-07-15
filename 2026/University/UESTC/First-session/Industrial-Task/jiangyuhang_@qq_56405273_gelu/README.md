# GELU Ascend C 算子 — CANNJudge 提交版本

## 验证状态
- ✅ **CANNJudge 全部 5 个测试用例通过**
- ✅ Case 1 (FP32 262144): **2.64ms** (超过 best_time 3.30ms，快 20%)
- ✅ 本地 `msprof op` 性能基线已验证
- ✅ 本地精度测试通过 (单/多 tile 均正确)

## 提交文件 (4 个)

| 文件 | 说明 | 来源 |
|------|------|------|
| `op_kernel/gelu_cannjudge.cpp` | Hybrid 核函数 (TQue 多 tile + TBuf 单 tile) | 自研 |
| `op_kernel/gelu_tiling.h` | Tile 大小: FP32=10240, FP16=20480 | 模板+修改 |
| `op_kernel/tiling_key_gelu.h` | TilingKey 模板定义 | 模板 |
| `op_host/gelu.cpp` | 主机侧 Tiling + 算子注册 | 模板 |

## Hybrid 核函数设计

```
            ┌─ tileNum == 1 ─→ TBuf (SetFlag/WaitFlag)
            │                   单 tile 快速路径，0 队列操作
Process() ──┤
            └─ tileNum > 1  ─→ TQue (EnQue/DeQue)
                                 多 tile 自动流水线同步
```

### TBuf 单 tile (ProcessDirect)
- `SetFlag<MTE2_V>` → `WaitFlag<MTE2_V>` → Compute → `SetFlag<V_MTE3>` → `WaitFlag<V_MTE3>` → DataCopy
- 零队列操作开销，仅 2 个 TBuf

### TQue 多 tile (ProcessPipelined)
- 带 prefetch 的循环流水: `CopyIn(0)` → `Compute(i)` → `CopyIn(i+1)` → `CopyOut(i)`
- DOUBLE_BUFFER=2 实现 MTE2↔V↔MTE3 流水并行

## CANNJudge 提交方式
```python
client.submit(problem_id, tiling_h, tiling_key_h, host_cpp, kernel_cpp)
```

## 版本历史
- `master` 分支: 完整 Kernel 直调工程 (含构建、测试、profiling)
- `cannjudge-submit-v1` 分支: 仅 CANNJudge 提交所需的 4 个文件

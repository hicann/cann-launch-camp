# 基于强化学习的算子优化探索

---

## 1. 背景与动机

### 1.1 当前优化挑战

- 黑盒环境，无编译日志
- 评测噪声大（波动 0.3-2 μs）
- 参数空间大，组合爆炸
- 优化依赖专家经验

### 1.2 RL 潜力

- 自动探索参数空间
- 从反馈学习最优策略
- 发现人类忽略的模式

---

## 2. 问题建模

### 2.1 状态空间

```
状态 = [
    totalLength, dtype, rank, broadcastMode,
    tileLength, usedCoreNum, bufferNum, mode,
    ubSize, platformCoreNum
]
```

### 2.2 动作空间

```
动作 = {
    调整 tileLength,
    调整 usedCoreNum,
    切换 bufferNum,
    切换执行模式
}
```

### 2.3 奖励函数

```python
reward = baseline - performance
```

---

## 3. 算法设计

推荐 PPO（Proximal Policy Optimization）

### 3.1 网络架构

```
状态编码器 → Actor 网络（输出动作概率）
          → Critic 网络（输出价值估计）
```

### 3.2 训练流程

```
循环：
  采样状态 → 选择动作 → 生成代码 → 评测 → 计算奖励 → 更新策略
```

---

## 4. 关键设计

### 4.1 噪声处理

多次提交取中位数

### 4.2 安全探索

约束动作偏离基线程度

---

详细探索方案请参考 `finaltest` 目录下的完整文档。
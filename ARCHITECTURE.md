# 系统架构文档

## 数据流

```
初始空间配置
    ↓
[WorldState] 状态表示
    ↓
[RasterObservation] 编码为栅格观测（CNN处理）
    ↓
[CNN Encoder] 状态压缩（CNN）
    ↓
[Policy/Value MLP] 输出动作分布和价值
    ↓
[ActionSpace] 采样合法动作
    ↓
[Transition] 执行动作，更新状态
    ↓
[FeatureExtractor] 提取STGNN特征
    ↓
[STGNNWrapper] 预测客流 F_hat
    ↓
[VitalityMetrics] 计算活力指标 V
    ↓
[RewardCalculator] 计算奖励 R = ΔV - λ·Cost - μ·Violation
    ↓
[TerminationChecker] 检查终止条件
    ↓
[EpisodeLogger] 记录轨迹
```

## 模块依赖关系

```
main.py
├── PPOTrainer (rl/train_ppo.py)
│   ├── CNNActor (rl/policy_cnn.py) - CNN编码器 + MLP策略头
│   ├── ValueHead (rl/value_head.py) - MLP价值函数
│   └── EpisodeLogger (logging/episode_logger.py)
│
└── Environment Loop
    ├── WorldState (env/world_state.py)
    ├── RasterObservation (env/representation/raster_obs.py) - RL观测
    ├── ActionSpace (env/action_space.py)
    ├── Transition (env/transition.py)
    ├── STGNNGraphBuilder (simulator/stgnn_graph_builder.py) - STGNN图构建
    ├── FeatureExtractor (simulator/features.py)
    ├── STGNNWrapper (simulator/stgnn_wrapper.py)
    ├── VitalityMetrics (objective/vitality_metrics.py)
    ├── RewardCalculator (objective/reward.py)
    └── TerminationChecker (objective/termination.py)
```

## 关键设计决策

### 1. RL观测表示
- **Raster**: 固定维度栅格观测，由CNN编码器处理，然后输入MLP策略和价值函数
- **注意**: GNN仅用于STGNN人流预测，不用于RL策略

### 2. 动作编码
- 统一使用 `Action` 数据类
- 支持 multi-discrete 编码/解码
- 通过 `action_mask` 保证合法性

### 3. 奖励分解
- 奖励 = 活力变化 - 成本 - 违规惩罚
- 所有分项可追踪，便于可解释性分析

### 4. 终止原因记录
- 明确的终止原因枚举
- 便于分析策略失败模式

### 5. 模块解耦
- STGNN作为独立simulator，可替换
- Policy encoder与head解耦，便于实验不同架构

## 扩展点

1. **新的动作类型**: 在 `ActionType` 枚举中添加，在 `Transition` 中实现
2. **新的活力指标**: 在 `VitalityMetrics` 中添加计算方法
3. **新的RL算法**: 实现新的Trainer类，复用Policy/Value接口
4. **新的表示方法**: 实现新的Observation类，实现 `encode()` 方法（当前仅支持Raster）

# Playground - 强化学习相关代码

这个文件夹包含了所有与强化学习（Reinforcement Learning）相关的代码和配置。

## 目录结构

```
playground/
├── rl/                    # 强化学习算法实现
│   ├── train_ppo.py      # PPO训练器
│   ├── policy_cnn.py      # CNN策略网络
│   ├── value_head.py      # 价值函数头
│   └── callbacks/         # 回调函数
├── env/                   # RL环境相关
│   ├── action_space.py   # 动作空间定义
│   ├── transition.py     # 状态转移
│   └── raster_obs.py     # 栅格化观测（RL观测）
├── objective/             # RL目标函数
│   ├── reward.py         # 奖励函数
│   └── termination.py    # 终止条件
├── configs/               # RL配置文件
│   ├── train.yaml        # 训练配置
│   └── reward.yaml       # 奖励配置
├── tests/                 # RL相关测试
│   ├── test_action_mask.py
│   ├── test_reward_termination.py
│   └── test_transition.py
├── scripts/               # RL相关脚本
│   ├── view_training_logs.py
│   └── render_episode_observations.py
├── logging/               # RL日志记录
│   └── episode_logger.py
└── main.py                # RL训练主入口
```

## 说明

这些代码已经从主项目中移出，因为项目决定放弃强化学习路线，专注于STGNN（时空图神经网络）的开发和优化。

如果需要恢复RL功能，可以将这些文件移回原位置，并修复导入路径。

## 注意事项

- 这些文件中的导入路径可能需要调整，因为它们原本位于项目根目录
- 如果需要在playground中运行这些代码，需要修改导入路径，例如：
  - `from env.world_state import WorldState` → `from ...env.world_state import WorldState`
  - 或者将playground添加到Python路径中

"""PPO训练脚本 - 基于stable-baselines3"""
from typing import Dict, Any, Optional
import os
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


class VitalStreetEnv(gym.Env):
    """VitalStreet GYM环境包装器（精简版）"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 观测空间：栅格图像 (C, H, W)
        obs_config = config.get('env', {}).get('representation', {}).get('raster', {})
        resolution = obs_config.get('resolution', [256, 256])
        n_channels = len(obs_config.get('channels', []))
        
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(n_channels, resolution[0], resolution[1]),
            dtype=np.float32
        )
        
        # 动作空间：multi-discrete（简化版，实际需要根据ActionSpace定义）
        action_config = config.get('env', {}).get('action_space', {})
        # 假设动作空间：5种动作类型，最大1000个节点，最大2000条边
        n_action_types = len(action_config.get('action_types', []))
        self.action_space = spaces.MultiDiscrete([
            n_action_types,  # 动作类型
            action_config.get('max_nodes', 1000),  # 目标节点
            action_config.get('max_edges', 2000),  # 目标边
        ])
        
        # 初始化环境组件（假设这些模块存在）
        # 注意：测试阶段可以先用mock数据
        self._init_components()
        
    def _init_components(self):
        """初始化环境组件"""
        # TODO: 实际实现中需要导入并初始化：
        # - WorldState
        # - RasterObservation
        # - ActionSpace
        # - Transition
        # - RewardCalculator
        # - TerminationChecker
        # - STGNNWrapper等
        self.state = None
        self.step_count = 0
        self.max_steps = self.config.get('env', {}).get('termination', {}).get('max_steps', 100)
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """重置环境"""
        super().reset(seed=seed)
        
        # TODO: 初始化WorldState
        # self.state = WorldState.initialize(...)
        
        # TODO: 获取初始观测
        # obs = RasterObservation.from_state(self.state)
        
        # 测试阶段：返回随机观测
        obs = self.observation_space.sample()
        
        self.step_count = 0
        info = {}
        return obs, info
    
    def step(self, action):
        """执行一步"""
        # TODO: 实际实现：
        # 1. 解码动作: action_obj = ActionSpace.decode(action)
        # 2. 执行转移: next_state = Transition.step(self.state, action_obj)
        # 3. 计算奖励: reward, reward_terms = RewardCalculator.compute(...)
        # 4. 检查终止: done, reason = TerminationChecker.check(...)
        # 5. 获取观测: obs = RasterObservation.from_state(next_state)
        
        # 测试阶段：简单模拟
        self.step_count += 1
        
        # 随机奖励（测试用）
        reward = np.random.randn() * 0.1
        
        # 检查终止
        done = self.step_count >= self.max_steps
        truncated = False
        
        # 下一个观测
        obs = self.observation_space.sample()
        
        info = {
            'step': self.step_count,
            'episode': {'r': reward}
        }
        
        return obs, reward, done, truncated, info


class PPOTrainer:
    """PPO训练器（基于stable-baselines3）"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化训练配置"""
        self.config = config
        self.ppo_config = config.get('ppo', {})
        
        # 创建环境
        def make_env():
            env = VitalStreetEnv(config)
            env = Monitor(env)  # 监控环境
            return env
        self.env = DummyVecEnv([make_env])
        
        # 创建PPO模型
        self.model = PPO(
            policy='CnnPolicy',  # CNN策略，适合图像观测
            env=self.env,
            learning_rate=self.ppo_config.get('learning_rate', 3e-4),
            n_steps=self.ppo_config.get('n_steps', 2048),  # 每次收集的步数
            batch_size=self.ppo_config.get('batch_size', 64),
            n_epochs=self.ppo_config.get('n_epochs', 4),
            gamma=self.ppo_config.get('gamma', 0.99),
            gae_lambda=self.ppo_config.get('gae_lambda', 0.95),
            clip_range=self.ppo_config.get('clip_epsilon', 0.2),
            ent_coef=self.ppo_config.get('entropy_coef', 0.01),
            vf_coef=self.ppo_config.get('value_coef', 0.5),
            max_grad_norm=self.ppo_config.get('max_grad_norm', 0.5),
            verbose=1,
            device=config.get('device', 'auto'),
        )
        
        # 检查点目录
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def train(self, total_timesteps: Optional[int] = None, log_interval: int = 10):
        """训练主循环"""
        if total_timesteps is None:
            # 从配置中获取，如果没有则使用默认值
            n_episodes = self.config.get('n_episodes', 1000)
            # 估算总步数（假设每个episode平均100步）
            total_timesteps = n_episodes * 100
        
        # 设置回调函数
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.get('save_interval', 100) * 1000,  # 每N个episode保存
            save_path=self.checkpoint_dir,
            name_prefix='ppo_model'
        )
        
        # 训练
        print(f"开始训练，总步数: {total_timesteps}")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            log_interval=log_interval,
            progress_bar=True
        )
        
        # 保存最终模型
        final_model_path = os.path.join(self.checkpoint_dir, 'ppo_final')
        self.model.save(final_model_path)
        print(f"训练完成！模型已保存至: {final_model_path}")
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        self.model.save(path)
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        self.model = PPO.load(path, env=self.env)
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

# 导入动作空间定义
from env.action_space import ActionSpace, ActionType
# 导入环境组件
from env.world_state import WorldState
from env.representation.raster_obs import RasterObservation
from env.transition import Transition
# 导入目标函数组件
from objective.reward import RewardCalculator
from objective.termination import TerminationChecker


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
        
        # 动作空间：使用真实的ActionSpace定义
        action_config = config.get('env', {}).get('action_space', {})
        
        # 创建ActionSpace实例以获取正确的动作维度
        action_space_manager = ActionSpace(action_config)
        action_dims = action_space_manager.get_action_dim()
        
        # 确保所有维度都为正数
        for i, dim in enumerate(action_dims):
            if dim <= 0:
                raise ValueError(f"动作空间维度 {i} 必须为正数，当前: {dim}")
        
        # 创建MultiDiscrete动作空间
        self.action_space = spaces.MultiDiscrete(action_dims)
        
        # 保存ActionSpace管理器供后续使用
        self.action_space_manager = action_space_manager
        
        # 初始化环境组件（假设这些模块存在）
        # 注意：测试阶段可以先用mock数据
        self._init_components()
        
    def _init_components(self):
        """初始化环境组件"""
        env_config = self.config.get('env', {})
        
        # 初始化RasterObservation编码器
        raster_config = env_config.get('representation', {}).get('raster', {})
        self.raster_obs = None  # 将在reset时根据初始状态创建
        
        # 获取初始状态路径（从配置或默认值）
        self.initial_geojson_path = env_config.get('initial_state', {}).get('geojson_path', 'data/0123_2.geojson')
        self.budget = env_config.get('constraints', {}).get('max_budget', 1000000.0)
        self.constraints = env_config.get('constraints', {})
        
        # 初始化Transition（状态转移引擎）
        transition_config = env_config.get('transition', {})
        self.transition = Transition(transition_config)
        
        # 初始化RewardCalculator（奖励计算器）
        self.reward_calculator = RewardCalculator(self.config.get('reward', {}))
        
        # 初始化TerminationChecker（终止条件检查器）
        termination_config = env_config.get('termination', {})
        self.termination_checker = TerminationChecker(termination_config)
        
        # 状态和步数
        self.state = None
        self.step_count = 0
        self.max_steps = env_config.get('termination', {}).get('max_steps', 100)
        
        # 历史记录（仅用于终止检查，精简：只存储reward）
        self.history = []  # List[float]: 最近N步的reward
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 从GeoJSON加载初始状态
        try:
            self.state = WorldState.from_geojson(
                geojson_path=self.initial_geojson_path,
                budget=self.budget,
                constraints=self.constraints,
                episode_id=f"episode_{self.step_count}"
            )
        except FileNotFoundError:
            # 如果文件不存在，使用默认状态（空状态）
            from env.geo.space_unit import SpaceUnitCollection
            from env.geo.street_network import StreetNetworkCollection
            from env.geo.business_type import BusinessTypeCollection
            
            self.state = WorldState(
                space_units=SpaceUnitCollection(),
                street_network=StreetNetworkCollection(),
                business_types=BusinessTypeCollection(),
                graph=None,
                budget=self.budget,
                constraints=self.constraints,
                step_idx=0,
                episode_id=f"episode_{self.step_count}"
            )
        
        # 创建或重置RasterObservation编码器
        raster_config = self.config.get('env', {}).get('representation', {}).get('raster', {})
        resolution = raster_config.get('resolution', [256, 256])
        channels = raster_config.get('channels', ['walkable_mask', 'predicted_flow', 'landuse_id'])
        
        if resolution == 'auto' or self.raster_obs is None:
            # 使用自动分辨率（根据初始状态计算）
            self.raster_obs = RasterObservation.create_with_auto_resolution(
                state=self.state,
                channels=channels,
                target_pixels=256,
                min_resolution=128,
                max_resolution=1024,
                padding=0.05
            )
        else:
            # 使用固定分辨率
            if self.raster_obs.resolution != resolution:
                # 如果分辨率改变，重新创建编码器
                self.raster_obs = RasterObservation({
                    'resolution': resolution,
                    'channels': channels
                })
        
        # 重置RewardCalculator（保存初始状态的活力值）
        self.reward_calculator.reset(self.state)
        
        # 重置历史记录（用于终止检查）
        self.history = []
        
        # 编码初始观测
        obs = self.raster_obs.encode(self.state)
        
        self.step_count = 0
        info = {
            'episode_id': self.state.episode_id,
            'step': self.step_count
        }
        return obs, info
    
    def step(self, action):
        """执行一步"""
        action = np.asarray(action).flatten()
        action_obj = self.action_space_manager.decode(action, self.state)
        
        # 2. 执行转移
        prev_state = self.state
        next_state, transition_info = self.transition.step(self.state, action_obj)
        self.state = next_state
        
        # 3. 计算奖励
        reward, reward_terms = self.reward_calculator.compute(
            state=prev_state,
            action=action_obj,
            next_state=next_state
        )
        
        # 4. 检查终止（history仅存储reward用于停滞检查）
        self.history.append(reward)
        # 保持history长度不超过stagnation_threshold
        stagnation_threshold = self.config.get('env', {}).get('termination', {}).get('stagnation_threshold', 10)
        if len(self.history) > stagnation_threshold:
            self.history.pop(0)
        
        done, reason = self.termination_checker.check(next_state, self.history)
        truncated = False
        
        # 5. 编码观测
        obs = self.raster_obs.encode(next_state)
        
        info = {
            'step': next_state.step_idx,
            'episode': {'r': reward},
            'reward_terms': reward_terms,
            'termination_reason': reason,
            'transition_info': transition_info
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
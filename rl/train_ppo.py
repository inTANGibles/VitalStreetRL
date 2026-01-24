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
        
        # 预先检测单元数量（从geojson文件）
        self._detect_unit_count(config)
        
        # 更新action_config：使用检测到的单元数量和固定的business_types=8
        if not hasattr(self, 'detected_max_total_units'):
            self.detected_max_total_units = action_config.get('max_total_units', 120)
        action_config['max_total_units'] = self.detected_max_total_units
        action_config['max_business_types'] = 8  # 固定为8（对应BusinessCategory的8个枚举值）
        
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
        
        # 打印环境初始化信息
        print(f"[环境初始化] 观测空间: {self.observation_space.shape}, 动作空间维度: {action_dims}")
        print(f"[环境初始化] 检测到的单元数量: {self.detected_max_total_units}, max_business_types: 8")
        
    def _detect_unit_count(self, config: Dict[str, Any]):
        """从geojson文件检测单元数量"""
        env_config = config.get('env', {})
        geojson_path = env_config.get('initial_state', {}).get('geojson_path', 'data/0123_2.geojson')
        
        try:
            # 尝试加载geojson文件并检测单元数量
            import geopandas as gpd
            
            geojson_path = Path(geojson_path)
            if geojson_path.exists():
                # 读取GeoJSON文件
                gdf = gpd.read_file(geojson_path)
                
                # 转换为SpaceUnitCollection以获取实际单元数量
                import sys
                project_root = Path(__file__).resolve().parent.parent
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                
                from scripts.geojson_to_raster import geojson_to_spaceunit_collection
                collection = geojson_to_spaceunit_collection(gdf)
                all_units = collection.get_all_space_units()
                
                # 直接使用检测到的单元数量
                self.detected_max_total_units = len(all_units)
                
                print(f"[单元检测] 从 {geojson_path} 检测到 {self.detected_max_total_units} 个单元，设置 max_total_units = {self.detected_max_total_units}")
            else:
                # 文件不存在，使用默认值
                self.detected_max_total_units = env_config.get('action_space', {}).get('max_total_units', 120)
                print(f"[单元检测] GeoJSON文件不存在: {geojson_path}，使用默认值 max_total_units = {self.detected_max_total_units}")
        except Exception as e:
            # 检测失败，使用默认值
            self.detected_max_total_units = env_config.get('action_space', {}).get('max_total_units', 120)
            print(f"[单元检测] 检测失败: {e}，使用默认值 max_total_units = {self.detected_max_total_units}")
    
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
        
        # 用于控制打印频率
        self.print_interval = 3  # 每N步打印一次（改为3步）
        self.episode_reward_sum = 0.0  # 当前episode的累计奖励
        self.episode_count = 0  # Episode计数器（用于生成正确的episode_id）
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """重置环境（带异常处理）"""
        try:
            super().reset(seed=seed)
            
            # 递增episode计数器（在创建WorldState之前）
            self.episode_count += 1
            current_episode_id = f"episode_{self.episode_count}"
            
            # 从GeoJSON加载初始状态
            try:
                self.state = WorldState.from_geojson(
                    geojson_path=self.initial_geojson_path,
                    budget=self.budget,
                    constraints=self.constraints,
                    episode_id=current_episode_id
                )
            except (FileNotFoundError, Exception) as e:
                # 如果文件不存在或其他错误，使用默认状态（空状态）
                print(f"[警告] 无法加载GeoJSON文件 {self.initial_geojson_path}: {e}")
                from env.geo.space_unit import SpaceUnitCollection
                from env.geo.business_type import BusinessTypeCollection
                
                self.state = WorldState(
                    space_units=SpaceUnitCollection(),
                    business_types=BusinessTypeCollection(),
                    graph=None,
                    budget=self.budget,
                    constraints=self.constraints,
                    step_idx=0,
                    episode_id=current_episode_id
                )
            
            # 创建或重置RasterObservation编码器
            raster_config = self.config.get('env', {}).get('representation', {}).get('raster', {})
            resolution = raster_config.get('resolution', [256, 256])
            channels = raster_config.get('channels', ['walkable_mask', 'predicted_flow', 'landuse_id'])
            
            # 获取目标分辨率（用于填充到固定尺寸，从observation_space获取）
            # observation_space.shape = (C, H, W)
            target_resolution = list(self.observation_space.shape[1:])  # [H, W]
            
            try:
                if resolution == 'auto' or self.raster_obs is None:
                    # 使用自动分辨率（根据初始状态计算）
                    self.raster_obs = RasterObservation.create_with_auto_resolution(
                        state=self.state,
                        channels=channels,
                        target_pixels=256,
                        min_resolution=128,
                        max_resolution=1024,
                        padding=0.05,
                        target_resolution=target_resolution  # 填充到固定尺寸
                    )
                else:
                    # 使用固定分辨率
                    if self.raster_obs is None or (hasattr(self.raster_obs, 'resolution') and self.raster_obs.resolution != resolution):
                        # 如果分辨率改变，重新创建编码器
                        self.raster_obs = RasterObservation({
                            'resolution': resolution,
                            'channels': channels,
                            'target_resolution': target_resolution
                        })
            except Exception as e:
                print(f"[错误] RasterObservation初始化失败: {e}")
                # 使用默认配置
                self.raster_obs = RasterObservation({
                    'resolution': resolution,
                    'channels': channels,
                    'target_resolution': target_resolution
                })
            
            # 重置RewardCalculator（保存初始状态的活力值）
            try:
                self.reward_calculator.reset(self.state)
            except Exception as e:
                print(f"[警告] RewardCalculator重置失败: {e}")
            
            # 重置历史记录（用于终止检查）
            self.history = []
            self.episode_reward_sum = 0.0
            
            # 编码初始观测（自动填充到固定尺寸）
            try:
                obs = self.raster_obs.encode(self.state)
                obs = np.asarray(obs, dtype=np.float32)
                if obs.shape != self.observation_space.shape:
                    print(f"[警告] 初始观测形状不匹配: 期望 {self.observation_space.shape}, 得到 {obs.shape}")
                    if obs.size == np.prod(self.observation_space.shape):
                        obs = obs.reshape(self.observation_space.shape)
                    else:
                        obs = self.observation_space.sample()
            except Exception as e:
                print(f"[错误] 初始观测编码失败: {e}")
                obs = self.observation_space.sample()
            
            self.step_count = 0
            
            # 打印episode开始信息
            try:
                space_units_count = len(self.state.space_units.get_all_space_units())
                print(f"[Episode {self.state.episode_id}] 开始 | 空间单元数: {space_units_count} | 预算: {self.budget:.0f}")
            except Exception:
                print(f"[Episode {self.state.episode_id}] 开始 | 预算: {self.budget:.0f}")
            
            info = {
                'episode_id': self.state.episode_id if hasattr(self.state, 'episode_id') else 'unknown',
                'step': self.step_count
            }
            return obs, info
            
        except Exception as e:
            # 如果reset完全失败，返回安全的默认值
            import traceback
            print(f"[严重错误] reset方法异常: {e}")
            print(traceback.format_exc())
            obs = self.observation_space.sample()
            return obs, {'error': str(e)}
    
    def step(self, action):
        """执行一步（带异常处理防止内存访问错误）"""
        try:
            # 1. 解码动作
            action = np.asarray(action).flatten()
            try:
                action_obj = self.action_space_manager.decode(action, self.state)
            except Exception as e:
                print(f"[错误] 动作解码失败: {e}, action={action}")
                # 使用NO_OP作为fallback
                from env.action_space import Action, ActionType
                action_obj = Action(type=ActionType.NO_OP, target_id=None, params={})
            
            # 2. 执行转移
            prev_state = self.state
            try:
                next_state, transition_info = self.transition.step(self.state, action_obj)
                self.state = next_state
            except Exception as e:
                print(f"[错误] 状态转移失败: {e}")
                # 如果转移失败，保持当前状态，但标记为done
                next_state = self.state
                transition_info = {'error': str(e)}
                done = True
                truncated = True
                obs = self.raster_obs.encode(self.state) if self.raster_obs else self.observation_space.sample()
                return obs, -100.0, done, truncated, {'error': 'transition_failed'}
            
            # 3. 计算奖励
            try:
                reward, reward_terms = self.reward_calculator.compute(
                    state=prev_state,
                    action=action_obj,
                    next_state=next_state
                )
                # 确保reward是标量
                reward = float(reward)
                if not isinstance(reward_terms, dict):
                    reward_terms = {'total': reward}
            except Exception as e:
                print(f"[错误] 奖励计算失败: {e}")
                reward = 0.0
                reward_terms = {'error': str(e), 'total': 0.0}
            
            # 累计episode奖励
            self.episode_reward_sum += reward
            
            # 4. 检查终止（history仅存储reward用于停滞检查）
            self.history.append(reward)
            # 保持history长度不超过stagnation_threshold
            stagnation_threshold = self.config.get('env', {}).get('termination', {}).get('stagnation_threshold', 10)
            if len(self.history) > stagnation_threshold:
                self.history.pop(0)
            
            try:
                done, reason = self.termination_checker.check(next_state, self.history)
            except Exception as e:
                print(f"[错误] 终止检查失败: {e}")
                done = self.step_count >= self.max_steps
                reason = 'error'
            
            truncated = False
            
            # 定期打印步骤信息
            step_idx = next_state.step_idx if hasattr(next_state, 'step_idx') else self.step_count
            self.step_count += 1
            
            if step_idx % self.print_interval == 0 or done:
                action_type_str = action_obj.type.name if hasattr(action_obj.type, 'name') else str(action_obj.type)
                reward_vitality = reward_terms.get('vitality', 0.0) if isinstance(reward_terms, dict) else 0.0
                reward_cost = reward_terms.get('cost', 0.0) if isinstance(reward_terms, dict) else 0.0
                reward_violation = reward_terms.get('violation', 0.0) if isinstance(reward_terms, dict) else 0.0
                print(f"  Step {step_idx:3d} | 动作: {action_type_str:25s} | "
                      f"奖励: {reward:7.3f} (活力:{reward_vitality:6.3f}, 成本:{reward_cost:6.3f}, 违规:{reward_violation:6.3f}) | "
                      f"累计: {self.episode_reward_sum:7.3f}")
            
            # Episode结束时打印总结
            if done:
                print(f"[Episode {self.state.episode_id}] 结束 | 总步数: {step_idx} | "
                      f"累计奖励: {self.episode_reward_sum:.3f} | 终止原因: {reason}")
            
            # 5. 编码观测（自动填充到固定尺寸）
            try:
                obs = self.raster_obs.encode(next_state)
                # 确保obs是正确的形状和类型
                obs = np.asarray(obs, dtype=np.float32)
                if obs.shape != self.observation_space.shape:
                    print(f"[警告] 观测形状不匹配: 期望 {self.observation_space.shape}, 得到 {obs.shape}")
                    # 尝试reshape或使用默认观测
                    if obs.size == np.prod(self.observation_space.shape):
                        obs = obs.reshape(self.observation_space.shape)
                    else:
                        obs = self.observation_space.sample()
            except Exception as e:
                print(f"[错误] 观测编码失败: {e}")
                # 使用默认观测
                obs = self.observation_space.sample()
            
            info = {
                'step': step_idx,
                'episode': {'r': reward},
                'reward_terms': reward_terms,
                'termination_reason': reason,
                'transition_info': transition_info
            }
            
            return obs, reward, done, truncated, info
            
        except Exception as e:
            # 捕获所有未预期的错误
            import traceback
            print(f"[严重错误] step方法异常: {e}")
            print(traceback.format_exc())
            # 返回安全的默认值
            obs = self.observation_space.sample()
            return obs, -100.0, True, True, {'error': str(e), 'traceback': traceback.format_exc()}


class PPOTrainer:
    """PPO训练器（基于stable-baselines3）"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化训练配置"""
        self.config = config
        self.ppo_config = config.get('ppo', {})
        
        print("=" * 80)
        print("[PPO训练器] 初始化中...")
        
        # 创建环境
        # 设置日志目录（Monitor会保存monitor.csv到这里）
        log_dir = config.get('log_dir', './logs')
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        def make_env():
            env = VitalStreetEnv(config)
            env = Monitor(env, filename=os.path.join(log_dir, 'monitor.csv'))  # 监控环境，指定日志文件
            return env
        self.env = DummyVecEnv([make_env])
        print(f"[PPO训练器] 环境创建完成")
        
        # 创建PPO模型（确保数值参数为正确的类型）
        learning_rate = float(self.ppo_config.get('learning_rate', 3e-4))
        n_steps = int(self.ppo_config.get('n_steps', 2048))
        batch_size = int(self.ppo_config.get('batch_size', 64))
        n_epochs = int(self.ppo_config.get('n_epochs', 4))
        
        print(f"[PPO训练器] 模型配置:")
        print(f"  - 策略: CnnPolicy")
        print(f"  - 学习率: {learning_rate}")
        print(f"  - n_steps: {n_steps}")
        print(f"  - batch_size: {batch_size}")
        print(f"  - n_epochs: {n_epochs}")
        print(f"  - 设备: {config.get('device', 'auto')}")
        
        self.model = PPO(
            policy='CnnPolicy',  # CNN策略，适合图像观测
            env=self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,  # 每次收集的步数
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=float(self.ppo_config.get('gamma', 0.99)),
            gae_lambda=float(self.ppo_config.get('gae_lambda', 0.95)),
            clip_range=float(self.ppo_config.get('clip_epsilon', 0.2)),
            ent_coef=float(self.ppo_config.get('entropy_coef', 0.01)),
            vf_coef=float(self.ppo_config.get('value_coef', 0.5)),
            max_grad_norm=float(self.ppo_config.get('max_grad_norm', 0.5)),
            policy_kwargs={'normalize_images': False},  # 观测已经是归一化的[0,1]图像
            verbose=1,
            device=config.get('device', 'auto'),
        )
        
        # 检查点目录
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        print(f"[PPO训练器] 检查点目录: {self.checkpoint_dir}")
        
        # 日志目录
        self.log_dir = config.get('log_dir', './logs')
        print(f"[PPO训练器] 日志目录: {self.log_dir} (monitor.csv将保存在这里)")
        print("=" * 80)
    
    def train(self, total_timesteps: Optional[int] = None, log_interval: int = 10):
        """训练主循环"""
        if total_timesteps is None:
            # 从配置中获取，如果没有则使用默认值
            n_episodes = self.config.get('n_episodes', 1000)
            # 估算总步数（假设每个episode平均100步）
            total_timesteps = n_episodes * 100
        
        # 设置回调函数
        save_interval = self.config.get('save_interval', 100)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_interval * 1000,  # 每N个episode保存
            save_path=self.checkpoint_dir,
            name_prefix='ppo_model'
        )
        
        # 训练
        print("\n" + "=" * 80)
        print(f"[训练开始] 总步数: {total_timesteps} | "
              f"预计episode数: ~{total_timesteps // 20} | "
              f"检查点保存间隔: 每 {save_interval}K 步")
        print("=" * 80 + "\n")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            log_interval=log_interval,
            progress_bar=self.config.get('show_progress_bar', True)  # 可从配置控制
        )
        
        # 保存最终模型
        final_model_path = os.path.join(self.checkpoint_dir, 'ppo_final')
        self.model.save(final_model_path)
        print("\n" + "=" * 80)
        print(f"[训练完成] 模型已保存至: {final_model_path}")
        print("=" * 80)
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        self.model.save(path)
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        self.model = PPO.load(path, env=self.env)
"""PPO训练脚本 - 基于stable-baselines3"""
from typing import Dict, Any, Optional
import os
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# 导入render日志回调
try:
    from rl.callbacks.render_logger import RenderLoggerCallback
except ImportError:
    RenderLoggerCallback = None
    print("[警告] RenderLoggerCallback未找到，render日志功能将不可用")

# 导入动作空间定义
from env.action_space import ActionSpace, ActionType
# 导入环境组件
from env.world_state import WorldState
from env.representation.raster_obs import RasterObservation
from env.transition import Transition
# 导入目标函数组件
from objective.reward import RewardCalculator
from objective.termination import TerminationChecker
# 导入flow计算函数
from scripts.compute_flow_from_complexity import compute_flow_from_complexity


class EpisodeCheckpointCallback(BaseCallback):
    """按Episode保存检查点的回调函数"""
    
    def __init__(self, save_path: str, save_every_n_episodes: int = 1, verbose: int = 0):
        """
        Args:
            save_path: 保存路径
            save_every_n_episodes: 每N个episode保存一次（1表示每个episode都保存）
            verbose: 详细程度
        """
        super().__init__(verbose)
        self.save_path = save_path
        self.save_every_n_episodes = save_every_n_episodes
        self.episode_count = 0
        self.last_done = False  # 跟踪上一个step的done状态，避免重复计数
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _on_step(self) -> bool:
        # 获取当前step的done状态
        dones = self.locals.get('dones', [False])
        truncated = self.locals.get('truncated', [False])
        current_done = dones[0] if dones else False
        current_truncated = truncated[0] if truncated else False
        
        # 检测episode结束：从False变为True（避免在同一个episode结束的多个step中重复计数）
        if (current_done or current_truncated) and not self.last_done:
            self.episode_count += 1
            
            # 每N个episode保存一次
            if self.episode_count % self.save_every_n_episodes == 0:
                episode_num = self.episode_count
                checkpoint_path = os.path.join(self.save_path, f'ppo_model_episode_{episode_num:04d}')
                self.model.save(checkpoint_path)
                if self.verbose > 0:
                    print(f"[检查点] Episode {episode_num} 完成，模型已保存: {checkpoint_path}.zip")
        
        # 更新last_done状态
        self.last_done = current_done or current_truncated
        
        return True


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
            
            # 计算初始状态的flow_prediction（确保predicted_flow通道有值）
            try:
                flow_config = self.config.get('env', {}).get('transition', {}).get('flow', {})
                compute_flow_from_complexity(
                    collection=self.state.space_units,
                    buffer_distance=flow_config.get('buffer_distance', 10.0),
                    diversity_weight=flow_config.get('diversity_weight', 0.5),
                    weighted_sum_weight=flow_config.get('weighted_sum_weight', 0.5)
                )
                print(f"[环境重置] 已计算初始状态的flow_prediction")
            except Exception as e:
                print(f"[警告] 计算初始flow_prediction失败: {e}")
            
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
        step_idx = self.step_count
        print(f"[DEBUG Step {step_idx}] ========== 开始执行step ==========")
        
        try:
            # ========== 1. 解码动作 ==========
            print(f"[DEBUG Step {step_idx}] [1/6] 开始 decode_action()")
            action = np.asarray(action).flatten()
            try:
                action_obj = self.action_space_manager.decode(action, self.state)
                print(f"[DEBUG Step {step_idx}] [1/6] ✓ decode_action() 完成: {action_obj.type.name if hasattr(action_obj.type, 'name') else action_obj.type}")
            except Exception as e:
                print(f"[DEBUG Step {step_idx}] [1/6] ✗ decode_action() 失败: {e}, action={action}")
                # 使用NO_OP作为fallback
                from env.action_space import Action, ActionType
                action_obj = Action(type=ActionType.NO_OP, target_id=None, params={})
                print(f"[DEBUG Step {step_idx}] [1/6] 使用NO_OP作为fallback")
            
            # ========== 2. 执行转移（包含apply_action_to_state, rebuild_graph, compute_flow） ==========
            print(f"[DEBUG Step {step_idx}] [2/6] 开始 apply_action_to_state()")
            prev_state = self.state
            try:
                print(f"[DEBUG Step {step_idx}] [2/6] 调用 transition.step()...")
                next_state, transition_info = self.transition.step(self.state, action_obj)
                print(f"[DEBUG Step {step_idx}] [2/6] ✓ transition.step() 完成")
                self.state = next_state
                print(f"[DEBUG Step {step_idx}] [2/6] ✓ apply_action_to_state() 完成")
            except Exception as e:
                print(f"[DEBUG Step {step_idx}] [2/6] ✗ apply_action_to_state() 失败: {e}")
                import traceback
                print(f"[DEBUG Step {step_idx}] [2/6] 堆栈跟踪:\n{traceback.format_exc()}")
                # 如果转移失败，保持当前状态，但标记为done
                next_state = self.state
                transition_info = {'error': str(e)}
                done = True
                truncated = True
                obs = self.raster_obs.encode(self.state) if self.raster_obs else self.observation_space.sample()
                return obs, -100.0, done, truncated, {'error': 'transition_failed'}
            
            # ========== 3. 计算奖励 ==========
            print(f"[DEBUG Step {step_idx}] [3/6] 开始 compute_reward()")
            try:
                print(f"[DEBUG Step {step_idx}] [3/6] 调用 reward_calculator.compute()...")
                reward, reward_terms = self.reward_calculator.compute(
                    state=prev_state,
                    action=action_obj,
                    next_state=next_state
                )
                print(f"[DEBUG Step {step_idx}] [3/6] ✓ reward_calculator.compute() 完成")
                # 确保reward是标量
                reward = float(reward)
                if not isinstance(reward_terms, dict):
                    reward_terms = {'total': reward}
                print(f"[DEBUG Step {step_idx}] [3/6] ✓ compute_reward() 完成: reward={reward:.3f}")
            except Exception as e:
                print(f"[DEBUG Step {step_idx}] [3/6] ✗ compute_reward() 失败: {e}")
                import traceback
                print(f"[DEBUG Step {step_idx}] [3/6] 堆栈跟踪:\n{traceback.format_exc()}")
                reward = 0.0
                reward_terms = {'error': str(e), 'total': 0.0}
            
            # ========== 4. 检查终止条件 ==========
            print(f"[DEBUG Step {step_idx}] [4/6] 开始 check_termination()")
            # 累计episode奖励
            self.episode_reward_sum += reward
            
            # 检查终止（history仅存储reward用于停滞检查）
            self.history.append(reward)
            # 保持history长度不超过stagnation_threshold
            stagnation_threshold = self.config.get('env', {}).get('termination', {}).get('stagnation_threshold', 10)
            if len(self.history) > stagnation_threshold:
                self.history.pop(0)
            
            try:
                print(f"[DEBUG Step {step_idx}] [4/6] 调用 termination_checker.check()...")
                done, reason = self.termination_checker.check(next_state, self.history)
                print(f"[DEBUG Step {step_idx}] [4/6] ✓ check_termination() 完成: done={done}, reason={reason}")
            except Exception as e:
                print(f"[DEBUG Step {step_idx}] [4/6] ✗ check_termination() 失败: {e}")
                import traceback
                print(f"[DEBUG Step {step_idx}] [4/6] 堆栈跟踪:\n{traceback.format_exc()}")
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
            
            # ========== 5. 编码观测（get_observation） ==========
            print(f"[DEBUG Step {step_idx}] [5/6] 开始 get_observation()")
            try:
                print(f"[DEBUG Step {step_idx}] [5/6] 调用 raster_obs.encode()...")
                obs = self.raster_obs.encode(next_state)
                print(f"[DEBUG Step {step_idx}] [5/6] ✓ raster_obs.encode() 完成")
                # 确保obs是正确的形状和类型
                print(f"[DEBUG Step {step_idx}] [5/6] 转换obs为numpy数组...")
                obs = np.asarray(obs, dtype=np.float32)
                print(f"[DEBUG Step {step_idx}] [5/6] ✓ numpy转换完成, shape={obs.shape}")
                if obs.shape != self.observation_space.shape:
                    print(f"[DEBUG Step {step_idx}] [5/6] 警告: 观测形状不匹配: 期望 {self.observation_space.shape}, 得到 {obs.shape}")
                    # 尝试reshape或使用默认观测
                    if obs.size == np.prod(self.observation_space.shape):
                        obs = obs.reshape(self.observation_space.shape)
                        print(f"[DEBUG Step {step_idx}] [5/6] ✓ reshape完成")
                    else:
                        obs = self.observation_space.sample()
                        print(f"[DEBUG Step {step_idx}] [5/6] 使用默认观测")
                print(f"[DEBUG Step {step_idx}] [5/6] ✓ get_observation() 完成")
            except Exception as e:
                print(f"[DEBUG Step {step_idx}] [5/6] ✗ get_observation() 失败: {e}")
                import traceback
                print(f"[DEBUG Step {step_idx}] [5/6] 堆栈跟踪:\n{traceback.format_exc()}")
                # 使用默认观测
                obs = self.observation_space.sample()
            
            # ========== 6. 构建返回信息 ==========
            print(f"[DEBUG Step {step_idx}] [6/6] 开始 build_info()")
            info = {
                'step': step_idx,
                'episode': {'r': reward},
                'reward_terms': reward_terms,
                'termination_reason': reason,
                'transition_info': transition_info
            }
            print(f"[DEBUG Step {step_idx}] [6/6] ✓ build_info() 完成")
            print(f"[DEBUG Step {step_idx}] ========== step执行完成 ==========")
            
            return obs, reward, done, truncated, info
            
        except Exception as e:
            # 捕获所有未预期的错误
            import traceback
            print(f"[DEBUG Step {step_idx}] ========== step方法异常 ==========")
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
        
        # 检查点目录和日志目录（需要在创建模型前设置）
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        print(f"[PPO训练器] 检查点目录: {self.checkpoint_dir}")
        
        # 日志目录
        self.log_dir = config.get('log_dir', './logs')
        print(f"[PPO训练器] 日志目录: {self.log_dir} (monitor.csv将保存在这里)")
        
        # TensorBoard日志目录（需要在创建PPO模型前设置）
        self.tensorboard_log = config.get('tensorboard_log', None)
        if self.tensorboard_log:
            Path(self.tensorboard_log).mkdir(parents=True, exist_ok=True)
            print(f"[PPO训练器] TensorBoard日志目录: {self.tensorboard_log}")
        
        # 创建PPO模型（确保数值参数为正确的类型）
        learning_rate = float(self.ppo_config.get('learning_rate', 3e-4))
        n_steps = int(self.ppo_config.get('n_steps', 2048))
        batch_size = int(self.ppo_config.get('batch_size', 64))
        n_epochs = int(self.ppo_config.get('n_epochs', 4))
        
        # 获取网络结构配置
        policy_config = self.config.get('policy', {})
        policy_type = policy_config.get('type', 'cnn')
        
        # 构建policy_kwargs
        policy_kwargs = {'normalize_images': False}  # 观测已经是归一化的[0,1]图像
        
        if policy_type == 'cnn':
            cnn_config = policy_config.get('cnn', {})
            # 配置MLP网络结构（CNN编码器使用默认结构）
            mlp_hidden = cnn_config.get('mlp_hidden', [256, 128])
            # net_arch格式: dict(pi=[...], vf=[...]) 或 [...]
            # 如果只提供一个列表，会同时用于actor和critic
            policy_kwargs['net_arch'] = mlp_hidden
            print(f"[PPO训练器] 模型配置:")
            print(f"  - 策略: CnnPolicy (内置)")
            print(f"  - MLP隐藏层: {mlp_hidden}")
        else:
            print(f"[PPO训练器] 模型配置:")
            print(f"  - 策略: CnnPolicy (内置，使用默认网络结构)")
        
        print(f"  - 学习率: {learning_rate}")
        print(f"  - n_steps: {n_steps}")
        print(f"  - batch_size: {batch_size}")
        print(f"  - n_epochs: {n_epochs}")
        print(f"  - 设备: {config.get('device', 'auto')}")
        
        self.model = PPO(
            policy='CnnPolicy',  # CNN策略，适合图像观测（使用stable-baselines3内置实现）
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
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=config.get('device', 'auto'),
            tensorboard_log=self.tensorboard_log,  # TensorBoard日志
        )
        
        print("=" * 80)
    
    def train(self, total_timesteps: Optional[int] = None, log_interval: int = 10):
        """训练主循环"""
        if total_timesteps is None:
            # 从配置中获取，如果没有则使用默认值
            n_episodes = self.config.get('n_episodes', 1000)
            # 估算总步数（假设每个episode平均100步）
            total_timesteps = n_episodes * 100
        
        # 设置回调函数
        # 支持两种保存方式：
        # 1. 按episode保存（save_interval为负数或0，或save_by_episode=true）
        # 2. 按步数保存（save_interval为正数）
        save_interval = self.config.get('save_interval', 1)  # 默认每1个episode保存
        save_by_episode = self.config.get('save_by_episode', True)  # 默认按episode保存
        
        callbacks = []
        
        if save_by_episode or save_interval <= 0:
            # 按episode保存
            save_every_n_episodes = abs(save_interval) if save_interval != 0 else 1
            episode_callback = EpisodeCheckpointCallback(
                save_path=self.checkpoint_dir,
                save_every_n_episodes=save_every_n_episodes,
                verbose=1
            )
            callbacks.append(episode_callback)
            save_info = f"每 {save_every_n_episodes} 个episode"
        else:
            # 按步数保存
            checkpoint_callback = CheckpointCallback(
                save_freq=save_interval,
                save_path=self.checkpoint_dir,
                name_prefix='ppo_model'
            )
            callbacks.append(checkpoint_callback)
            save_info = f"每 {save_interval} 步"
        
        # 添加Render日志回调（如果启用）
        render_log_config = self.config.get('render_log', {})
        if render_log_config.get('enabled', False) and RenderLoggerCallback is not None:
            # 获取通道名称
            obs_config = self.config.get('env', {}).get('representation', {}).get('raster', {})
            channel_names = obs_config.get('channels', ['walkable_mask', 'predicted_flow', 'landuse_id'])
            
            # 创建TensorBoard日志目录
            tensorboard_log_dir = self.tensorboard_log or os.path.join(self.log_dir, 'tensorboard')
            Path(tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
            
            # 获取本地保存目录（如果配置了）
            save_dir = render_log_config.get('save_dir', None)
            if save_dir:
                # 创建保存目录
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                print(f"[训练] Render图像将保存到本地: {save_dir}")
            
            render_callback = RenderLoggerCallback(
                log_dir=tensorboard_log_dir,
                log_every_n_episodes=render_log_config.get('log_every_n_episodes', 1),
                log_every_n_steps=render_log_config.get('log_every_n_steps', None),
                channel_names=channel_names,
                save_dir=save_dir,
                verbose=1 if render_log_config.get('verbose', True) else 0  # 默认启用verbose以便看到保存信息
            )
            callbacks.append(render_callback)
            print(f"[训练] Render日志已启用，TensorBoard目录: {tensorboard_log_dir}")
        
        # 训练
        print("\n" + "=" * 80)
        print(f"[训练开始] 总步数: {total_timesteps} | "
              f"预计episode数: ~{total_timesteps // 20} | "
              f"检查点保存间隔: {save_info}")
        print("=" * 80 + "\n")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=log_interval,
            progress_bar=self.config.get('show_progress_bar', True)  # 可从配置控制
        )
        
        # 保存最终模型（stable-baselines3会自动添加.zip扩展名）
        final_model_path = os.path.join(self.checkpoint_dir, 'ppo_final')
        try:
            self.model.save(final_model_path)
            # 检查文件是否真的保存了
            final_model_zip = final_model_path + '.zip'
            if os.path.exists(final_model_zip):
                file_size = os.path.getsize(final_model_zip) / (1024 * 1024)  # MB
                print("\n" + "=" * 80)
                print(f"[训练完成] 模型已保存至: {final_model_zip}")
                print(f"文件大小: {file_size:.2f} MB")
                print("=" * 80)
            else:
                print("\n" + "=" * 80)
                print(f"[警告] 模型保存可能失败，文件不存在: {final_model_zip}")
                print("=" * 80)
        except Exception as e:
            print(f"\n[错误] 保存模型失败: {e}")
            import traceback
            traceback.print_exc()
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        self.model.save(path)
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        self.model = PPO.load(path, env=self.env)
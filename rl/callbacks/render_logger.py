"""Render图像日志记录回调（用于TensorBoard和本地保存）"""
from typing import Dict, Any, Optional
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免在服务器上出错
import matplotlib.pyplot as plt
from io import BytesIO
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import BaseCallback
from env.representation.visualization import visualize_raster_channels


class RenderLoggerCallback(BaseCallback):
    """记录render图像到TensorBoard的回调函数"""
    
    def __init__(
        self,
        log_dir: str,
        log_every_n_episodes: int = 1,
        log_every_n_steps: Optional[int] = None,
        channel_names: Optional[list] = None,
        save_dir: Optional[str] = None,
        verbose: int = 0
    ):
        """
        Args:
            log_dir: TensorBoard日志目录
            log_every_n_episodes: 每N个episode记录一次（默认每个episode都记录）
            log_every_n_steps: 每N步记录一次（可选，如果设置则按步数记录）
            channel_names: 通道名称列表（用于可视化）
            save_dir: 本地保存目录（可选，如果设置则保存图像到本地）
            verbose: 详细程度
        """
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_every_n_episodes = log_every_n_episodes
        self.log_every_n_steps = log_every_n_steps
        self.channel_names = channel_names or ['walkable_mask', 'predicted_flow', 'landuse_id']
        self.save_dir = save_dir
        
        # 创建TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # 创建本地保存目录
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        
        # 跟踪状态
        self.episode_count = 0
        self.step_count = 0
        self.last_done = False
        self.last_obs = None
        self.initial_obs_saved = False  # 跟踪是否已保存初始observation
        self.episode_obs_buffer = []  # 存储当前episode的observations
        
        # 总是打印关键信息（即使verbose=0）
        print(f"[RenderLogger] 初始化完成")
        print(f"  - TensorBoard日志目录: {log_dir}")
        if self.save_dir:
            print(f"  - 本地保存目录: {self.save_dir}")
            if not os.path.exists(self.save_dir):
                print(f"  - ⚠ 警告: 保存目录不存在，将创建: {self.save_dir}")
        else:
            print(f"  - ⚠ 本地保存未启用（save_dir=None）")
        print(f"  - Episode记录间隔: {log_every_n_episodes}")
        if log_every_n_steps:
            print(f"  - Step记录间隔: {log_every_n_steps}")
        if self.verbose > 0:
            print(f"  - 详细日志: 启用")
    
    def _on_step(self) -> bool:
        """在每个step调用"""
        # 获取当前step的信息
        dones = self.locals.get('dones', [False])
        truncated = self.locals.get('truncated', [False])
        infos = self.locals.get('infos', [{}])
        
        # 在stable-baselines3中，尝试多个键来获取observation
        # 优先级：new_obs > observations > obs
        new_obs = None
        for key in ['new_obs', 'observations', 'obs']:
            if key in self.locals:
                new_obs = self.locals.get(key)
                if new_obs is not None:
                    if self.verbose > 0 and self.step_count == 1:
                        print(f"[RenderLogger] 从 '{key}' 获取observation: {type(new_obs)}, shape={new_obs.shape if hasattr(new_obs, 'shape') else 'N/A'}")
                    break
        
        current_done = dones[0] if dones else False
        current_truncated = truncated[0] if truncated else False
        is_episode_end = (current_done or current_truncated) and not self.last_done
        is_episode_start = self.last_done and not (current_done or current_truncated)  # 从done变为not done表示新episode开始
        
        # 更新step计数
        self.step_count += 1
        
        # Episode开始时记录初始observation
        if is_episode_start and not self.initial_obs_saved:
            # 清空上一个episode的buffer
            self.episode_obs_buffer = []
            
            if new_obs is not None:
                if isinstance(new_obs, np.ndarray) and len(new_obs.shape) >= 3:
                    initial_obs = new_obs[0] if len(new_obs.shape) == 4 else new_obs
                    if initial_obs is not None and (self.episode_count + 1) % self.log_every_n_episodes == 0:
                        print(f"[RenderLogger] Episode {self.episode_count + 1} 开始，保存初始observation...")
                        self._log_observation(
                            initial_obs,
                            tag=f"render/episode_{self.episode_count + 1}_initial",
                            global_step=self.episode_count + 1
                        )
                        self.initial_obs_saved = True
                        # 添加到buffer
                        self.episode_obs_buffer.append(initial_obs.copy())
        
        # 获取当前observation（优先使用new_obs，如果没有则尝试其他键）
        current_obs = None
        if new_obs is not None:
            # new_obs 可能是 (n_envs, C, H, W) 格式
            if isinstance(new_obs, np.ndarray) and len(new_obs.shape) >= 3:
                current_obs = new_obs[0] if len(new_obs.shape) == 4 else new_obs
            elif isinstance(new_obs, (list, tuple)) and len(new_obs) > 0:
                current_obs = new_obs[0]
            else:
                current_obs = new_obs
        # 如果没有new_obs，尝试从其他键获取
        elif 'observations' in self.locals:
            observations = self.locals.get('observations', None)
            if observations is not None:
                if isinstance(observations, np.ndarray) and len(observations.shape) >= 3:
                    current_obs = observations[0] if len(observations.shape) == 4 else observations
                elif isinstance(observations, (list, tuple)) and len(observations) > 0:
                    current_obs = observations[0]
                else:
                    current_obs = observations
        
        # 更新last_obs并存储到buffer
        if current_obs is not None:
            self.last_obs = current_obs
            # 存储到buffer（用于episode结束时）
            if isinstance(current_obs, np.ndarray) and len(current_obs.shape) == 3:
                self.episode_obs_buffer.append(current_obs.copy())
                # 限制buffer大小（只保留最近的一些）
                if len(self.episode_obs_buffer) > 100:
                    self.episode_obs_buffer = self.episode_obs_buffer[-50:]
        
        # 按步数记录
        if self.log_every_n_steps and self.step_count % self.log_every_n_steps == 0:
            if self.last_obs is not None:
                self._log_observation(
                    self.last_obs,
                    tag=f"render/step_{self.step_count}",
                    global_step=self.step_count
                )
        
        # Episode结束时记录
        if is_episode_end:
            self.episode_count += 1
            self.initial_obs_saved = False  # 重置标志，准备下一个episode
            
            # 按episode记录
            if self.episode_count % self.log_every_n_episodes == 0:
                # 优先使用buffer中的最后一个observation（episode结束前的最后一个）
                obs_to_save = None
                
                # 1. 尝试从buffer获取（最可靠）
                if len(self.episode_obs_buffer) > 0:
                    obs_to_save = self.episode_obs_buffer[-1]
                    print(f"[RenderLogger] Episode {self.episode_count} 结束，从buffer获取observation (buffer大小={len(self.episode_obs_buffer)})")
                
                # 2. 如果buffer为空，使用last_obs
                if obs_to_save is None and self.last_obs is not None:
                    obs_to_save = self.last_obs
                    print(f"[RenderLogger] Episode {self.episode_count} 结束，使用last_obs")
                
                # 3. 如果还是为空，尝试从new_obs获取
                if obs_to_save is None and new_obs is not None:
                    if isinstance(new_obs, np.ndarray) and len(new_obs.shape) >= 3:
                        obs_to_save = new_obs[0] if len(new_obs.shape) == 4 else new_obs
                        print(f"[RenderLogger] Episode {self.episode_count} 结束，使用new_obs")
                
                if obs_to_save is not None:
                    # 记录episode结束时的observation
                    print(f"[RenderLogger] Episode {self.episode_count} 结束，准备保存observation (shape={obs_to_save.shape})...")
                    self._log_observation(
                        obs_to_save,
                        tag=f"render/episode_{self.episode_count}_final",
                        global_step=self.episode_count
                    )
                else:
                    print(f"[RenderLogger] ✗ 警告: Episode {self.episode_count} 结束时没有observation可保存")
                    print(f"[RenderLogger]   last_obs: {self.last_obs is not None}")
                    print(f"[RenderLogger]   new_obs: {new_obs is not None}")
                    print(f"[RenderLogger]   buffer大小: {len(self.episode_obs_buffer)}")
                    if self.verbose > 0:
                        print(f"[RenderLogger]   locals keys: {list(self.locals.keys())[:10]}")
                
                # 清空buffer，准备下一个episode
                self.episode_obs_buffer = []
                
                # 记录episode统计信息
                if infos and len(infos) > 0:
                    info = infos[0]
                    if isinstance(info, dict):
                        # 记录奖励信息
                        if 'episode' in info:
                            episode_info = info['episode']
                            if 'r' in episode_info:
                                self.writer.add_scalar(
                                    'episode/reward',
                                    episode_info['r'],
                                    self.episode_count
                                )
                            if 'l' in episode_info:
                                self.writer.add_scalar(
                                    'episode/length',
                                    episode_info['l'],
                                    self.episode_count
                                )
        
        # 更新last_done状态
        self.last_done = current_done or current_truncated
        
        return True
    
    def _on_rollout_end(self) -> None:
        """在rollout结束时调用（可以获取observations）"""
        # 尝试从rollout buffer获取observations
        if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer is not None:
            try:
                # rollout_buffer.observations 包含所有step的observations
                if hasattr(self.model.rollout_buffer, 'observations'):
                    obs_buffer = self.model.rollout_buffer.observations
                    if obs_buffer is not None and len(obs_buffer) > 0:
                        # 获取最后一个observation（episode结束时的）
                        if isinstance(obs_buffer, np.ndarray):
                            if len(obs_buffer.shape) == 4:  # (n_steps, n_envs, C, H, W)
                                last_obs = obs_buffer[-1, 0]  # 最后一个step，第一个env
                            elif len(obs_buffer.shape) == 3:  # (n_steps, C, H, W)
                                last_obs = obs_buffer[-1]
                            else:
                                last_obs = None
                            
                            if last_obs is not None and len(last_obs.shape) == 3:
                                self.last_obs = last_obs
                                if self.verbose > 0:
                                    print(f"[RenderLogger] 从rollout_buffer获取observation: shape={last_obs.shape}")
            except Exception as e:
                if self.verbose > 0:
                    print(f"[RenderLogger] 从rollout_buffer获取observation失败: {e}")
    
    def _log_observation(self, obs: np.ndarray, tag: str, global_step: int):
        """将observation记录到TensorBoard和本地文件"""
        try:
            # 确保obs是正确的格式
            if obs is None:
                print(f"[RenderLogger] ✗ 警告: observation为None，跳过 {tag}")
                return
            
            if not isinstance(obs, np.ndarray):
                obs = np.asarray(obs)
            
            if len(obs.shape) != 3:
                print(f"[RenderLogger] ✗ 警告: observation形状不正确 {obs.shape}，期望 (C, H, W)，跳过 {tag}")
                return
            
            # 打印调试信息
            if self.verbose > 0:
                print(f"[RenderLogger] 处理observation: shape={obs.shape}, tag={tag}, step={global_step}")
            
            # 创建可视化图像
            fig = self._create_visualization(obs, self.channel_names)
            
            # 转换为TensorBoard格式
            img_tensor = self._fig_to_tensor(fig)
            
            # 记录到TensorBoard
            self.writer.add_image(tag, img_tensor, global_step=global_step)
            
            # 保存到本地文件（复用已创建的图像）
            if self.save_dir:
                if self.verbose > 0:
                    print(f"[RenderLogger] 准备保存到本地: {self.save_dir}")
                self._save_observation_image(fig, tag, global_step)
            else:
                plt.close(fig)
                if self.verbose > 0:
                    print(f"[RenderLogger] 已记录到TensorBoard {tag} (step={global_step})，但未保存到本地（save_dir=None）")
            
            if self.verbose > 0:
                print(f"[RenderLogger] ✓ 已记录 {tag} (step={global_step})")
        except Exception as e:
            print(f"[RenderLogger] ✗ 记录失败 {tag}: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_observation_image(self, fig: plt.Figure, tag: str, epoch: int):
        """保存observation图像到本地文件"""
        if not self.save_dir:
            plt.close(fig)
            print(f"[RenderLogger] ✗ 无法保存: save_dir未设置")
            return
        
        # 确保目录存在
        if not os.path.exists(self.save_dir):
            try:
                os.makedirs(self.save_dir, exist_ok=True)
                print(f"[RenderLogger] 已创建保存目录: {self.save_dir}")
            except Exception as e:
                print(f"[RenderLogger] ✗ 创建目录失败: {self.save_dir}, 错误: {e}")
                plt.close(fig)
                return
        
        try:
            # 生成文件名
            # tag格式: "render/episode_X_final" 或 "render/episode_X_initial" 或 "render/step_X"
            if "episode" in tag:
                if "initial" in tag:
                    filename = f'frame_epoch_{epoch:04d}_initial.png'
                else:
                    filename = f'frame_epoch_{epoch:04d}_final.png'
            elif "step" in tag:
                filename = f'frame_step_{epoch:06d}.png'
            else:
                filename = f'frame_{epoch:06d}.png'
            
            filepath = os.path.join(self.save_dir, filename)
            
            # 保存图像，确保每个通道是256x256像素
            # 使用固定的DPI和尺寸，不使用tight bounding box
            # figure尺寸是 (n_channels * 2.56, 2.56) 英寸，DPI=100
            # 这样总像素尺寸是 (n_channels * 256, 256)
            target_dpi = 100
            
            # 保存图像，保持固定尺寸（不使用tight，避免裁剪）
            fig.savefig(
                filepath, 
                dpi=target_dpi, 
                bbox_inches=None,  # 不使用tight，保持figure的固定尺寸
                pad_inches=0,  # 无额外内边距
                facecolor='white',
                edgecolor='none',
                format='png'
            )
            plt.close(fig)
            
            # 验证文件是否真的保存了
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"[RenderLogger] ✓ 已保存到本地: {filepath} ({file_size} bytes)")
            else:
                print(f"[RenderLogger] ✗ 警告: 文件保存失败，文件不存在: {filepath}")
                print(f"[RenderLogger]   尝试的路径: {os.path.abspath(filepath)}")
        except Exception as e:
            plt.close(fig)
            print(f"[RenderLogger] ✗ 保存本地文件失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_visualization(self, obs: np.ndarray, channel_names: list) -> plt.Figure:
        """创建可视化图像，保持256x256的尺寸和正确比例（直接调用visualize_raster_channels函数）"""
        # 直接调用统一的visualize_raster_channels函数，设置return_figure=True以获取figure对象
        fig = visualize_raster_channels(
            obs=obs,
            channel_names=channel_names,
            output_path=None,
            maintain_256x256=True,
            return_figure=True
        )
        if fig is None:
            raise RuntimeError("无法创建可视化图像")
        return fig
    
    def _fig_to_tensor(self, fig: plt.Figure) -> np.ndarray:
        """将matplotlib figure转换为numpy数组（TensorBoard格式）"""
        # 保存到内存缓冲区
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # 读取为numpy数组
        import matplotlib.image as mpimg
        img = mpimg.imread(buf)
        buf.close()
        
        # 转换为 (C, H, W) 格式（TensorBoard需要）
        if len(img.shape) == 3:
            # RGB图像，转换为 (H, W, C) -> (C, H, W)
            img = np.transpose(img, (2, 0, 1))
        elif len(img.shape) == 2:
            # 灰度图像，添加通道维度
            img = img[np.newaxis, :, :]
        
        return img
    
    def _on_training_end(self) -> None:
        """训练结束时关闭writer"""
        if hasattr(self, 'writer'):
            self.writer.close()
            if self.verbose > 0:
                print(f"[RenderLogger] TensorBoard日志已保存到: {self.log_dir}")
    
    def __del__(self):
        """析构函数，确保writer被关闭"""
        if hasattr(self, 'writer'):
            try:
                self.writer.close()
            except:
                pass

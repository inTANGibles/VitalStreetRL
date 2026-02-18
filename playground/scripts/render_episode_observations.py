"""渲染Episode的Observation工具"""
import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import os
import sys

# 添加项目根目录到Python路径（确保可以导入项目模块）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from rl.train_ppo import VitalStreetEnv
from env.representation.visualization import visualize_raster_channels


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def render_episodes(
    checkpoint_path: str,
    config_path: str,
    output_dir: str,
    n_episodes: int = 5,
    save_every_n_steps: int = 1
):
    """
    运行模型并渲染每个Episode的Observation（三个通道）
    
    Args:
        checkpoint_path: 模型检查点路径
        config_path: 配置文件路径
        output_dir: 输出目录
        n_episodes: 要渲染的episode数量
        save_every_n_steps: 每N步保存一次observation（1表示每步都保存）
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    config = load_config(config_path)
    train_config = config.get('train', config)
    
    # 加载环境配置
    if 'env' not in train_config:
        env_config = load_config('configs/env.yaml')
        train_config['env'] = env_config.get('env', {})
    
    # 加载奖励配置
    if 'reward' not in train_config:
        reward_config = load_config('configs/reward.yaml')
        train_config['reward'] = reward_config.get('reward', {})
    
    print("=" * 80)
    print("渲染Episode Observations")
    print("=" * 80)
    print(f"模型检查点: {checkpoint_path}")
    print(f"输出目录: {output_dir}")
    print(f"Episode数量: {n_episodes}")
    print(f"每 {save_every_n_steps} 步保存一次")
    print("=" * 80 + "\n")
    
    # 创建环境
    def make_env():
        env = VitalStreetEnv(train_config)
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # 加载模型
    print(f"加载模型: {checkpoint_path}")
    try:
        model = PPO.load(checkpoint_path, env=env)
        print("模型加载成功\n")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 获取通道名称
    obs_config = train_config.get('env', {}).get('representation', {}).get('raster', {})
    channel_names = obs_config.get('channels', ['walkable_mask', 'predicted_flow', 'landuse_id'])
    print(f"通道名称: {channel_names}")
    print(f"通道数量: {len(channel_names)}")
    
    # 运行episodes
    for episode_idx in range(n_episodes):
        print(f"\n{'='*80}")
        print(f"Episode {episode_idx + 1}/{n_episodes}")
        print(f"{'='*80}")
        
        # 创建episode输出目录
        episode_dir = output_dir / f"episode_{episode_idx + 1:04d}"
        episode_dir.mkdir(exist_ok=True)
        
        # 重置环境
        obs, info = env.reset()
        episode_id = info[0].get('episode_id', f'episode_{episode_idx + 1}')
        
        print(f"Episode ID: {episode_id}")
        print(f"Observation形状: {obs[0].shape if isinstance(obs, (list, tuple, np.ndarray)) else obs.shape}")
        print(f"输出目录: {episode_dir.absolute()}")
        
        step_count = 0
        done = False
        
        # 保存初始observation
        initial_path = episode_dir / "step_000_initial_channels.png"
        try:
            print(f"  保存初始observation 到: {initial_path}")
            visualize_raster_channels(
                obs[0],  # 从vec_env中提取单个观测
                channel_names,
                output_path=str(initial_path),
                maintain_256x256=True  # 保持256x256尺寸
            )
            if initial_path.exists():
                print(f"  ✓ 已保存: {initial_path}")
            else:
                print(f"  ✗ 保存失败: {initial_path} 不存在")
        except Exception as e:
            print(f"  ✗ 保存初始observation失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 运行episode
        while not done:
            # 预测动作
            action, _ = model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            
            step_count += 1
            step_info = info[0]
            
            # 每N步保存一次observation
            if step_count % save_every_n_steps == 0 or done[0]:
                step_str = f"step_{step_count:03d}"
                channels_path = episode_dir / f"{step_str}_channels.png"
                try:
                    visualize_raster_channels(
                        obs[0],
                        channel_names,
                        output_path=str(channels_path),
                        maintain_256x256=True  # 保持256x256尺寸
                    )
                    if not channels_path.exists():
                        print(f"  ⚠ 警告: {channels_path} 未成功保存")
                except Exception as e:
                    print(f"  ✗ 保存 {channels_path} 失败: {e}")
            
            # 打印步骤信息
            if step_count % 5 == 0 or done[0]:
                reward_val = reward[0] if isinstance(reward, np.ndarray) else reward
                print(f"  Step {step_count:3d} | 奖励: {reward_val:7.3f} | "
                      f"累计奖励: {step_info.get('episode', {}).get('r', 0.0):7.3f}")
            
            if done[0] or truncated[0]:
                break
        
        # Episode结束信息
        final_reward = step_info.get('episode', {}).get('r', 0.0)
        termination_reason = step_info.get('termination_reason', 'unknown')
        
        # 检查保存的文件
        saved_files = list(episode_dir.glob("*.png"))
        print(f"\nEpisode {episode_idx + 1} 结束:")
        print(f"  总步数: {step_count}")
        print(f"  累计奖励: {final_reward:.3f}")
        print(f"  终止原因: {termination_reason}")
        print(f"  输出目录: {episode_dir}")
        print(f"  已保存图像数: {len(saved_files)}")
        if len(saved_files) == 0:
            print(f"  ⚠ 警告: 没有保存任何图像文件！")
        else:
            print(f"  图像文件列表:")
            for f in sorted(saved_files)[:5]:  # 只显示前5个
                print(f"    - {f.name}")
            if len(saved_files) > 5:
                print(f"    ... 还有 {len(saved_files) - 5} 个文件")
    
    # 最终统计
    total_files = 0
    for episode_dir in output_dir.glob("episode_*"):
        if episode_dir.is_dir():
            files = list(episode_dir.glob("*.png"))
            total_files += len(files)
    
    print("\n" + "=" * 80)
    print("渲染完成！")
    print(f"输出目录: {output_dir}")
    print(f"总Episode数: {n_episodes}")
    print(f"总图像文件数: {total_files}")
    
    if total_files == 0:
        print("\n⚠ 警告: 没有保存任何图像文件！")
        print("可能的原因:")
        print("  1. observation为空或格式不正确")
        print("  2. matplotlib保存失败（检查权限）")
        print("  3. 输出目录路径问题")
        print(f"\n请检查输出目录: {output_dir.absolute()}")
    else:
        print(f"\n✓ 成功保存 {total_files} 个图像文件")
        print(f"查看图像: 打开文件夹 {output_dir.absolute()}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='渲染Episode的Observation（三个通道）')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='配置文件路径')
    parser.add_argument('--output-dir', type=str, default='./logs/episode_observations', help='输出目录')
    parser.add_argument('--n-episodes', type=int, default=5, help='要渲染的episode数量')
    parser.add_argument('--save-every-n-steps', type=int, default=1, help='每N步保存一次observation')
    
    args = parser.parse_args()
    
    render_episodes(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        n_episodes=args.n_episodes,
        save_every_n_steps=args.save_every_n_steps
    )


if __name__ == '__main__':
    main()

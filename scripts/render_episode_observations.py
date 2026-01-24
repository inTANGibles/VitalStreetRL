"""渲染Episode的Observation工具"""
import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from rl.train_ppo import VitalStreetEnv
from env.representation.visualization import visualize_raster_channels, visualize_rgb_composite


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def render_episodes(
    checkpoint_path: str,
    config_path: str,
    output_dir: str,
    n_episodes: int = 5,
    save_every_n_steps: int = 1,
    render_channels: bool = True,
    render_rgb: bool = True
):
    """
    运行模型并渲染每个Episode的Observation
    
    Args:
        checkpoint_path: 模型检查点路径
        config_path: 配置文件路径
        output_dir: 输出目录
        n_episodes: 要渲染的episode数量
        save_every_n_steps: 每N步保存一次observation（1表示每步都保存）
        render_channels: 是否渲染各通道图像
        render_rgb: 是否渲染RGB合成图
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
        
        step_count = 0
        done = False
        
        # 保存初始observation
        if render_channels:
            initial_path = episode_dir / "step_000_initial_channels.png"
            visualize_raster_channels(
                obs[0],  # 从vec_env中提取单个观测
                channel_names,
                output_path=str(initial_path)
            )
        
        if render_rgb:
            initial_rgb_path = episode_dir / "step_000_initial_rgb.png"
            visualize_rgb_composite(
                obs[0],
                channel_names,
                output_path=str(initial_rgb_path)
            )
        
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
                
                if render_channels:
                    channels_path = episode_dir / f"{step_str}_channels.png"
                    visualize_raster_channels(
                        obs[0],
                        channel_names,
                        output_path=str(channels_path)
                    )
                
                if render_rgb:
                    rgb_path = episode_dir / f"{step_str}_rgb.png"
                    visualize_rgb_composite(
                        obs[0],
                        channel_names,
                        output_path=str(rgb_path)
                    )
            
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
        print(f"\nEpisode {episode_idx + 1} 结束:")
        print(f"  总步数: {step_count}")
        print(f"  累计奖励: {final_reward:.3f}")
        print(f"  终止原因: {termination_reason}")
        print(f"  输出目录: {episode_dir}")
    
    print("\n" + "=" * 80)
    print("渲染完成！")
    print(f"所有图像已保存到: {output_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='渲染Episode的Observation')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='配置文件路径')
    parser.add_argument('--output-dir', type=str, default='./logs/episode_observations', help='输出目录')
    parser.add_argument('--n-episodes', type=int, default=5, help='要渲染的episode数量')
    parser.add_argument('--save-every-n-steps', type=int, default=1, help='每N步保存一次observation')
    parser.add_argument('--no-channels', action='store_true', help='不渲染各通道图像')
    parser.add_argument('--no-rgb', action='store_true', help='不渲染RGB合成图')
    
    args = parser.parse_args()
    
    render_episodes(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        n_episodes=args.n_episodes,
        save_every_n_steps=args.save_every_n_steps,
        render_channels=not args.no_channels,
        render_rgb=not args.no_rgb
    )


if __name__ == '__main__':
    main()

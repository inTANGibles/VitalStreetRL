"""主入口：训练与评估"""
import argparse
import yaml
from pathlib import Path
import numpy as np
from rl.train_ppo import PPOTrainer
# from logging.episode_logger import EpisodeLogger  # 暂时未使用，注释掉


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='VitalStreetRL训练')
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='训练配置文件')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'test'], default='train', help='运行模式')
    parser.add_argument('--checkpoint', type=str, default=None, help='检查点路径（用于eval/test）')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 加载环境配置（如果train.yaml中没有）
    if 'env' not in config:
        cfg = load_config('configs/env.yaml')
        config['env'] = cfg.get('env', {})
    if args.mode == 'train':
        train_config = config['train']
        train_config['env'] = config['env']
        if 'reward' not in config:
            cfg = load_config('configs/reward.yaml')
            config['reward'] = cfg.get('reward', {})
        train_config['reward'] = config.get('reward', {})
        
        trainer = PPOTrainer(train_config)
        
        # 计算总步数（从episode数估算，或直接使用timesteps）
        n_episodes = train_config.get('n_episodes', 1000)
        # 假设平均每个episode 100步
        total_timesteps = n_episodes * 20
        
        trainer.train(
            total_timesteps=total_timesteps,
            log_interval=train_config.get('log_interval', 10)
        )
        print("训练完成！")
    
    elif args.mode == 'eval':
        # 评估模式：运行多个episode并统计性能
        if args.checkpoint is None:
            print("错误: eval模式需要指定 --checkpoint 参数")
            return
        
        train_config = config.get('train', config)
        train_config['env'] = config.get('env', {})
        if 'reward' not in train_config:
            cfg = load_config('configs/reward.yaml')
            train_config['reward'] = cfg.get('reward', {})
        
        # 使用render_episode_observations脚本进行评估
        print("提示: 使用以下命令进行详细评估和observation渲染:")
        print(f"  python scripts/render_episode_observations.py --checkpoint {args.checkpoint} --config {args.config}")
        print("\n或者直接运行评估...")
        
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        from rl.train_ppo import VitalStreetEnv
        
        def make_env():
            env = VitalStreetEnv(train_config)
            env = Monitor(env)
            return env
        
        env = DummyVecEnv([make_env])
        model = PPO.load(args.checkpoint, env=env)
        
        n_eval_episodes = train_config.get('n_eval_episodes', 10)
        print(f"\n运行 {n_eval_episodes} 个评估episode...")
        
        rewards = []
        lengths = []
        
        for i in range(n_eval_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward[0] if isinstance(reward, (list, np.ndarray)) else reward
                episode_length += 1
                
                if done[0] or truncated[0]:
                    break
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
            print(f"Episode {i+1}: 奖励={episode_reward:.3f}, 长度={episode_length}")
        
        print(f"\n评估结果:")
        print(f"  平均奖励: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
        print(f"  平均长度: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        print(f"  最大奖励: {np.max(rewards):.3f}")
        print(f"  最小奖励: {np.min(rewards):.3f}")
    
    elif args.mode == 'test':
        # 测试模式：运行单个episode并输出详细日志
        if args.checkpoint is None:
            print("错误: test模式需要指定 --checkpoint 参数")
            return
        
        train_config = config.get('train', config)
        train_config['env'] = config.get('env', {})
        if 'reward' not in train_config:
            cfg = load_config('configs/reward.yaml')
            train_config['reward'] = cfg.get('reward', {})
        
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        from rl.train_ppo import VitalStreetEnv
        import numpy as np
        
        def make_env():
            env = VitalStreetEnv(train_config)
            env = Monitor(env)
            return env
        
        env = DummyVecEnv([make_env])
        model = PPO.load(args.checkpoint, env=env)
        
        print("\n" + "=" * 80)
        print("测试模式: 运行单个Episode")
        print("=" * 80)
        
        obs, info = env.reset()
        episode_id = info[0].get('episode_id', 'test_episode')
        print(f"Episode ID: {episode_id}\n")
        
        step = 0
        total_reward = 0.0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            step += 1
            reward_val = reward[0] if isinstance(reward, (list, np.ndarray)) else reward
            total_reward += reward_val
            
            step_info = info[0]
            action_type = step_info.get('transition_info', {}).get('action_type', 'unknown')
            
            print(f"Step {step:3d} | 动作: {action_type:25s} | "
                  f"奖励: {reward_val:7.3f} | 累计: {total_reward:7.3f}")
            
            if done[0] or truncated[0]:
                termination_reason = step_info.get('termination_reason', 'unknown')
                print(f"\nEpisode结束:")
                print(f"  总步数: {step}")
                print(f"  累计奖励: {total_reward:.3f}")
                print(f"  终止原因: {termination_reason}")
                break


if __name__ == '__main__':
    main()

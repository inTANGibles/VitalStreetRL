"""主入口：训练与评估"""
import argparse
import yaml
from pathlib import Path
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
        total_timesteps = n_episodes * 100
        
        trainer.train(
            total_timesteps=total_timesteps,
            log_interval=train_config.get('log_interval', 10)
        )
        print("训练完成！")
    
    elif args.mode == 'eval':
        # 评估模式
        pass
    
    elif args.mode == 'test':
        # 测试模式（运行单个episode并输出日志）
        pass


if __name__ == '__main__':
    main()

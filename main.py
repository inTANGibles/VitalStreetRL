"""主入口：训练与评估"""
import argparse
import yaml
from pathlib import Path
from rl.train_ppo import PPOTrainer
from logging.episode_logger import EpisodeLogger


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
    
    if args.mode == 'train':
        # 训练模式
        logger = EpisodeLogger(log_dir=config['train']['checkpoint_dir'].replace('checkpoints', 'logs'))
        trainer = PPOTrainer(config['train'])
        metrics = trainer.train(
            n_episodes=config['train']['n_episodes'],
            log_interval=config['train']['log_interval']
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

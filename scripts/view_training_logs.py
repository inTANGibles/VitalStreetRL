"""查看训练日志工具"""
import argparse
import pandas as pd
import os
from pathlib import Path
from typing import Optional
import glob


def find_monitor_files(log_dir: str = "./logs") -> list:
    """
    查找所有monitor.csv文件
    
    Args:
        log_dir: 日志目录
        
    Returns:
        monitor_files: monitor.csv文件路径列表
    """
    monitor_files = []
    
    # 在指定目录下查找
    if os.path.exists(log_dir):
        pattern = os.path.join(log_dir, "**", "monitor.csv")
        monitor_files.extend(glob.glob(pattern, recursive=True))
    
    # 也在当前目录和常见位置查找
    common_dirs = [".", "./logs", "./train_logs", "./output"]
    for dir_path in common_dirs:
        if os.path.exists(dir_path):
            pattern = os.path.join(dir_path, "**", "monitor.csv")
            monitor_files.extend(glob.glob(pattern, recursive=True))
    
    # 去重
    monitor_files = list(set(monitor_files))
    return monitor_files


def load_monitor_log(monitor_path: str) -> pd.DataFrame:
    """
    加载monitor.csv日志文件
    
    Args:
        monitor_path: monitor.csv文件路径
        
    Returns:
        df: 包含训练日志的DataFrame
    """
    try:
        df = pd.read_csv(monitor_path, skiprows=1)  # 跳过第一行（元数据）
        return df
    except Exception as e:
        print(f"加载日志文件失败: {e}")
        return pd.DataFrame()


def print_episode_summary(df: pd.DataFrame, n_episodes: Optional[int] = None):
    """
    打印Episode摘要统计
    
    Args:
        df: 训练日志DataFrame
        n_episodes: 要显示的episode数量（None表示显示所有）
    """
    if df.empty:
        print("日志文件为空")
        return
    
    # 确保有必要的列
    required_cols = ['r', 'l', 't']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"警告: 日志文件缺少列: {missing_cols}")
        print(f"可用列: {df.columns.tolist()}")
        return
    
    # 重命名列以便理解
    df_display = df.copy()
    df_display = df_display.rename(columns={
        'r': '累计奖励',
        'l': 'Episode长度',
        't': '时间(秒)'
    })
    
    # 选择要显示的列
    display_cols = ['累计奖励', 'Episode长度', '时间(秒)']
    if 'episode' in df.columns:
        display_cols.insert(0, 'episode')
    
    # 限制显示数量
    if n_episodes is not None:
        df_display = df_display.tail(n_episodes)
    
    print("\n" + "=" * 80)
    print("Episode 摘要统计")
    print("=" * 80)
    print(df_display[display_cols].to_string(index=False))
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("总体统计")
    print("=" * 80)
    print(f"总Episode数: {len(df)}")
    print(f"平均累计奖励: {df['r'].mean():.3f}")
    print(f"奖励标准差: {df['r'].std():.3f}")
    print(f"最大奖励: {df['r'].max():.3f}")
    print(f"最小奖励: {df['r'].min():.3f}")
    print(f"平均Episode长度: {df['l'].mean():.1f}")
    print(f"总训练时间: {df['t'].sum():.2f} 秒 ({df['t'].sum()/60:.2f} 分钟)")
    
    # 最近N个episode的统计
    if len(df) >= 10:
        recent_df = df.tail(10)
        print(f"\n最近10个Episode:")
        print(f"  平均奖励: {recent_df['r'].mean():.3f}")
        print(f"  平均长度: {recent_df['l'].mean():.1f}")


def plot_training_curve(df: pd.DataFrame, output_path: Optional[str] = None):
    """
    绘制训练曲线
    
    Args:
        df: 训练日志DataFrame
        output_path: 输出图像路径（可选）
    """
    try:
        import matplotlib.pyplot as plt
        
        if df.empty or 'r' not in df.columns:
            print("无法绘制训练曲线：数据为空或缺少必要列")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Episode索引
        episode_idx = range(len(df))
        
        # 奖励曲线
        axes[0].plot(episode_idx, df['r'], alpha=0.6, label='Episode奖励')
        if len(df) > 10:
            # 移动平均
            window = min(10, len(df) // 10)
            rolling_mean = df['r'].rolling(window=window, center=True).mean()
            axes[0].plot(episode_idx, rolling_mean, 'r-', linewidth=2, label=f'{window}期移动平均')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('累计奖励')
        axes[0].set_title('训练奖励曲线')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Episode长度曲线
        axes[1].plot(episode_idx, df['l'], alpha=0.6, label='Episode长度')
        if len(df) > 10:
            window = min(10, len(df) // 10)
            rolling_mean = df['l'].rolling(window=window, center=True).mean()
            axes[1].plot(episode_idx, rolling_mean, 'r-', linewidth=2, label=f'{window}期移动平均')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('步数')
        axes[1].set_title('Episode长度曲线')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n训练曲线已保存到: {output_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("警告: matplotlib未安装，无法绘制训练曲线")
    except Exception as e:
        print(f"绘制训练曲线时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description='查看训练日志')
    parser.add_argument('--log-dir', type=str, default='./logs', help='日志目录')
    parser.add_argument('--monitor-file', type=str, default=None, help='monitor.csv文件路径（直接指定）')
    parser.add_argument('--n-episodes', type=int, default=None, help='显示的episode数量（默认显示所有）')
    parser.add_argument('--plot', action='store_true', help='绘制训练曲线')
    parser.add_argument('--plot-output', type=str, default=None, help='训练曲线输出路径')
    
    args = parser.parse_args()
    
    # 查找或使用指定的monitor文件
    if args.monitor_file:
        monitor_files = [args.monitor_file]
    else:
        monitor_files = find_monitor_files(args.log_dir)
    
    if not monitor_files:
        print("未找到monitor.csv文件")
        print("请检查以下位置:")
        print(f"  - {args.log_dir}")
        print("  - ./logs")
        print("  - ./train_logs")
        print("\n或者使用 --monitor-file 参数直接指定文件路径")
        return
    
    # 如果有多个文件，让用户选择
    if len(monitor_files) > 1:
        print(f"找到 {len(monitor_files)} 个monitor.csv文件:")
        for i, f in enumerate(monitor_files):
            print(f"  [{i+1}] {f}")
        
        try:
            choice = input(f"\n请选择文件 (1-{len(monitor_files)})，或按Enter使用第一个: ").strip()
            if choice:
                idx = int(choice) - 1
                if 0 <= idx < len(monitor_files):
                    monitor_path = monitor_files[idx]
                else:
                    print("无效选择，使用第一个文件")
                    monitor_path = monitor_files[0]
            else:
                monitor_path = monitor_files[0]
        except (ValueError, KeyboardInterrupt):
            print("\n使用第一个文件")
            monitor_path = monitor_files[0]
    else:
        monitor_path = monitor_files[0]
    
    print(f"\n加载日志文件: {monitor_path}")
    
    # 加载日志
    df = load_monitor_log(monitor_path)
    
    if df.empty:
        print("日志文件为空或无法加载")
        return
    
    # 打印摘要
    print_episode_summary(df, args.n_episodes)
    
    # 绘制曲线
    if args.plot:
        plot_output = args.plot_output or os.path.join(os.path.dirname(monitor_path), 'training_curve.png')
        plot_training_curve(df, plot_output)


if __name__ == '__main__':
    main()

"""栅格观测可视化工具"""
import numpy as np
import matplotlib
# 不在导入时设置后端，而是在函数内部根据环境动态设置
import matplotlib.pyplot as plt
from typing import Optional, List, Union


def _is_jupyter_environment():
    """检测是否在Jupyter环境中"""
    # 方法1: 检查matplotlib当前使用的后端（最可靠）
    try:
        backend = matplotlib.get_backend().lower()
        # 如果后端是inline、notebook、widget等，说明在Jupyter中
        jupyter_backends = ['inline', 'notebook', 'widget', 'ipympl', 'nbagg']
        if any(b in backend for b in jupyter_backends):
            return True
    except:
        pass
    
    # 方法2: 检查是否有get_ipython函数
    try:
        get_ipython()
        return True
    except NameError:
        pass
    
    # 方法3: 检查是否在IPython环境中（通过检查模块）
    try:
        import sys
        if 'ipykernel' in sys.modules or 'IPython' in sys.modules:
            return True
    except:
        pass
    
    return False


def visualize_raster_channels(
    obs: np.ndarray,
    channel_names: List[str],
    output_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
    dpi: int = 100,
    maintain_256x256: bool = True,
    return_figure: bool = False
) -> Union[plt.Figure, None]:
    """
    可视化栅格通道，支持保持256x256尺寸和正确比例
    
    Args:
        obs: 栅格观测数组 (C, H, W)
        channel_names: 通道名称列表
        output_path: 输出路径（可选）
        figsize: 图像大小（可选，如果maintain_256x256=True则自动计算）
        dpi: 分辨率（默认100，用于保证256x256）
        maintain_256x256: 是否保持256x256尺寸（默认True）
        return_figure: 是否返回figure对象而不关闭（默认False）
    
    Returns:
        Figure对象（如果return_figure=True或在Jupyter中且没有指定output_path），否则返回None
    """
    # 确保obs是正确的格式
    if len(obs.shape) != 3:
        raise ValueError(f"Observation形状不正确: {obs.shape}，期望 (C, H, W)")
    
    n_channels, H, W = obs.shape
    
    # 如果启用256x256模式，使用固定尺寸
    # 注意：如果obs已经被填充到(3, 256, 256)，H和W应该是256
    # 如果obs是(3, 144, 256)，需要确保显示完整的256x256（包括填充的黑色部分）
    if maintain_256x256:
        # 每个通道2.56英寸（256像素/100 DPI）
        fig_width = n_channels * 2.56
        fig_height = 2.56
        save_dpi = 100
        use_tight = False  # 不使用tight，保持固定尺寸
        # 强制使用256x256作为显示范围（即使obs的实际高度是144，也要显示完整的256x256）
        target_H, target_W = 256, 256
    else:
        # 使用传统模式（灵活尺寸）
        if figsize is None:
            fig_width = n_channels * 5
            fig_height = 5
        else:
            fig_width, fig_height = figsize
        save_dpi = dpi
        use_tight = True  # 使用tight bounding box
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    
    axes = []
    for i in range(n_channels):
        ax = fig.add_subplot(1, n_channels, i + 1)
        axes.append(ax)
    
    for i, (ax, channel_name) in enumerate(zip(axes, channel_names)):
        channel_data = obs[i]
        
        # 根据通道类型选择不同的colormap
        if channel_name == 'walkable_mask':
            cmap = 'gray'
            vmin, vmax = 0, 1
        elif channel_name == 'predicted_flow':
            cmap = 'hot'
            vmin, vmax = 0, 1
        elif channel_name == 'landuse_id':
            cmap = 'viridis'
            vmin, vmax = 0, 1
        else:
            cmap = 'viridis'
            vmin, vmax = None, None
        
        # 显示图像
        if maintain_256x256:
            # obs是(3, 144, 256)，需要填充到256x256显示
            # 创建256x256的画布，将channel_data居中放置（上下填充黑色）
            padded_channel = np.zeros((target_H, target_W), dtype=channel_data.dtype)
            h_start = (target_H - channel_data.shape[0]) // 2
            w_start = (target_W - channel_data.shape[1]) // 2
            padded_channel[h_start:h_start+channel_data.shape[0], w_start:w_start+channel_data.shape[1]] = channel_data
            
            # 使用extent和aspect保持宽高比，显示完整的256x256（包括填充的黑色部分）
            im = ax.imshow(
                padded_channel,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                origin='lower',
                extent=[0, target_W, 0, target_H],  # 像素坐标范围（256x256）
                aspect='equal',  # 保持宽高比
                interpolation='nearest'  # 保持像素清晰
            )
            ax.set_xlim(0, target_W)
            ax.set_ylim(0, target_H)
        else:
            # 传统模式
            im = ax.imshow(channel_data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        
        ax.set_title(f'通道 {i+1}: {channel_name}', fontsize=10 if maintain_256x256 else 12, fontweight='bold')
        ax.set_xlabel('X (像素)')
        ax.set_ylabel('Y (像素)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout(pad=0.1 if maintain_256x256 else None)
    
    if output_path:
        try:
            plt.savefig(
                output_path,
                dpi=save_dpi,
                bbox_inches='tight' if use_tight else None,
                pad_inches=0 if maintain_256x256 else 0.1,
                facecolor='white',
                edgecolor='none',
                format='png'
            )
            import os
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"可视化图像已保存到: {output_path} ({file_size} bytes)")
            else:
                print(f"警告: 保存命令执行但文件不存在: {output_path}")
            if return_figure:
                return fig
            plt.close(fig)
            return None
        except Exception as e:
            print(f"保存图像失败: {e}")
            import traceback
            traceback.print_exc()
            if return_figure:
                return fig
            plt.close(fig)
            return None
    else:
        # 如果指定了return_figure，直接返回figure
        if return_figure:
            return fig
        
        # 动态检测是否在Jupyter环境中
        is_jupyter = _is_jupyter_environment()
        
        if is_jupyter:
            # 在Jupyter中，尝试使用IPython的display来显示
            try:
                from IPython.display import display
                display(fig)
                # 不关闭figure，让Jupyter管理
                return fig
            except ImportError:
                # 如果没有IPython.display，返回figure对象让Jupyter自动显示
                return fig
        else:
            # 在脚本中，尝试设置Agg后端（如果还没设置）
            try:
                current_backend = matplotlib.get_backend().lower()
                if 'agg' not in current_backend:
                    matplotlib.use('Agg')
            except:
                pass
            # 显示并关闭
            try:
                plt.show()
            except:
                # 如果show失败（比如在服务器上），只关闭figure
                pass
            plt.close(fig)
            return None


"""栅格观测可视化工具"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List


def visualize_raster_channels(
    obs: np.ndarray,
    channel_names: List[str],
    output_path: Optional[str] = None,
    figsize: tuple = (15, 5),
    dpi: int = 150
):
    """
    可视化栅格通道
    
    Args:
        obs: 栅格观测数组 (C, H, W)
        channel_names: 通道名称列表
        output_path: 输出路径（可选）
        figsize: 图像大小
        dpi: 分辨率
    """
    fig, axes = plt.subplots(1, obs.shape[0], figsize=figsize)
    if obs.shape[0] == 1:
        axes = [axes]
    
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
        
        im = ax.imshow(channel_data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(f'通道 {i+1}: {channel_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X (像素)')
        ax.set_ylabel('Y (像素)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"可视化图像已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_rgb_composite(
    obs: np.ndarray,
    channel_names: List[str],
    output_path: Optional[str] = None,
    figsize: tuple = (12, 12),
    dpi: int = 150
):
    """
    可视化RGB合成图
    
    Args:
        obs: 栅格观测数组 (C, H, W)
        channel_names: 通道名称列表
        output_path: 输出路径（可选）
        figsize: 图像大小
        dpi: 分辨率
    """
    if obs.shape[0] < 3:
        print("需要至少3个通道才能进行RGB合成")
        return
    
    # 将3个通道映射到RGB
    rgb_image = np.zeros((obs.shape[1], obs.shape[2], 3), dtype=np.float32)
    rgb_image[:, :, 0] = obs[0]  # R: walkable_mask
    rgb_image[:, :, 1] = obs[1]  # G: predicted_flow
    rgb_image[:, :, 2] = obs[2]  # B: landuse_id
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb_image, origin='lower')
    ax.set_title('RGB合成图 (R=可走区域, G=预测流量, B=土地利用)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('X (像素)')
    ax.set_ylabel('Y (像素)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"RGB合成图已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()

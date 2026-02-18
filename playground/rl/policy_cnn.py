"""CNN策略：用于Raster观测

注意：当前训练使用的是 stable-baselines3 内置的 'CnnPolicy'，而不是这个自定义的 CNNActor 类。
这个文件中的 CNNActor 类目前只是占位符，未实现。

如果将来需要自定义CNN策略，可以在这里实现 CNNActor 类，然后修改 train_ppo.py 来使用它。
"""
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np


class CNNActor(nn.Module):
    """
    CNN Actor网络（用于栅格观测）
    
    注意：此类目前未实现，仅作为占位符。
    当前训练使用的是 stable-baselines3 内置的 'CnnPolicy'。
    """
    
    def __init__(self, obs_shape: Tuple[int, int, int], action_dim: Tuple[int, ...], config: Dict[str, Any]):
        """
        Args:
            obs_shape: (C, H, W)
            action_dim: multi-discrete动作维度
            config: 网络配置
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        
        # TODO: 实现CNN encoder
        # TODO: 实现MLP head
        raise NotImplementedError("CNNActor 类尚未实现。当前使用 stable-baselines3 内置的 CnnPolicy。")
    
    def forward(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            action_logits: 动作logits
            value: 状态价值
        """
        raise NotImplementedError("CNNActor 类尚未实现。")
    
    def get_action(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        采样动作
        
        Returns:
            action: 动作
            log_prob: log概率
            value: 价值
        """
        raise NotImplementedError("CNNActor 类尚未实现。")

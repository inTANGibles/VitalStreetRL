"""CNN策略：用于Raster观测"""
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np


class CNNActor(nn.Module):
    """CNN Actor网络（用于栅格观测）"""
    
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
        
        # CNN encoder
        # MLP head
        pass
    
    def forward(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            action_logits: 动作logits
            value: 状态价值
        """
        pass
    
    def get_action(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        采样动作
        
        Returns:
            action: 动作
            log_prob: log概率
            value: 价值
        """
        pass

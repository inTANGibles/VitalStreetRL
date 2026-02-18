"""价值函数头：评估长期回报"""
from typing import Tuple
import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """价值函数头（可与Actor共享encoder）"""
    
    def __init__(self, feature_dim: int, hidden_dims: Tuple[int, ...] = (256, 128)):
        """
        Args:
            feature_dim: 编码器输出特征维度
            hidden_dims: MLP隐藏层维度
        """
        super().__init__()
        layers = []
        prev_dim = feature_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU()])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """输出状态价值"""
        return self.net(features).squeeze(-1)

"""
Homogeneous GraphSAGE: 2-layer message passing, regression head.
Input [N, 11], output [N, 1]. 仅 public 节点有标签，训练时用中心节点损失。
"""
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, Linear


class SAGE(nn.Module):
    def __init__(self, in_channels: int = 11, hidden_channels: int = 64, out_channels: int = 1):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        return self.lin(h)

"""
Homogeneous GraphSAGE: 2-layer message passing, regression head.
Input [N, 11], output [N, 1]. 仅 public 节点有标签，训练时用中心节点损失。
支持 dropout 以缓解过拟合、稳定 Val MAE。
"""
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, Linear


class SAGE(nn.Module):
    def __init__(
        self,
        in_channels: int = 11,
        hidden_channels: int = 64,
        out_channels: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout_p = dropout
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        if self.dropout_p > 0:
            h = nn.functional.dropout(h, p=self.dropout_p, training=self.training)
        h = self.conv2(h, edge_index).relu()
        if self.dropout_p > 0:
            h = nn.functional.dropout(h, p=self.dropout_p, training=self.training)
        return self.lin(h)

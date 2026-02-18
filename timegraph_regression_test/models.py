"""
模型定义：GCN 时间节点回归器 与 MLP 基线。
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNTimeNodeRegressor(nn.Module):
    """
    输入 data，输出每个节点的预测 [num_nodes]。
    结构：Embedding(func_type) + x_cont + time_feat -> GCNConv x2 -> Linear -> pred[101]
    """

    def __init__(
        self,
        in_cont: int = 11,
        num_func_types: int = 2,
        time_dim: int = 4,
        emb_dim: int = 8,
        hidden: int = 64,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_func_types, emb_dim)
        self.in_dim = in_cont + emb_dim + time_dim
        self.conv1 = GCNConv(self.in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, 1)

    def forward(self, data):
        x_cont = data.x_cont
        func_type = data.func_type
        time_feat = data.time_feat
        edge_index = data.edge_index

        x_emb = self.emb(func_type)
        x = torch.cat([x_cont, x_emb, time_feat], dim=-1)

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.lin(x).squeeze(-1)
        return x


class MLPTimeBaseline(nn.Module):
    """
    不使用 edge_index，直接对每个节点做 MLP 回归。
    输入：x_cont + func_type embedding + time_feat
    """

    def __init__(
        self,
        in_cont: int = 11,
        num_func_types: int = 2,
        time_dim: int = 4,
        emb_dim: int = 8,
        hidden: int = 64,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_func_types, emb_dim)
        self.in_dim = in_cont + emb_dim + time_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, data):
        x_cont = data.x_cont
        func_type = data.func_type
        time_feat = data.time_feat

        x_emb = self.emb(func_type)
        x = torch.cat([x_cont, x_emb, time_feat], dim=-1)
        return self.mlp(x).squeeze(-1)

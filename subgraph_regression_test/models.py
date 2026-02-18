"""
模型定义：GCN 节点回归器 与 MLP 中心节点基线。
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNNodeRegressor(nn.Module):
    """
    输入 data，输出每个节点的预测 [num_nodes_sub]。
    结构：Embedding(func_type) -> 拼接 x_cont -> GCNConv x2 -> Linear -> pred
    """

    def __init__(self, in_cont: int = 11, num_func_types: int = 2, emb_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.emb = nn.Embedding(num_func_types, emb_dim)
        self.in_dim = in_cont + emb_dim
        self.conv1 = GCNConv(self.in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, 1)

    def forward(self, data):
        x_cont = data.x_cont
        func_type = data.func_type
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else None

        x_emb = self.emb(func_type)
        x = torch.cat([x_cont, x_emb], dim=-1)

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.lin(x).squeeze(-1)
        return x


class MLPCenterBaseline(nn.Module):
    """
    仅对中心节点特征做 MLP 回归，不使用图结构。
    输入：中心节点的 x_cont + func_type embedding
    """

    def __init__(self, in_cont: int = 11, num_func_types: int = 2, emb_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.emb = nn.Embedding(num_func_types, emb_dim)
        self.in_dim = in_cont + emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_cont, func_type):
        x_emb = self.emb(func_type)
        x = torch.cat([x_cont, x_emb], dim=-1)
        return self.mlp(x).squeeze(-1)

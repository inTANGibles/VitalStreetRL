#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model/gnn.py

定义用于单图节点回归的 GNN 模型:
Embedding(func_type) -> MLP_in -> GNN 堆叠 (GCN/GraphSAGE) -> MLP_out -> y_hat(log 空间)
"""

from typing import Literal, Optional

import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv

GNNType = Literal["GCN", "SAGE"]


def _get_conv(gnn_type: str, in_dim: int, out_dim: int):
    if gnn_type.upper() == "GCN":
        return GCNConv(in_dim, out_dim)
    return SAGEConv(in_dim, out_dim)


class NodeRegressor(nn.Module):
    def __init__(
        self,
        num_func_types: int,
        in_cont_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        emb_dim: int = 8,
        dropout: float = 0.2,
        gnn_type: str = "GCN",
    ):
        super().__init__()
        self.num_func_types = num_func_types
        self.in_cont_dim = in_cont_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.gnn_type = gnn_type.upper()

        # 类别特征 Embedding
        self.emb = nn.Embedding(num_embeddings=num_func_types, embedding_dim=emb_dim)

        # 连续 + embedding 的输入投影
        self.mlp_in = nn.Sequential(
            nn.Linear(in_cont_dim + emb_dim, hidden_dim),
            nn.ReLU(),
        )

        # GNN 层堆叠（GCN 或 GraphSAGE）
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim
            self.convs.append(_get_conv(self.gnn_type, in_dim, out_dim))

        self.act = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

        # 输出层（回归，log 空间）
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x_cont: torch.Tensor,
        func_type: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_cont: [N, F_cont]
        func_type: [N] (Long)
        edge_index: [2, E]
        返回: y_hat [N]，为 log 空间的预测值
        """
        # Embedding
        emb = self.emb(func_type)  # [N, emb_dim]

        # 拼接
        x = torch.cat([x_cont, emb], dim=-1)  # [N, F_cont + emb_dim]
        x = self.mlp_in(x)

        # GNN 堆叠
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.dropout_layer(x)

        # 输出回归（log 空间）
        out = self.mlp_out(x).squeeze(-1)  # [N]
        return out
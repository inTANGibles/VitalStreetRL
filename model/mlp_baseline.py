#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model/mlp_baseline.py

MLP 基线模型：仅使用节点特征，不使用图结构（edge_index）。
用于评估图结构对预测的增益：若 MLP 与 GCN 的 MAE 相近，则连边对预测贡献有限。
"""

import torch
from torch import nn


class MLPBaseline(nn.Module):
    """仅用节点特征的 MLP 回归器，与 NodeRegressor 输入输出接口一致。"""

    def __init__(
        self,
        num_func_types: int,
        in_cont_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        emb_dim: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=num_func_types, embedding_dim=emb_dim)
        in_dim = in_cont_dim + emb_dim

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x_cont: torch.Tensor,
        func_type: torch.Tensor,
        edge_index: torch.Tensor = None,  # 忽略，保持接口兼容
    ) -> torch.Tensor:
        emb = self.emb(func_type)
        x = torch.cat([x_cont, emb], dim=-1)
        x = self.mlp(x)
        return self.mlp_out(x).squeeze(-1)

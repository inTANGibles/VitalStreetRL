#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model/evaluate.py

GCN 训练结果评估：
1. 5-fold 交叉验证：报告 MAE_mean ± std，提供统计学意义的评估
2. MLP 基线：仅用节点特征，评估图结构对预测的增益
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data

import sys
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from model.gnn import NodeRegressor
from model.mlp_baseline import MLPBaseline


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mae_loss(pred, target):
    return torch.mean(torch.abs(pred - target))


def _train_one_fold(
    data: Data,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    model_class,
    model_kwargs: dict,
    epochs: int = 500,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    device=None,
) -> float:
    """训练一个 fold，返回最佳 val_MAE。"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    train_ids = train_ids.to(device)
    val_ids = val_ids.to(device)

    model = model_class(**model_kwargs).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss(reduction="mean")

    best_val_mae = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        if model_class is MLPBaseline:
            y_hat = model(data.x_cont, data.func_type)
        else:
            y_hat = model(data.x_cont, data.func_type, data.edge_index)
        train_loss = criterion(y_hat[train_ids], data.y[train_ids])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if model_class is MLPBaseline:
                val_pred = model(data.x_cont, data.func_type)[val_ids]
            else:
                val_pred = model(data.x_cont, data.func_type, data.edge_index)[val_ids]
            val_mae = mae_loss(val_pred, data.y[val_ids]).item()
        if val_mae < best_val_mae:
            best_val_mae = val_mae
    return best_val_mae


def run_5fold_cv(
    data_path: str,
    n_folds: int = 5,
    epochs: int = 500,
    hidden_dim: int = 32,
    num_layers: int = 2,
    gnn_type: str = "GCN",
    seed: int = 42,
):
    """
    在 labeled 节点上做 n_fold 交叉验证，返回每 fold 的 val_MAE 列表及 mean ± std。
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(data_path, map_location=device, weights_only=False)
    data: Data = bundle["data"]
    num_func_types = int(bundle["num_func_types"])
    in_cont_dim = data.x_cont.size(1)

    labeled_idx = torch.nonzero(data.mask, as_tuple=False).view(-1)
    n = labeled_idx.size(0)
    if n < n_folds:
        raise ValueError(f"labeled 节点数 {n} < n_folds {n_folds}")

    # 打乱并划分 fold
    perm = labeled_idx[torch.randperm(n)]
    fold_size = n // n_folds
    val_maes = []

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else n
        val_ids = perm[start:end]
        train_ids = torch.cat([perm[:start], perm[end:]])

        mae = _train_one_fold(
            data,
            train_ids,
            val_ids,
            NodeRegressor,
            {
                "num_func_types": num_func_types,
                "in_cont_dim": in_cont_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": 0.2,
                "gnn_type": gnn_type,
            },
            epochs=epochs,
            device=device,
        )
        val_maes.append(mae)
        print(f"  Fold {fold + 1}/{n_folds} val_MAE(log)={mae:.4f}")

    mean_mae = float(np.mean(val_maes))
    std_mae = float(np.std(val_maes))
    return val_maes, mean_mae, std_mae


def run_mlp_baseline(
    data_path: str,
    epochs: int = 500,
    hidden_dim: int = 32,
    num_layers: int = 2,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> float:
    """
    训练 MLP 基线（仅节点特征，无图结构），返回最佳 val_MAE。
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(data_path, map_location=device, weights_only=False)
    data: Data = bundle["data"]
    num_func_types = int(bundle["num_func_types"])
    in_cont_dim = data.x_cont.size(1)

    labeled_idx = torch.nonzero(data.mask, as_tuple=False).view(-1)
    n = labeled_idx.size(0)
    if n < 5:
        raise ValueError(f"labeled 节点数 {n} 太少")

    perm = labeled_idx[torch.randperm(n)]
    val_size = max(int(val_ratio * n), 3)
    train_ids = perm[: n - val_size]
    val_ids = perm[n - val_size :]

    best_mae = _train_one_fold(
        data,
        train_ids,
        val_ids,
        MLPBaseline,
        {
            "num_func_types": num_func_types,
            "in_cont_dim": in_cont_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": 0.2,
        },
        epochs=epochs,
        device=device,
    )
    return best_mae


if __name__ == "__main__":
    default_data = str(PROJECT_ROOT / "data" / "FlowData" / "Jul" / "data.pt")
    print("=== 5-fold 交叉验证 (GCN) ===")
    maes, mean_mae, std_mae = run_5fold_cv(default_data, n_folds=5, epochs=500)
    print(f"GCN val_MAE(log): {mean_mae:.4f} ± {std_mae:.4f}")

    print("\n=== MLP 基线（仅节点特征）===")
    mlp_mae = run_mlp_baseline(default_data, epochs=500)
    print(f"MLP val_MAE(log): {mlp_mae:.4f}")

    print("\n=== 对比 ===")
    print(f"GCN: {mean_mae:.4f} ± {std_mae:.4f}")
    print(f"MLP: {mlp_mae:.4f}")
    if mean_mae < mlp_mae - std_mae:
        print("GCN 明显优于 MLP，图结构对预测有显著增益。")
    elif mean_mae > mlp_mae + std_mae:
        print("MLP 优于 GCN，当前图结构可能未提供有效信息。")
    else:
        print("GCN 与 MLP 差异不大，图结构增益有限。")

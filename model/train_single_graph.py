#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_single_graph.py

在单张图上进行半监督节点回归:
- 仅对有 flow 标签的节点计算 loss（masked regression）
- 从 labeled 节点中划分 80% train, 20% val
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data

# 保证能导入 model.gnn（需将项目根目录加入 path）
import sys

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from model.gnn import NodeRegressor  # noqa: E402


def parse_args():
    default_data = str(PROJECT_ROOT / "data" / "FlowData" / "Jul" / "data.pt")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=default_data, help="data.pt 路径（由 build_graph.py 生成），默认 Jul/data.pt")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gnn_type", type=str, default="GCN", choices=["GCN", "SAGE"], help="GNN 类型：GCN 或 GraphSAGE")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mae_loss(pred, target):
    return torch.mean(torch.abs(pred - target))


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = torch.load(args.data, map_location=device, weights_only=False)
    data: Data = bundle["data"]
    cont_mean = bundle["cont_mean"].to(device)
    cont_std = bundle["cont_std"].to(device)
    orig_node_ids = bundle["orig_node_ids"].to(device)
    num_func_types = int(bundle["num_func_types"])

    # 连续特征维度
    in_cont_dim = data.x_cont.size(1)

    # 模型（半监督：仅对 labeled 节点计算 loss）
    model = NodeRegressor(
        num_func_types=num_func_types,
        in_cont_dim=in_cont_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gnn_type=args.gnn_type,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.SmoothL1Loss(reduction="mean")  # Huber

    data = data.to(device)

    # labeled 节点索引
    labeled_idx = torch.nonzero(data.mask, as_tuple=False).view(-1)
    num_labeled = labeled_idx.size(0)
    if num_labeled < 5:
        raise ValueError(f"有标签节点太少: {num_labeled}，无法可靠划分 train/val。")

    # 划分 80% train, 20% val（至少 3 个 val）
    perm = labeled_idx[torch.randperm(num_labeled)]
    val_size = max(int(0.2 * num_labeled), 3)
    train_size = num_labeled - val_size
    train_ids = perm[:train_size]
    val_ids = perm[train_size:]

    print(f"总 labeled 节点: {num_labeled}, train: {train_size}, val: {val_size}")

    best_val_mae = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        y_hat = model(data.x_cont, data.func_type, data.edge_index)
        train_loss = criterion(y_hat[train_ids], data.y[train_ids])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = y_hat[val_ids]
            val_true = data.y[val_ids]
            val_mae = mae_loss(val_pred, val_true).item()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = model.state_dict()

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} "
                f"train_loss={train_loss.item():.4f} "
                f"val_MAE(log)={val_mae:.4f}"
            )

    # 保存最好模型
    if best_state is not None:
        ckpt_path = THIS_DIR / "best.pt"
        torch.save(
            {
                "model_state": best_state,
                "num_func_types": num_func_types,
                "in_cont_dim": in_cont_dim,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "gnn_type": args.gnn_type,
            },
            ckpt_path,
        )
        print(f"最佳模型已保存到: {ckpt_path}")
    print(f"最佳 val_MAE(log 空间): {best_val_mae:.4f}")


if __name__ == "__main__":
    main()
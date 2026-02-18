#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data/build_graph.py

从 nodes.csv 和 edges.csv 构建单张图的 PyG Data:
- 连续特征做 z-score 标准化
- func_type 保留为 LongTensor（给模型 Embedding）
- y 使用 log1p(flow)，无标签填 0
- mask = flow 非空
- 只保留 open == 1 的边，并补无向 (u,v),(v,u)
- 保存 data.pt，同时保存连续特征 mean/std 及原始 node_id 映射
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", required=True, help="nodes.csv 路径")
    parser.add_argument("--edges", required=True, help="edges.csv 路径")
    parser.add_argument("--out", required=True, help="输出 data.pt 路径")
    parser.add_argument(
        "--angle_unit",
        choices=["deg", "rad"],
        default="deg",
        help="main_axis_dir 角度单位（默认 deg）",
    )
    return parser.parse_args()


def check_nodes_columns(df: pd.DataFrame):
    required_cols = [
        "node_id",
        "func_type",
        "x",
        "y",
        "compactness",
        "extensibility",
        "concavity",
        "fractal_degree",
        "main_axis_dir",
        "street_length",
        "closeness",
        "betweenness",
        "flow",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"nodes.csv 缺少列: {missing}")


def check_edges_columns(df: pd.DataFrame):
    required_cols = ["u", "v", "open"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"edges.csv 缺少列: {missing}")


def remap_node_ids(nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
    """将任意 node_id 映射到 [0..N-1]，并检查边端点是否越界。"""
    unique_ids = np.sort(nodes_df["node_id"].unique())
    id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    # 映射 node 部分
    nodes_df = nodes_df.copy()
    nodes_df["node_id_new"] = nodes_df["node_id"].map(id_map)

    # 边端点检查与映射
    edges_df = edges_df.copy()
    if edges_df[["u", "v"]].isnull().any().any():
        raise ValueError("edges.csv 中存在空的 u/v。")

    unknown_u = [u for u in edges_df["u"].unique() if u not in id_map]
    unknown_v = [v for v in edges_df["v"].unique() if v not in id_map]
    if unknown_u or unknown_v:
        raise ValueError(
            f"edges.csv 中出现 node_id 不在 nodes.csv 中: u_unknown={unknown_u}, v_unknown={unknown_v}"
        )

    edges_df["u_new"] = edges_df["u"].map(id_map)
    edges_df["v_new"] = edges_df["v"].map(id_map)

    return nodes_df, edges_df, unique_ids  # unique_ids 是按新 id 顺序排列的原始 node_id


def build_data(nodes_path: Path, edges_path: Path, angle_unit: str, out_path: Path):
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    check_nodes_columns(nodes_df)
    check_edges_columns(edges_df)

    # 先重映射 node_id
    nodes_df, edges_df, orig_node_ids = remap_node_ids(nodes_df, edges_df)
    num_nodes = len(orig_node_ids)

    # 过滤 open == 1 的边
    edges_open = edges_df[edges_df["open"] == 1].copy()
    if edges_open.empty:
        raise ValueError("open == 1 的边为空，图将没有边。请检查 edges.csv 中 open 列。")

    # 无向图：为每条边加入 (u,v) 和 (v,u)，PyG 格式为 (2, num_edges)
    u = edges_open["u_new"].to_numpy(dtype=np.int64)
    v = edges_open["v_new"].to_numpy(dtype=np.int64)
    uv = np.stack([u, v], axis=0)   # (2, E)
    vu = np.stack([v, u], axis=0)   # (2, E)
    edge_pairs = np.concatenate([uv, vu], axis=1)  # (2, 2*E)
    edge_index = torch.tensor(edge_pairs, dtype=torch.long)

    # func_type 作为离散特征（embedding 输入）
    func_type = nodes_df["func_type"].fillna(0).astype(int).to_numpy()
    func_type = torch.tensor(func_type, dtype=torch.long)

    # 连续特征
    # main_axis_dir -> 弧度 -> sin, cos
    angle = nodes_df["main_axis_dir"].astype(float).to_numpy()
    if angle_unit == "deg":
        theta = np.deg2rad(angle)
    else:
        theta = angle.astype(float)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    cont_cols = [
        "x",
        "y",
        "compactness",
        "extensibility",
        "concavity",
        "fractal_degree",
        "street_length",
        "closeness",
        "betweenness",
    ]
    cont_feats = nodes_df[cont_cols].astype(float).to_numpy()
    cont_feats = np.concatenate([cont_feats, sin_theta[:, None], cos_theta[:, None]], axis=1)

    # z-score 标准化
    cont_mean = cont_feats.mean(axis=0, keepdims=True)
    cont_std = cont_feats.std(axis=0, keepdims=True)
    # 避免除 0
    cont_std_safe = np.where(cont_std < 1e-6, 1.0, cont_std)
    cont_norm = (cont_feats - cont_mean) / cont_std_safe

    x_cont = torch.tensor(cont_norm, dtype=torch.float32)

    # y: log1p(flow)，mask: flow 非空
    flow_series = nodes_df["flow"]
    mask = ~flow_series.isna()
    flow = flow_series.fillna(0.0).astype(float).to_numpy()
    y = np.log1p(flow)
    y = torch.tensor(y, dtype=torch.float32)
    mask = torch.tensor(mask.to_numpy(), dtype=torch.bool)

    # 构建 Data
    data = Data()
    data.num_nodes = int(num_nodes)
    data.edge_index = edge_index
    data.x_cont = x_cont            # 连续特征
    data.func_type = func_type      # 类别特征
    data.y = y                      # log1p(flow)
    data.mask = mask               # labeled 节点

    # 额外信息（用于推理）
    cont_mean_t = torch.tensor(cont_mean.squeeze(0), dtype=torch.float32)
    cont_std_t = torch.tensor(cont_std_safe.squeeze(0), dtype=torch.float32)
    orig_node_ids_t = torch.tensor(orig_node_ids, dtype=torch.long)
    num_func_types = int(func_type.max().item() + 1)

    out_dict = {
        "data": data,
        "cont_mean": cont_mean_t,
        "cont_std": cont_std_t,
        "orig_node_ids": orig_node_ids_t,
        "num_func_types": num_func_types,
        "angle_unit": angle_unit,
        "cont_cols": cont_cols,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_dict, out_path)
    print(f"保存 Data 到: {out_path}")
    print(f"节点数: {num_nodes}, 边数(无向): {edge_index.size(1)}")
    print(f"有标签节点数: {int(mask.sum().item())}")


def main():
    args = parse_args()
    nodes_path = Path(args.nodes)
    edges_path = Path(args.edges)
    out_path = Path(args.out)

    build_data(nodes_path, edges_path, args.angle_unit, out_path)


if __name__ == "__main__":
    main()
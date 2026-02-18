#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inference.py

加载 nodes.csv / edges.csv + data.pt 的标准化参数 + 训练好的 best.pt，
重新构建图并输出节点客流预测（原始尺度，expm1）。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

import sys

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from model.gnn import NodeRegressor  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", required=True, help="nodes.csv 路径")
    parser.add_argument("--edges", required=True, help="edges.csv 路径")
    parser.add_argument("--data", required=True, help="data.pt 路径（用于读取 mean/std 等）")
    parser.add_argument("--ckpt", required=True, help="best.pt 模型权重路径")
    parser.add_argument("--out", required=True, help="输出 pred.csv 路径")
    parser.add_argument(
        "--public_only",
        action="store_true",
        help="仅输出 public_space 节点的预测（按 unit_type 或 func_type 过滤）",
    )
    parser.add_argument(
        "--public_func_type",
        type=int,
        default=0,
        help="public_space 对应的 func_type（当 nodes 无 unit_type 时使用，默认 0）",
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
    unique_ids = np.sort(nodes_df["node_id"].unique())
    id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    nodes_df = nodes_df.copy()
    nodes_df["node_id_new"] = nodes_df["node_id"].map(id_map)

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
    return nodes_df, edges_df, unique_ids


def build_graph_for_inference(
    nodes_path: Path,
    edges_path: Path,
    bundle,
):
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    check_nodes_columns(nodes_df)
    check_edges_columns(edges_df)

    nodes_df, edges_df, orig_node_ids = remap_node_ids(nodes_df, edges_df)
    num_nodes = len(orig_node_ids)

    # 检查与 data.pt 中 orig_node_ids 一致性
    saved_orig_ids = bundle["orig_node_ids"].cpu().numpy()
    if len(saved_orig_ids) != num_nodes or not np.array_equal(
        np.sort(saved_orig_ids), np.sort(orig_node_ids)
    ):
        raise ValueError(
            "当前 nodes.csv 的 node_id 集合与 data.pt 中保存的不一致，"
            "请确保使用同一版本的 nodes/edges。"
        )

    # 构造一个从 new_id -> 在 saved_orig_ids 中的位置 的索引
    # 为简单起见，我们假设节点顺序一致（即 orig_node_ids 按 new_id 排序），
    # 前面 build_graph 中就是按升序排列的。
    # 若顺序不一致可以添加额外映射，这里保持简单且一致。
    if not np.array_equal(saved_orig_ids, orig_node_ids):
        raise ValueError(
            "data.pt 中 orig_node_ids 顺序与当前 nodes.csv 重映射顺序不一致，"
            "为安全起见中止，请保持两次构图方式完全相同。"
        )

    # 连续特征列与 build_graph 中保持一致
    cont_cols = bundle["cont_cols"]
    angle_unit = bundle["angle_unit"]
    cont_mean = bundle["cont_mean"]
    cont_std = bundle["cont_std"]

    # main_axis_dir -> 弧度 -> sin, cos
    angle = nodes_df["main_axis_dir"].astype(float).to_numpy()
    if angle_unit == "deg":
        theta = np.deg2rad(angle)
    else:
        theta = angle.astype(float)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    cont_feats = nodes_df[cont_cols].astype(float).to_numpy()
    cont_feats = np.concatenate([cont_feats, sin_theta[:, None], cos_theta[:, None]], axis=1)

    # 使用 data.pt 中的 mean/std 做标准化
    cont_mean_np = cont_mean.cpu().numpy().reshape(1, -1)
    cont_std_np = cont_std.cpu().numpy().reshape(1, -1)
    cont_std_safe = np.where(cont_std_np < 1e-6, 1.0, cont_std_np)
    cont_norm = (cont_feats - cont_mean_np) / cont_std_safe

    x_cont = torch.tensor(cont_norm, dtype=torch.float32)

    func_type = nodes_df["func_type"].fillna(0).astype(int).to_numpy()
    func_type = torch.tensor(func_type, dtype=torch.long)

    # 边：open==1，补无向
    edges_open = edges_df[edges_df["open"] == 1].copy()
    if edges_open.empty:
        raise ValueError("open == 1 的边为空，图将没有边。")

    u = edges_open["u_new"].to_numpy(dtype=np.int64)
    v = edges_open["v_new"].to_numpy(dtype=np.int64)
    uv = np.stack([u, v], axis=0)
    vu = np.stack([v, u], axis=0)
    edge_pairs = np.concatenate([uv, vu], axis=1)
    edge_index = torch.tensor(edge_pairs, dtype=torch.long)

    data = Data()
    data.num_nodes = int(num_nodes)
    data.x_cont = x_cont
    data.func_type = func_type
    data.edge_index = edge_index

    # 为了输出时对齐 node_id
    data.orig_node_ids = torch.tensor(orig_node_ids, dtype=torch.long)
    return data


def main():
    args = parse_args()
    nodes_path = Path(args.nodes)
    edges_path = Path(args.edges)
    data_path = Path(args.data)
    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = torch.load(data_path, map_location=device, weights_only=False)
    data_train: Data = bundle["data"]
    cont_mean = bundle["cont_mean"]
    cont_std = bundle["cont_std"]
    num_func_types = int(bundle["num_func_types"])
    in_cont_dim = int(data_train.x_cont.size(1))

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hidden_dim = ckpt.get("hidden_dim", 32)
    num_layers = ckpt.get("num_layers", 2)
    dropout = ckpt.get("dropout", 0.2)
    gnn_type = ckpt.get("gnn_type", "GCN")

    model = NodeRegressor(
        num_func_types=num_func_types,
        in_cont_dim=in_cont_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        gnn_type=gnn_type,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 重新构图（使用 data.pt 的 mean/std 等信息）
    bundle_for_infer = {
        "cont_mean": cont_mean,
        "cont_std": cont_std,
        "orig_node_ids": bundle["orig_node_ids"],
        "cont_cols": bundle["cont_cols"],
        "angle_unit": bundle["angle_unit"],
    }
    data = build_graph_for_inference(nodes_path, edges_path, bundle_for_infer)
    data = data.to(device)

    with torch.no_grad():
        y_log = model(data.x_cont, data.func_type, data.edge_index)  # [N]
        # 反 log1p
        y_pred = torch.expm1(y_log).cpu().numpy()

    # 输出 pred.csv: node_id, pred_flow
    node_ids = data.orig_node_ids.cpu().numpy()
    pred_df = pd.DataFrame({"node_id": node_ids, "pred_flow": y_pred})

    # 仅 public_space：按 unit_type 或 func_type 过滤
    if args.public_only:
        nodes_df = pd.read_csv(nodes_path)
        if "unit_type" in nodes_df.columns:
            public_mask = nodes_df["unit_type"] == "public_space"
        else:
            public_mask = nodes_df["func_type"] == args.public_func_type
        public_node_ids = set(nodes_df.loc[public_mask, "node_id"].astype(int))
        pred_df = pred_df[pred_df["node_id"].isin(public_node_ids)].copy()
        print(f"仅输出 public_space 节点: {len(pred_df)} 个")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False)
    print(f"预测结果已保存到: {out_path}")


if __name__ == "__main__":
    main()
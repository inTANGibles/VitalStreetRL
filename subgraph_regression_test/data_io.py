"""
数据加载模块：从 CSV 或 graph_cache.pt 加载整图。
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_graph_from_csv(
    nodes_csv: str,
    edges_csv: str,
    flow_csv: str,
    pt_path: str = None,
) -> dict:
    """
    从 nodes.csv, edges.csv, flow.csv 加载图数据并生成 graph dict。
    返回 dict: x_cont, func_type, y, mask, edge_index, labeled_node_ids
    若指定 pt_path，则保存到 .pt 文件。
    """
    nodes = pd.read_csv(nodes_csv).sort_values("node_id").reset_index(drop=True)
    edges = pd.read_csv(edges_csv)
    flow = pd.read_csv(flow_csv)

    num_nodes = len(nodes)

    # func_type [N]
    func_type = torch.tensor(nodes["func_type"].values, dtype=torch.long)

    # x_cont [N, 11]: compactness, extensibility, concavity, fractal_degree, main_axis_dir,
    #                 street_length, closeness, betweenness, x, y + pad
    cont_cols = [
        "compactness", "extensibility", "concavity", "fractal_degree",
        "main_axis_dir", "street_length", "closeness", "betweenness", "x", "y",
    ]
    x_cont = nodes[cont_cols].values.astype(np.float32)
    x_cont = np.nan_to_num(x_cont, nan=0.0)
    x_cont = (x_cont - x_cont.mean(axis=0)) / (x_cont.std(axis=0) + 1e-8)
    x_cont = np.hstack([x_cont, np.zeros((num_nodes, 1))])  # pad to 11
    x_cont = torch.tensor(x_cont, dtype=torch.float32)

    # y, mask, labeled_node_ids from flow.csv
    flow_labeled = flow.dropna(subset=["flow_raw"])
    labeled_node_ids = flow_labeled["node_id"].values.astype(int)
    y = torch.zeros(num_nodes, dtype=torch.float32)
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    for _, row in flow_labeled.iterrows():
        idx = int(row["node_id"])
        flow_raw = float(row["flow_raw"])
        y[idx] = np.log1p(flow_raw)
        mask[idx] = True
    labeled_node_ids = torch.tensor(labeled_node_ids, dtype=torch.long)

    # edge_index [2, 2*E] undirected
    edge_list = []
    for _, row in edges.iterrows():
        u, v = int(row["u"]), int(row["v"])
        edge_list.append([u, v])
        edge_list.append([v, u])
    edge_index = torch.tensor(edge_list, dtype=torch.long).T.contiguous()

    out = {
        "x_cont": x_cont,
        "func_type": func_type,
        "y": y,
        "mask": mask,
        "edge_index": edge_index,
        "labeled_node_ids": labeled_node_ids,
    }
    if pt_path:
        Path(pt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, pt_path)
    return out


def load_graph_from_pt(pt_path: str) -> dict:
    """
    从 graph_cache.pt 加载图数据。
    """
    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    return {
        "x_cont": data["x_cont"],
        "func_type": data["func_type"],
        "y": data["y"],
        "mask": data["mask"],
        "edge_index": data["edge_index"],
        "labeled_node_ids": data["labeled_node_ids"],
    }


def generate_mock_cache(pt_path: str, num_nodes: int = 101, num_edges: int = 340) -> dict:
    """
    生成 mock 数据并保存到 graph_cache.pt。
    - 101 节点，340 条无向边（edge_index 存 680 条）
    - 20 个 labeled 节点
    """
    torch.manual_seed(42)
    x_cont = torch.randn(num_nodes, 11).float()
    x_cont = (x_cont - x_cont.mean(dim=0)) / (x_cont.std(dim=0) + 1e-8)
    func_type = torch.randint(0, 2, (num_nodes,)).long()
    y = torch.zeros(num_nodes).float()
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    labeled_node_ids = torch.randperm(num_nodes)[:20]
    mask[labeled_node_ids] = True
    y[mask] = torch.log1p(torch.rand(20).float() * 10 + 0.1)
    y[~mask] = 0.0

    # 随机生成无向边（先连成链保证连通，再随机加边）
    seen = set()
    edge_list = []
    for i in range(num_nodes - 1):
        a, b = i, i + 1
        seen.add((min(a, b), max(a, b)))
        edge_list.extend([[a, b], [b, a]])
    remaining = num_edges - (num_nodes - 1)
    while len(seen) < num_edges:
        u, v = torch.randint(0, num_nodes, (2,)).tolist()
        if u != v:
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                edge_list.extend([[u, v], [v, u]])
    edge_index = torch.tensor(edge_list, dtype=torch.long).T

    out = {
        "x_cont": x_cont,
        "func_type": func_type,
        "y": y,
        "mask": mask,
        "edge_index": edge_index,
        "labeled_node_ids": labeled_node_ids,
    }
    Path(pt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, pt_path)
    return out


def load_graph(
    nodes_csv: str = None,
    edges_csv: str = None,
    flow_csv: str = None,
    pt_path: str = "graph_cache.pt",
) -> dict:
    """
    主入口：优先从 CSV 加载（需 nodes, edges, flow 三个），否则从 pt 加载；pt 不存在则生成 mock 并保存。
    """
    if (
        nodes_csv
        and edges_csv
        and flow_csv
        and os.path.exists(nodes_csv)
        and os.path.exists(edges_csv)
        and os.path.exists(flow_csv)
    ):
        return load_graph_from_csv(nodes_csv, edges_csv, flow_csv, pt_path=pt_path)
    if os.path.exists(pt_path):
        return load_graph_from_pt(pt_path)
    return generate_mock_cache(pt_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build graph.pt from CSV files")
    parser.add_argument("--nodes", required=True, help="nodes.csv path")
    parser.add_argument("--edges", required=True, help="edges.csv path")
    parser.add_argument("--flow", required=True, help="flow.csv path")
    parser.add_argument("--output", default="graph_cache.pt", help="Output .pt path")
    args = parser.parse_args()
    load_graph_from_csv(args.nodes, args.edges, args.flow, pt_path=args.output)
    print(f"Saved graph to {args.output}")

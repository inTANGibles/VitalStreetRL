#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
update_edges_open.py

根据 nodes.csv 的节点类型更新 edges.csv 的 open 列：
- open=1: shop-public_space, public_space-shop, public_space-public_space
- open=0: shop-shop

约定：nodes 顺序与 build_nodes_edges 一致，shop 在前、public_space 在后。
func_type 编码由 build_nodes_edges 的 func_field 决定，此处通过 unit_type 推断：
若 nodes 来自 street.geojson，则需传入 --shop_count 或依赖 nodes 中 func_type 的众数分布。
为简化，默认：前 n_shop 个节点为 shop（n_shop = func_type==1 的数量，若 1 较多则 1=shop）。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="根据节点类型更新 edges.csv 的 open 列")
    parser.add_argument("--nodes", required=True, help="nodes.csv 路径")
    parser.add_argument("--edges", required=True, help="edges.csv 路径")
    parser.add_argument(
        "--shop_func_type",
        type=int,
        default=1,
        help="shop 对应的 func_type 值（默认 1）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    nodes_path = Path(args.nodes)
    edges_path = Path(args.edges)

    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    if "func_type" not in nodes_df.columns:
        raise ValueError("nodes.csv 需包含 func_type 列")
    if "open" not in edges_df.columns:
        edges_df["open"] = 1

    # node_id -> is_shop (1=shop, 0=public)
    node_id_to_func = nodes_df.set_index("node_id")["func_type"].to_dict()
    shop_func = args.shop_func_type

    u_ids = edges_df["u"].to_numpy()
    v_ids = edges_df["v"].to_numpy()
    u_shop = np.array([node_id_to_func.get(u, 0) == shop_func for u in u_ids])
    v_shop = np.array([node_id_to_func.get(v, 0) == shop_func for v in v_ids])
    shop_shop = u_shop & v_shop
    edges_df["open"] = np.where(shop_shop, 0, 1)

    edges_df.to_csv(edges_path, index=False)
    n_open = int(edges_df["open"].sum())
    print(f"已更新 {edges_path}")
    print(f"  边数: {len(edges_df)}, open=1: {n_open}, open=0: {len(edges_df) - n_open}")


if __name__ == "__main__":
    main()

"""
2-hop 异构子图抽取：以有标签节点为中心，提取 2-hop 邻域。
"""
from typing import Dict, Set, Tuple

import torch
from torch_geometric.data import HeteroData

EDGE_TYPES = [
    ("shop", "adj", "shop"),
    ("shop", "adj", "public"),
    ("public", "adj", "shop"),
    ("public", "adj", "public"),
]


def _get_neighbors(
    data: HeteroData,
    center_type: str,
    center_idx: int,
) -> Tuple[Set[int], Set[int]]:
    """从 center (center_type, center_idx) 出发，返回 1-hop 和 2-hop 邻居。"""
    shop_1hop: Set[int] = set()
    public_1hop: Set[int] = set()
    shop_2hop: Set[int] = set()
    public_2hop: Set[int] = set()

    # 1-hop: 与 center 相连的节点
    if center_type == "public":
        for (st, _, dt) in EDGE_TYPES:
            key = (st, "adj", dt)
            if key not in data.edge_types or not hasattr(data[key], "edge_index"):
                continue
            ei = data[key].edge_index
            if ei is None or ei.numel() == 0:
                continue
            for j in range(ei.size(1)):
                s, d = int(ei[0, j].item()), int(ei[1, j].item())
                if st == "public" and s == center_idx:
                    if dt == "shop":
                        shop_1hop.add(d)
                    else:
                        public_1hop.add(d)
                elif dt == "public" and d == center_idx:
                    if st == "shop":
                        shop_1hop.add(s)
                    else:
                        public_1hop.add(s)
    else:
        # center_type == "shop"
        for (st, _, dt) in EDGE_TYPES:
            key = (st, "adj", dt)
            if key not in data.edge_types or not hasattr(data[key], "edge_index"):
                continue
            ei = data[key].edge_index
            if ei is None or ei.numel() == 0:
                continue
            for j in range(ei.size(1)):
                s, d = int(ei[0, j].item()), int(ei[1, j].item())
                if st == "shop" and s == center_idx:
                    if dt == "shop":
                        shop_1hop.add(d)
                    else:
                        public_1hop.add(d)
                elif dt == "shop" and d == center_idx:
                    if st == "shop":
                        shop_1hop.add(s)
                    else:
                        public_1hop.add(s)

    # 2-hop: 与 1-hop 相连的节点
    def collect_2hop(nodes: Set[int], ntype: str):
        for (st, _, dt) in EDGE_TYPES:
            key = (st, "adj", dt)
            if key not in data.edge_types or not hasattr(data[key], "edge_index"):
                continue
            ei = data[key].edge_index
            if ei is None or ei.numel() == 0:
                continue
            for j in range(ei.size(1)):
                s, d = int(ei[0, j].item()), int(ei[1, j].item())
                if st == ntype and s in nodes:
                    if dt == "shop":
                        shop_2hop.add(d)
                    else:
                        public_2hop.add(d)
                elif dt == ntype and d in nodes:
                    if st == "shop":
                        shop_2hop.add(s)
                    else:
                        public_2hop.add(s)

    collect_2hop(shop_1hop, "shop")
    collect_2hop(public_1hop, "public")

    return (shop_1hop | shop_2hop, public_1hop | public_2hop)


def extract_2hop_subgraph(
    data: HeteroData,
    center_type: str,
    center_idx: int,
) -> Tuple[HeteroData, int]:
    """
    以 (center_type, center_idx) 为中心抽取 2-hop 异构子图。
    返回 (subgraph, center_idx_in_subgraph)。
    center_idx_in_subgraph 为 center 在新 public 节点中的局部索引（当 center_type 为 public 时）。
    """
    shop_keep, public_keep = _get_neighbors(data, center_type, center_idx)
    if center_type == "public":
        public_keep.add(center_idx)
    else:
        shop_keep.add(center_idx)

    shop_keep = sorted(shop_keep)
    public_keep = sorted(public_keep)
    old_to_new_shop = {o: i for i, o in enumerate(shop_keep)}
    old_to_new_public = {o: i for i, o in enumerate(public_keep)}

    sub = HeteroData()
    sub["shop"].x = data["shop"].x[shop_keep]
    sub["public"].x = data["public"].x[public_keep]
    sub["shop"].num_nodes = len(shop_keep)
    sub["public"].num_nodes = len(public_keep)

    # Copy node_id for all nodes so viz can use gdf positions (full graph order = gdf row index).
    if hasattr(data["shop"], "node_id"):
        sub["shop"].node_id = data["shop"].node_id[shop_keep]
    if hasattr(data["public"], "node_id"):
        sub["public"].node_id = data["public"].node_id[public_keep]

    center_new = -1
    if center_type == "public":
        center_new = old_to_new_public[center_idx]
        sub["public"].y = data["public"].y[center_idx : center_idx + 1]
        sub["public"].mask = data["public"].mask[center_idx : center_idx + 1]
    else:
        pass  # shop center already handled above

    for (st, _, dt) in EDGE_TYPES:
        key = (st, "adj", dt)
        if key not in data.edge_types or not hasattr(data[key], "edge_index"):
            continue
        ei = data[key].edge_index
        if ei is None or ei.numel() == 0:
            continue
        old_src, old_dst = ei[0].tolist(), ei[1].tolist()
        old_to_new_src = old_to_new_shop if st == "shop" else old_to_new_public
        old_to_new_dst = old_to_new_shop if dt == "shop" else old_to_new_public
        new_edges = []
        idx_keep = []
        for i in range(len(old_src)):
            s, d = old_src[i], old_dst[i]
            if s in old_to_new_src and d in old_to_new_dst:
                new_edges.append([old_to_new_src[s], old_to_new_dst[d]])
                idx_keep.append(i)
        if new_edges:
            sub[key].edge_index = torch.tensor(new_edges, dtype=torch.long).t()
            if hasattr(data[key], "edge_attr") and data[key].edge_attr is not None and idx_keep:
                sub[key].edge_attr = data[key].edge_attr[idx_keep]

    sub.center_idx = center_new
    sub.center_type = center_type
    sub.shop_keep = shop_keep  # 原图 shop 索引 -> node_id 通过 data["shop"].node_id
    sub.public_keep = public_keep  # 原图 public 索引 -> node_id 通过 data["public"].node_id
    return sub, center_new

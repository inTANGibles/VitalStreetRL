"""
2-hop 同构子图抽取：以指定节点（全局索引）为中心，提取 2-hop 邻域。
"""
from typing import Set, Tuple

import torch
from torch_geometric.data import Data


def _get_2hop_neighbors(data: Data, center: int) -> Set[int]:
    """从 center 出发，返回 1-hop 和 2-hop 邻居（含 center）。"""
    ei = data.edge_index
    if ei is None or ei.numel() == 0:
        return {center}
    one_hop: Set[int] = {center}
    for j in range(ei.size(1)):
        s, d = int(ei[0, j].item()), int(ei[1, j].item())
        if s == center:
            one_hop.add(d)
        elif d == center:
            one_hop.add(s)
    two_hop = set(one_hop)
    for j in range(ei.size(1)):
        s, d = int(ei[0, j].item()), int(ei[1, j].item())
        if s in one_hop or d in one_hop:
            two_hop.add(s)
            two_hop.add(d)
    return two_hop


def extract_2hop_subgraph(data: Data, center_global: int) -> Tuple[Data, int]:
    """
    以全局节点索引 center_global 为中心抽取 2-hop 子图。
    返回 (subgraph, center_idx_in_subgraph)。
    若 center 为 public 节点，sub.y / sub.mask 为该中心在当时间片的标签。
    """
    keep = sorted(_get_2hop_neighbors(data, center_global))
    old_to_new = {o: i for i, o in enumerate(keep)}
    center_local = old_to_new[center_global]

    sub = Data(
        x=data.x[keep],
        edge_index=torch.empty(2, 0, dtype=torch.long),
    )
    ei = data.edge_index
    if ei is not None and ei.numel() > 0:
        src, dst = ei[0].tolist(), ei[1].tolist()
        new_edges = []
        for i in range(len(src)):
            s, d = src[i], dst[i]
            if s in old_to_new and d in old_to_new:
                new_edges.append([old_to_new[s], old_to_new[d]])
        if new_edges:
            sub.edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()

    sub.num_nodes = len(keep)
    sub.center_idx = center_local
    if hasattr(data, "node_id"):
        sub.node_id = data.node_id[keep]

    n_shop = getattr(data, "num_shop", 0)
    if center_global >= n_shop and hasattr(data, "y") and hasattr(data, "mask"):
        public_idx = center_global - n_shop
        sub.y = data.y[public_idx : public_idx + 1]
        sub.mask = data.mask[public_idx : public_idx + 1]
    else:
        sub.y = torch.zeros(1, 1, dtype=torch.float32)
        sub.mask = torch.zeros(1, dtype=torch.bool)

    return sub, center_local

"""
1-hop / 2-hop 同构子图抽取：以指定节点（全局索引）为中心，提取邻域。
"""
from typing import List, Set, Tuple

import torch
from torch_geometric.data import Data


def _get_1hop_neighbors(data: Data, center: int) -> Set[int]:
    """从 center 出发，返回 center 与 1-hop 邻居（含 center）。"""
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
    return one_hop


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


def _extract_subgraph_from_keep(data: Data, keep: list, center_global: int) -> Tuple[Data, int]:
    """给定节点集合 keep（全局索引排序列表）和中心 center_global，构建子图。供 1-hop/2-hop 共用。"""
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
        new_attr = [] if hasattr(data, "edge_attr") and data.edge_attr is not None else None
        for i in range(len(src)):
            s, d = src[i], dst[i]
            if s in old_to_new and d in old_to_new:
                new_edges.append([old_to_new[s], old_to_new[d]])
                if new_attr is not None:
                    ea = data.edge_attr[i]
                    if ea.numel() == 1:
                        new_attr.append([ea.item()])
                    else:
                        new_attr.append(ea.tolist())
        if new_edges:
            sub.edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
            if new_attr is not None and len(new_attr) > 0:
                sub.edge_attr = torch.tensor(new_attr, dtype=torch.float32)
                if sub.edge_attr.dim() == 1:
                    sub.edge_attr = sub.edge_attr.unsqueeze(1)
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


def extract_1hop_subgraph(data: Data, center_global: int) -> Tuple[Data, int]:
    """
    以全局节点索引 center_global 为中心抽取 1-hop 子图（仅 center 及其直接邻居）。
    返回 (subgraph, center_idx_in_subgraph)。
    """
    keep = sorted(_get_1hop_neighbors(data, center_global))
    return _extract_subgraph_from_keep(data, keep, center_global)


def extract_2hop_subgraph(data: Data, center_global: int) -> Tuple[Data, int]:
    """
    以全局节点索引 center_global 为中心抽取 2-hop 子图。
    返回 (subgraph, center_idx_in_subgraph)。
    若 center 为 public 节点，sub.y / sub.mask 为该中心在当时间片的标签。
    """
    keep = sorted(_get_2hop_neighbors(data, center_global))
    return _extract_subgraph_from_keep(data, keep, center_global)


def build_1hop_subgraphs(data: Data, center_public_indices: List[int], num_shop: int = None) -> List[Tuple[Data, int]]:
    """
    对每个 public 中心索引（在 public 节点中的下标）抽取 1-hop 子图。
    返回 [(sub, center_idx), ...]，与 extract_1hop_subgraph 的 center_idx 一致（public 下标）。
    """
    n_shop = num_shop if num_shop is not None else getattr(data, "num_shop", 0)
    out = []
    for center_idx in center_public_indices:
        center_global = n_shop + center_idx
        sub, _ = extract_1hop_subgraph(data, center_global)
        out.append((sub, center_idx))
    return out


def build_2hop_subgraphs(data: Data, center_public_indices: List[int], num_shop: int = None) -> List[Tuple[Data, int]]:
    """
    对每个 public 中心索引抽取 2-hop 子图。
    返回 [(sub, center_idx), ...]。
    """
    n_shop = num_shop if num_shop is not None else getattr(data, "num_shop", 0)
    out = []
    for center_idx in center_public_indices:
        center_global = n_shop + center_idx
        sub, _ = extract_2hop_subgraph(data, center_global)
        out.append((sub, center_idx))
    return out

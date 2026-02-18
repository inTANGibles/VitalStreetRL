"""
子图构建模块：BFS k-hop 子图抽取，构建 PyG Data 列表。
"""
from collections import deque

import torch
from torch_geometric.data import Data


def build_khop_subgraph(edge_index: torch.Tensor, center: int, k: int):
    """
    BFS 抽取以 center 为中心的 k-hop 诱导子图。
    返回：
      sub_edge_index: [2, E_sub] 子图内边的局部编号
      sub_nodes: 子图内节点的全局 id 列表（顺序即局部编号）
      center_local_index: 中心节点在 sub_nodes 中的局部索引
    """
    num_nodes = edge_index.max().item() + 1
    adj = [[] for _ in range(num_nodes)]
    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        adj[u].append(v)

    visited = {center: 0}
    q = deque([center])
    while q:
        u = q.popleft()
        hop = visited[u]
        if hop >= k:
            continue
        for v in adj[u]:
            if v not in visited:
                visited[v] = hop + 1
                q.append(v)

    sub_nodes = sorted(visited.keys())
    global_to_local = {g: i for i, g in enumerate(sub_nodes)}
    center_local_index = global_to_local[center]

    sub_edges = []
    for u in sub_nodes:
        for v in adj[u]:
            if v in global_to_local:
                sub_edges.append([global_to_local[u], global_to_local[v]])
    if not sub_edges:
        sub_edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        sub_edge_index = torch.tensor(sub_edges, dtype=torch.long).T.contiguous()

    return sub_edge_index, sub_nodes, center_local_index


def build_dataset(graph: dict, k: int = 2):
    """
    以每个节点为中心抽取 k-hop ego-subgraph，构建 101 个子图样本。
    每个 Data 包含：
      x_cont, func_type, edge_index（局部编号）
      y_center: 中心节点的回归目标（有标签为 log1p(flow)，无标签为 0，仅用于有标签样本的 loss）
      center_idx: 中心节点在子图中的局部索引
      center_global_id: 中心节点的全局 id
      has_label: 该中心是否有客流观测（用于 masked supervision）
    """
    x_cont = graph["x_cont"]
    func_type = graph["func_type"]
    y = graph["y"]
    mask = graph["mask"]
    edge_index = graph["edge_index"]
    num_nodes = x_cont.shape[0]

    dataset = []
    for c in range(num_nodes):
        sub_edge_index, sub_nodes, center_local_index = build_khop_subgraph(edge_index, c, k)
        x_sub = x_cont[sub_nodes]
        func_sub = func_type[sub_nodes]
        has_label = mask[c].item()
        y_center = y[c].item() if has_label else 0.0

        data = Data(
            x_cont=x_sub,
            func_type=func_sub,
            edge_index=sub_edge_index,
            y_center=torch.tensor(y_center, dtype=torch.float),
            center_idx=torch.tensor(center_local_index, dtype=torch.long),
            center_global_id=torch.tensor(c, dtype=torch.long),
            has_label=torch.tensor(has_label, dtype=torch.bool),
            num_nodes=len(sub_nodes),
        )
        dataset.append(data)
    return dataset

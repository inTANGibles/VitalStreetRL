"""
构建 42 个时间片 PyG Data 样本。
"""
import math

import torch
from torch_geometric.data import Data


def build_time_feat(day_idx: int, hour: int, use_day_of_week: bool = True) -> torch.Tensor:
    """
    构造时间特征向量。
    - hour: 0-23
    - day_idx: 0-6
    返回 [time_dim] 的 float tensor。
    """
    feat_list = []
    # hour sin/cos
    feat_list.append(math.sin(2 * math.pi * hour / 24))
    feat_list.append(math.cos(2 * math.pi * hour / 24))
    if use_day_of_week:
        feat_list.append(math.sin(2 * math.pi * day_idx / 7))
        feat_list.append(math.cos(2 * math.pi * day_idx / 7))
    return torch.tensor(feat_list, dtype=torch.float32)


def build_42_graph_views(
    graph_static: dict,
    y_time: torch.Tensor,
    mask_time: torch.Tensor,
    use_day_of_week: bool = True,
) -> list[Data]:
    """
    从静态图与时间片标签构建 42 个 PyG Data 样本。

    graph_static: dict with keys x_cont [101,11], func_type [101], edge_index [2,680]
    y_time: [42, 101] float, log1p(flow)
    mask_time: [42, 101] bool

    返回 list[Data]，每个 Data 包含：
      x_cont, func_type, time_feat, edge_index, y, mask, meta
    """
    x_cont = graph_static["x_cont"]
    func_type = graph_static["func_type"]
    edge_index = graph_static["edge_index"]
    num_nodes = x_cont.shape[0]

    time_dim = 4 if use_day_of_week else 2
    data_list = []

    for t in range(42):
        day_idx = t // 6
        hour_offset = t % 6
        hour = 12 + hour_offset

        time_vec = build_time_feat(day_idx, hour, use_day_of_week)
        time_feat = time_vec.unsqueeze(0).expand(num_nodes, time_dim)

        data = Data(
            x_cont=x_cont.clone(),
            func_type=func_type.clone(),
            time_feat=time_feat.clone(),
            edge_index=edge_index.clone(),
            y=y_time[t].clone(),
            mask=mask_time[t].clone(),
        )
        data.meta = {"day_idx": day_idx, "hour": hour}
        data_list.append(data)

    return data_list

"""
TimeSliceDataset: one PyG Data sample per time slice.
SubgraphTimeSliceDataset: 220 time slices × labeled public nodes, 1-hop 或 2-hop 同构子图 each.
Train/val split by time slice index (80/20).
"""
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import Dataset

from geo.tool.build_graph import build_graph_data, load_normalizer, save_normalizer
from data.subgraph_utils import (
    extract_1hop_subgraph,
    extract_2hop_subgraph,
    extract_subgraph_with_shop_pseudo,
)


def _compute_flow_quantiles(flows_path: Path, use_slot: bool) -> List[float]:
    """从 flows.csv 收集所有 flow，计算 10/20/.../90 分位点，用于 remap_1_10。"""
    import csv
    all_flows = []
    with open(flows_path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            all_flows.append(float(r["flow"]))
    if not all_flows:
        return [0.0] * 9
    return np.percentile(all_flows, [10, 20, 30, 40, 50, 60, 70, 80, 90]).tolist()


def _compute_flow_range(flows_path: Path) -> Tuple[float, float]:
    """从 flows.csv 收集所有非零 flow，计算 min/max，用于 linear_1_10 线性映射 [1,10]。"""
    import csv
    all_flows = []
    with open(flows_path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            v = float(r["flow"])
            if v > 0:
                all_flows.append(v)
    if not all_flows:
        return 0.0, 1.0
    return float(np.min(all_flows)), float(np.max(all_flows))


class TimeSliceDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        time_slices: list,
        use_slot: bool = False,
        normalizer: Optional[dict] = None,
        normalizer_path: Optional[Path] = None,
        label_transform: str = "log1p",
    ):
        """
        data_dir: directory containing nodes.csv, edges.csv, flows.csv.
        time_slices: list of (day, hour) or (day, slot_idx) when use_slot=True.
        normalizer: if None and normalizer_path exists, load; else computed from first slice.
        """
        self.data_dir = Path(data_dir)
        self.time_slices = time_slices
        self.use_slot = use_slot
        self.label_transform = label_transform
        nodes_path = self.data_dir / "nodes.csv"
        edges_path = self.data_dir / "edges.csv"
        flows_path = self.data_dir / "flows.csv"
        if not flows_path.exists():
            flows_path = None

        if normalizer is not None:
            self.normalizer = normalizer
        elif normalizer_path and normalizer_path.exists():
            self.normalizer = load_normalizer(normalizer_path)
        else:
            self.normalizer = None

        day0, slot0 = time_slices[0]
        data0, norm0 = build_graph_data(
            nodes_path, edges_path, flows_path,
            day=day0, hour=slot0 if not use_slot else 12,
            slot_idx=slot0 if use_slot else None,
            use_slot=use_slot,
            normalizer=self.normalizer,
            label_transform=label_transform,
        )
        if self.normalizer is None and norm0 is not None:
            self.normalizer = norm0

        self._samples = [(data0, day0, slot0)]
        for (day, slot_or_hour) in time_slices[1:]:
            data, _ = build_graph_data(
                nodes_path, edges_path, flows_path,
                day=day, hour=slot_or_hour if not use_slot else 12,
                slot_idx=slot_or_hour if use_slot else None,
                use_slot=use_slot,
                normalizer=self.normalizer,
                label_transform=label_transform,
            )
            self._samples.append((data, day, slot_or_hour))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx][0]

    def get_time_slice(self, idx):
        return self._samples[idx][1], self._samples[idx][2]


def get_train_val_time_slices(time_slices: list, val_ratio: float = 0.2, seed: int = 42):
    import random
    rng = random.Random(seed)
    indices = list(range(len(time_slices)))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_ratio))
    val_idx = set(indices[:n_val])
    train_slices = [time_slices[i] for i in indices[n_val:]]
    val_slices = [time_slices[i] for i in indices[:n_val]]
    return train_slices, val_slices


class SubgraphTimeSliceDataset(Dataset):
    """
    4400 样本：220 时间片 × 20 有标签节点。
    每个样本 = 以有标签节点为中心的 1-hop 或 2-hop 异构子图 + 中心节点 1–10 客流等级标签。
    num_hops=1 为 1-hop，num_hops=2 为 2-hop（默认）。
    """

    def __init__(
        self,
        data_dir: Path,
        time_slices: List[Tuple[int, int]],
        use_slot: bool = True,
        normalizer: Optional[dict] = None,
        normalizer_path: Optional[Path] = None,
        label_transform: str = "remap_1_10",
        num_hops: int = 2,
    ):
        """
        data_dir: 含 nodes.csv, edges.csv, flows.csv 的目录。
        time_slices: [(day, slot_idx), ...]，共 220 个。
        use_slot: True 时使用 (day, slot_idx) 作为 flow 键。
        num_hops: 1 或 2，子图邻域跳数。
        """
        self.data_dir = Path(data_dir)
        self.time_slices = time_slices
        self.use_slot = use_slot
        self.label_transform = label_transform
        self.num_hops = num_hops
        nodes_path = self.data_dir / "nodes.csv"
        edges_path = self.data_dir / "edges.csv"
        flows_path = self.data_dir / "flows.csv"
        if not flows_path.exists():
            flows_path = None

        if normalizer is not None:
            self.normalizer = normalizer
        elif normalizer_path and normalizer_path.exists():
            self.normalizer = load_normalizer(normalizer_path)
        else:
            self.normalizer = None

        self.flow_quantiles = None
        self.flow_min, self.flow_max = None, None
        if label_transform == "remap_1_10" and flows_path and Path(flows_path).exists():
            self.flow_quantiles = _compute_flow_quantiles(Path(flows_path), use_slot)
        if label_transform == "linear_1_10" and flows_path and Path(flows_path).exists():
            self.flow_min, self.flow_max = _compute_flow_range(Path(flows_path))

        # 从 flows.csv 或首片获取有标签的 public 节点索引
        import pandas as pd
        labeled_public_indices: List[int] = []
        if flows_path and Path(flows_path).exists():
            flows_df = pd.read_csv(flows_path)
            nodes_df = pd.read_csv(nodes_path)
            public_ids = nodes_df[nodes_df["node_type"] == "public_space"]["node_id"].tolist()
            flow_node_ids = set(flows_df["node_id"].unique())
            labeled_public_indices = [
                i for i, nid in enumerate(public_ids) if nid in flow_node_ids
            ]
        if not labeled_public_indices:
            d0, s0 = time_slices[0][0], time_slices[0][1]
            data0, _ = build_graph_data(
                nodes_path, edges_path, flows_path,
                day=d0, hour=s0 if not use_slot else 12,
                slot_idx=s0 if use_slot else None,
                use_slot=use_slot,
                normalizer=None,
                label_transform=label_transform,
                flow_quantiles=self.flow_quantiles,
                flow_min=self.flow_min,
                flow_max=self.flow_max,
            )
            labeled_public_indices = [
                j for j in range(data0.mask.size(0)) if data0.mask[j].item()
            ]
            if not labeled_public_indices:
                labeled_public_indices = list(range(data0.num_public))

        self._samples: List[Tuple] = []
        for ts in time_slices:
            day, slot_or_hour = ts[0], ts[1]
            data, norm = build_graph_data(
                nodes_path, edges_path, flows_path,
                day=day,
                hour=slot_or_hour if not use_slot else 12,
                slot_idx=slot_or_hour if use_slot else None,
                use_slot=use_slot,
                normalizer=self.normalizer,
                label_transform=label_transform,
                flow_quantiles=self.flow_quantiles,
                flow_min=self.flow_min,
                flow_max=self.flow_max,
            )
            if self.normalizer is None and norm is not None:
                self.normalizer = norm

            extract_fn = extract_1hop_subgraph if self.num_hops == 1 else extract_2hop_subgraph
            for center_public_idx in labeled_public_indices:
                center_global = data.num_shop + center_public_idx
                sub, _ = extract_fn(data, center_global)
                self._samples.append((sub, day, slot_or_hour, center_public_idx))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx][0]

    def get_meta(self, idx):
        return self._samples[idx][1], self._samples[idx][2], self._samples[idx][3]


class FullSubgraphTimeSliceDataset(Dataset):
    """
    105+38 子图模式：每个时间片为 105 个 shop + 38 个 public 中心，共 143 个子图。
    - Public 子图：真实标签（flows.csv）
    - Shop 子图：伪标签（1-hop 邻居中 public 节点的平均 flow），用于半监督训练。
    """

    def __init__(
        self,
        data_dir: Path,
        time_slices: List[Tuple[int, int]],
        use_slot: bool = True,
        normalizer: Optional[dict] = None,
        normalizer_path: Optional[Path] = None,
        label_transform: str = "remap_1_10",
        num_hops: int = 1,
        use_shop_pseudo: bool = True,
    ):
        """
        use_shop_pseudo: 若 True，shop 中心用邻居 public 平均 flow 作伪标签；否则 mask=False。
        """
        self.data_dir = Path(data_dir)
        self.time_slices = time_slices
        self.use_slot = use_slot
        self.label_transform = label_transform
        self.num_hops = num_hops
        self.use_shop_pseudo = use_shop_pseudo
        nodes_path = self.data_dir / "nodes.csv"
        edges_path = self.data_dir / "edges.csv"
        flows_path = self.data_dir / "flows.csv"
        if not flows_path.exists():
            flows_path = None

        if normalizer is not None:
            self.normalizer = normalizer
        elif normalizer_path and normalizer_path.exists():
            self.normalizer = load_normalizer(normalizer_path)
        else:
            self.normalizer = None

        self.flow_quantiles = None
        self.flow_min, self.flow_max = None, None
        if label_transform == "remap_1_10" and flows_path and Path(flows_path).exists():
            self.flow_quantiles = _compute_flow_quantiles(Path(flows_path), use_slot)
        if label_transform == "linear_1_10" and flows_path and Path(flows_path).exists():
            self.flow_min, self.flow_max = _compute_flow_range(Path(flows_path))

        self._samples: List[Tuple] = []
        extract_fn = extract_1hop_subgraph if num_hops == 1 else extract_2hop_subgraph

        for ts in time_slices:
            day, slot_or_hour = ts[0], ts[1]
            data, norm = build_graph_data(
                nodes_path, edges_path, flows_path,
                day=day,
                hour=slot_or_hour if not use_slot else 12,
                slot_idx=slot_or_hour if use_slot else None,
                use_slot=use_slot,
                normalizer=self.normalizer,
                label_transform=label_transform,
                flow_quantiles=self.flow_quantiles,
                flow_min=self.flow_min,
                flow_max=self.flow_max,
            )
            if self.normalizer is None and norm is not None:
                self.normalizer = norm

            n_shop = data.num_shop
            n_public = data.num_public
            # 105 shop 子图
            for center_shop_idx in range(n_shop):
                center_global = center_shop_idx
                if use_shop_pseudo:
                    sub, _ = extract_subgraph_with_shop_pseudo(data, center_global, num_hops)
                else:
                    sub, _ = extract_fn(data, center_global)
                self._samples.append((sub, day, slot_or_hour, center_shop_idx, "shop"))
            # 38 public 子图
            for center_public_idx in range(n_public):
                center_global = n_shop + center_public_idx
                sub, _ = extract_fn(data, center_global)
                self._samples.append((sub, day, slot_or_hour, center_public_idx, "public"))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx][0]

    def get_meta(self, idx):
        return self._samples[idx][1], self._samples[idx][2], self._samples[idx][3], self._samples[idx][4]

"""
Build homogeneous PyG Data per time slice from nodes.csv, edges.csv, flows.csv.
- Node features: 11 continuous (z-score normalized) + node_type one-hot。所有节点同构。
- 节点顺序: shop 0..n_shop-1, public n_shop..N-1。仅 public 节点有标签 y/mask。
- is_public = (node_type == "public_space") 用于图和标签逻辑。
"""
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

CONT_FEAT_KEYS = [
    "x", "y", "compactness", "extensibility", "concavity",
    "fractal_degree", "street_length", "closeness", "betweenness",
]
# main_axis_dir_deg -> sin, cos (2 cols) -> total 11
MAIN_AXIS_KEY = "main_axis_dir_deg"
# node_type one-hot 顺序（与 build_nodes_edges 一致）
NODE_TYPE_OPTIONS = [
    "public_space", "shop_cultural", "shop_dining", "shop_residential",
    "shop_retail", "shop_undefined", "block",
]


def _is_public(node_type: str) -> bool:
    """is_public = (node_type == "public_space")，兼容旧数据 node_type=="public" """
    return node_type in ("public_space", "public")


def _load_nodes(nodes_path: Path) -> Tuple[List[Dict], List[int], List[int]]:
    with open(nodes_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # is_public = (node_type == "public_space")
    shop_ids = [r["node_id"] for r in rows if not _is_public(r.get("node_type", "shop"))]
    public_ids = [r["node_id"] for r in rows if _is_public(r.get("node_type", "public"))]
    # Ensure int
    for r in rows:
        r["node_id"] = int(r["node_id"])
        for k in ["x", "y", "compactness", "extensibility", "concavity", "fractal_degree",
                  "street_length", "closeness", "betweenness", "main_axis_dir_deg"]:
            r[k] = float(r[k])
    return rows, [int(x) for x in shop_ids], [int(x) for x in public_ids]


def _load_edges(edges_path: Path) -> List[Dict]:
    """Load edges; optional column 'open' (1=use in graph when no conversion, 0=only when incident to converted node)."""
    with open(edges_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["src"] = int(r["src"])
        r["dst"] = int(r["dst"])
        r["open"] = int(r.get("open", 1))  # backward compat: missing open -> 1
    return rows


def _load_flows(
    flows_path: Optional[Path],
    use_slot: bool = False,
) -> Dict[Tuple[int, ...], Dict[int, float]]:
    """
    Load flows. Key: (day, hour) if use_slot=False, else (day, slot_idx).
    Value: { node_id: flow }.
    """
    assert flows_path is None or isinstance(flows_path, Path), type(flows_path)
    out = {}
    if flows_path is None or not flows_path.exists():
        return out
    with open(flows_path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            day = int(r["day"])
            if use_slot and "slot_idx" in r:
                key = (day, int(r["slot_idx"]))
            else:
                key = (day, int(r.get("hour", r.get("slot_idx", 0))))
            nid = int(r["node_id"])
            flow = float(r["flow"])
            if key not in out:
                out[key] = {}
            out[key][nid] = flow
    return out


def _x_cont_from_rows(rows: List[Dict], node_id_to_idx: Dict[int, int], device=None) -> torch.Tensor:
    """Shape [N, 11]: 9 cont + sin(main_axis), cos(main_axis). Not normalized here."""
    N = len(node_id_to_idx)
    x = torch.zeros(N, 11)
    for r in rows:
        nid = r["node_id"]
        if nid not in node_id_to_idx:
            continue
        idx = node_id_to_idx[nid]
        for i, k in enumerate(CONT_FEAT_KEYS):
            x[idx, i] = r[k]
        rad = math.radians(r[MAIN_AXIS_KEY])
        x[idx, 9] = math.sin(rad)
        x[idx, 10] = math.cos(rad)
    if device is not None:
        x = x.to(device)
    return x


def _x_node_type_onehot(
    rows: List[Dict],
    node_id_to_idx: Dict[int, int],
    type_override: Optional[Dict[int, str]] = None,
    device=None,
) -> torch.Tensor:
    """Shape [N, len(NODE_TYPE_OPTIONS)]: node_type one-hot，不归一化。"""
    type_override = type_override or {}
    N = len(node_id_to_idx)
    n_types = len(NODE_TYPE_OPTIONS)
    type_to_idx = {t: i for i, t in enumerate(NODE_TYPE_OPTIONS)}
    # 兼容旧格式：shop -> shop_undefined, public -> public_space
    _type_map = {"shop": "shop_undefined", "public": "public_space"}
    x = torch.zeros(N, n_types)
    for r in rows:
        nid = r["node_id"]
        if nid not in node_id_to_idx:
            continue
        idx = node_id_to_idx[nid]
        nt = type_override.get(nid, r.get("node_type", "shop_undefined"))
        nt = _type_map.get(nt, nt)
        if nt not in type_to_idx:
            nt = "shop_undefined"
        x[idx, type_to_idx[nt]] = 1.0
    if device is not None:
        x = x.to(device)
    return x


def _nanstd(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Std ignoring NaN; compatible with PyTorch < 1.10 (no torch.nanstd)."""
    mean = torch.nanmean(x, dim=dim)
    var = torch.nanmean((x - mean) ** 2, dim=dim)
    return torch.sqrt(var)


def _normalizer_from_x(x: torch.Tensor) -> Dict[str, List[float]]:
    """Compute mean/std per column -> for JSON save."""
    mean = torch.nanmean(x, dim=0)
    std = _nanstd(x, dim=0)
    std[std == 0] = 1.0
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def normalize_x(x: torch.Tensor, normalizer: Optional[Dict] = None) -> torch.Tensor:
    """Z-score; if normalizer None, compute in-place and return (x_norm, normalizer)."""
    if normalizer is None:
        mean = torch.nanmean(x, dim=0)
        std = _nanstd(x, dim=0)
        std[std == 0] = 1.0
        x_norm = (x - mean) / std
        return x_norm, {"mean": mean.tolist(), "std": std.tolist()}
    mean = torch.tensor(normalizer["mean"], dtype=x.dtype, device=x.device)
    std = torch.tensor(normalizer["std"], dtype=x.dtype, device=x.device)
    return (x - mean) / std


def build_graph_data(
    nodes_path: Path,
    edges_path: Path,
    flows_path: Optional[Path] = None,
    day: int = 0,
    hour: int = 12,
    slot_idx: Optional[int] = None,
    use_slot: bool = False,
    normalizer: Optional[Dict] = None,
    label_transform: str = "log1p",
    flow_quantiles: Optional[List[float]] = None,
    flow_min: Optional[float] = None,
    flow_max: Optional[float] = None,
    device=None,
    shop_to_public: Optional[int] = None,
    include_all_edges: bool = False,
) -> Tuple[Data, Optional[Dict]]:
    """
    构建同构 PyG Data：节点顺序 shop 0..n_shop-1, public n_shop..N-1。
    - data.x [N, 18]：11 连续特征(z-score) + 7 维 node_type one-hot。
    - data.edge_index [2, E], data.edge_attr [E, 1]（open 0/1，若存在）。
    - include_all_edges=True 或仅公共节点时：保留所有边（含 open=0），并写入 data.edge_attr。
    - label_transform="linear_1_10" 时：非零 flow 线性映射到 [1,10] 作为客流效能指标（回归），需 flow_min/flow_max。
    """
    rows, shop_ids, public_ids = _load_nodes(nodes_path)
    edges_raw = _load_edges(edges_path)
    flows_map = _load_flows(flows_path, use_slot=use_slot)

    if shop_to_public is not None and shop_to_public in shop_ids:
        shop_ids = [nid for nid in shop_ids if nid != shop_to_public]
        public_ids = sorted(public_ids + [shop_to_public])

    nid_to_type = {}
    for r in rows:
        nid_to_type[r["node_id"]] = r["node_type"]
    type_override = {}
    if shop_to_public is not None:
        nid_to_type[shop_to_public] = "public_space"
        type_override[shop_to_public] = "public_space"
    shop_set = set(shop_ids)
    public_set = set(public_ids)
    if len(shop_set) == 0:
        include_all_edges = True  # 仅公共空间时保留所有边并记录 open

    node_id_to_idx_shop = {nid: i for i, nid in enumerate(shop_ids)}
    node_id_to_idx_public = {nid: i for i, nid in enumerate(public_ids)}

    all_rows = [r for r in rows if r["node_id"] in shop_set] + [r for r in rows if r["node_id"] in public_set]
    global_id_to_idx = {}
    for i, r in enumerate(all_rows):
        global_id_to_idx[r["node_id"]] = i
    # 11 维连续特征（z-score 归一化）
    x_raw_cont = _x_cont_from_rows(all_rows, global_id_to_idx, device=device)
    if normalizer is None:
        x_cont_norm, norm_dict = normalize_x(x_raw_cont, None)
        normalizer = norm_dict
    else:
        x_cont_norm = normalize_x(x_raw_cont, normalizer)
    # node_type one-hot（7 维，不归一化）
    x_onehot = _x_node_type_onehot(all_rows, global_id_to_idx, type_override, device=device)
    x_cont = torch.cat([x_cont_norm, x_onehot], dim=1)  # [N, 18]

    n_shop = len(shop_ids)
    n_public = len(public_ids)
    N = n_shop + n_public

    # 全局节点索引: 0..n_shop-1 shop, n_shop..N-1 public
    def global_idx(nid: int):
        if nid in node_id_to_idx_shop:
            return node_id_to_idx_shop[nid]
        return n_shop + node_id_to_idx_public[nid]

    edge_src, edge_dst, edge_open = [], [], []
    for e in edges_raw:
        u, v = e["src"], e["dst"]
        open_val = e.get("open", 1)
        if u not in global_id_to_idx or v not in global_id_to_idx:
            continue
        # 仅当非“全边模式”且 open=0 时跳过（保留与 shop 转换逻辑兼容）
        if not include_all_edges and open_val != 1 and (shop_to_public is None or (u != shop_to_public and v != shop_to_public)):
            continue
        for (s, d) in [(u, v), (v, u)]:
            edge_src.append(global_idx(s))
            edge_dst.append(global_idx(d))
            edge_open.append(float(open_val))

    key_ts = (day, slot_idx) if (use_slot and slot_idx is not None) else (day, hour)
    flow_dict = flows_map.get(key_ts, {})
    y_list = []
    mask_list = []
    is_classification = label_transform == "remap_1_10" and flow_quantiles is not None
    # linear_1_10: 非零 flow 线性映射到 [1,10]，消除量纲；零保持 0（回归任务）
    use_linear_1_10 = label_transform == "linear_1_10"
    for nid in public_ids:
        if nid in flow_dict:
            flow = flow_dict[nid]
            if label_transform == "log1p":
                y_list.append(math.log1p(flow))
            elif is_classification:
                # 分位点 [10,20,...,90] 将 flow 映射到 0-9
                cls_idx = int(np.digitize(flow, flow_quantiles, right=False))
                cls_idx = min(cls_idx, 9)
                y_list.append(cls_idx)
            elif use_linear_1_10:
                if flow <= 0:
                    y_list.append(0.0)
                else:
                    fmin = flow_min if flow_min is not None else 0.0
                    fmax = flow_max if flow_max is not None else (flow + 1.0)
                    if fmax <= fmin:
                        y_list.append(1.0)
                    else:
                        t = (flow - fmin) / (fmax - fmin)
                        y_list.append(1.0 + float(np.clip(t, 0.0, 1.0)) * 9.0)
            else:
                y_list.append(float(flow))
            mask_list.append(1)
        else:
            y_list.append(0 if is_classification else 0.0)
            mask_list.append(0)

    data = Data(
        x=x_cont,
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long) if edge_src else torch.empty(2, 0, dtype=torch.long),
    )
    if edge_open:
        data.edge_attr = torch.tensor(edge_open, dtype=torch.float32).unsqueeze(1)  # [E, 1]
    data.num_nodes = N
    data.num_shop = n_shop
    data.num_public = n_public
    data.y = torch.tensor(y_list, dtype=torch.long if is_classification else torch.float32).unsqueeze(1)
    data.mask = torch.tensor(mask_list, dtype=torch.bool)
    data.node_id = torch.tensor(shop_ids + public_ids, dtype=torch.long)

    return data, normalizer


def build_graph_data_from_rows(
    rows: List[Dict],
    edges_raw: List[Dict],
    normalizer: Optional[Dict] = None,
    device=None,
) -> Tuple[Data, Optional[Dict]]:
    """
    从内存中的节点行和边列表构建同构 PyG Data（无 flows，用于 GA 等场景）。
    rows: 每项为 dict，含 node_id, node_type, x, y, compactness, extensibility,
          concavity, fractal_degree, street_length, closeness, betweenness, main_axis_dir_deg。
    edges_raw: 每项为 dict，含 src, dst, open(可选，默认 1)。
    节点顺序：shop 按 node_id 升序，public 按 node_id 升序，整体 shop 在前 public 在后。
    """
    shop_ids = sorted([r["node_id"] for r in rows if not _is_public(r.get("node_type", "shop"))])
    public_ids = sorted([r["node_id"] for r in rows if _is_public(r.get("node_type", "public"))])
    for r in rows:
        r["node_id"] = int(r["node_id"])
        for k in ["x", "y", "compactness", "extensibility", "concavity", "fractal_degree",
                  "street_length", "closeness", "betweenness", "main_axis_dir_deg"]:
            if k in r:
                r[k] = float(r[k])
            else:
                r[k] = 0.0
    n_shop = len(shop_ids)
    n_public = len(public_ids)
    N = n_shop + n_public
    shop_set = set(shop_ids)
    public_set = set(public_ids)
    node_id_to_idx_shop = {nid: i for i, nid in enumerate(shop_ids)}
    node_id_to_idx_public = {nid: i for i, nid in enumerate(public_ids)}
    all_rows = [r for r in rows if r["node_id"] in shop_set] + [r for r in rows if r["node_id"] in public_set]
    global_id_to_idx = {r["node_id"]: i for i, r in enumerate(all_rows)}
    x_raw_cont = _x_cont_from_rows(all_rows, global_id_to_idx, device=device)
    if normalizer is None:
        x_cont_norm, normalizer = normalize_x(x_raw_cont, None)
    else:
        x_cont_norm = normalize_x(x_raw_cont, normalizer)
    x_onehot = _x_node_type_onehot(all_rows, global_id_to_idx, None, device=device)
    x_cont = torch.cat([x_cont_norm, x_onehot], dim=1)

    def global_idx(nid: int):
        if nid in node_id_to_idx_shop:
            return node_id_to_idx_shop[nid]
        return n_shop + node_id_to_idx_public[nid]

    edge_src, edge_dst, edge_open = [], [], []
    for e in edges_raw:
        u, v = int(e["src"]), int(e["dst"])
        open_val = float(e.get("open", 1))
        if u not in global_id_to_idx or v not in global_id_to_idx:
            continue
        for (s, d) in [(u, v), (v, u)]:
            edge_src.append(global_idx(s))
            edge_dst.append(global_idx(d))
            edge_open.append(open_val)

    data = Data(
        x=x_cont,
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long) if edge_src else torch.empty(2, 0, dtype=torch.long),
    )
    if edge_open:
        data.edge_attr = torch.tensor(edge_open, dtype=torch.float32).unsqueeze(1)
    data.num_nodes = N
    data.num_shop = n_shop
    data.num_public = n_public
    data.y = torch.zeros(n_public, 1, dtype=torch.float32)
    data.mask = torch.zeros(n_public, dtype=torch.bool)
    data.node_id = torch.tensor(shop_ids + public_ids, dtype=torch.long)
    return data, normalizer


def save_normalizer(normalizer: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalizer, f, indent=2)


def load_normalizer(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

"""
Build homogeneous PyG Data per time slice from nodes.csv, edges.csv, flows.csv.
- Node features: 11 continuous (z-score normalized)，不含时间特征。所有节点同构。
- 节点顺序: shop 0..n_shop-1, public n_shop..N-1。仅 public 节点有标签 y/mask。
"""
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data

CONT_FEAT_KEYS = [
    "x", "y", "compactness", "extensibility", "concavity",
    "fractal_degree", "street_length", "closeness", "betweenness",
]
# main_axis_dir_deg -> sin, cos (2 cols) -> total 11
MAIN_AXIS_KEY = "main_axis_dir_deg"


def _load_nodes(nodes_path: Path) -> Tuple[List[Dict], List[int], List[int]]:
    with open(nodes_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    shop_ids = [r["node_id"] for r in rows if r["node_type"] == "shop"]
    public_ids = [r["node_id"] for r in rows if r["node_type"] == "public"]
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
    out = {}
    if flows_path is None or not flows_path.exists():
        return out
    with open(flows_path, "r", encoding="utf-8") as f:
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
    device=None,
    shop_to_public: Optional[int] = None,
) -> Tuple[Data, Optional[Dict]]:
    """
    构建同构 PyG Data：节点顺序 shop 0..n_shop-1, public n_shop..N-1。
    - data.x [N, 11], data.edge_index [2, E], data.num_shop, data.num_public。
    - 仅 public 有 data.y [n_public, 1], data.mask [n_public]；data.node_id [N]。
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
    if shop_to_public is not None:
        nid_to_type[shop_to_public] = "public"
    shop_set = set(shop_ids)
    public_set = set(public_ids)

    node_id_to_idx_shop = {nid: i for i, nid in enumerate(shop_ids)}
    node_id_to_idx_public = {nid: i for i, nid in enumerate(public_ids)}

    all_rows = [r for r in rows if r["node_id"] in shop_set] + [r for r in rows if r["node_id"] in public_set]
    global_id_to_idx = {}
    for i, r in enumerate(all_rows):
        global_id_to_idx[r["node_id"]] = i
    x_raw = _x_cont_from_rows(all_rows, global_id_to_idx, device=device)
    if normalizer is None:
        x_cont, norm_dict = normalize_x(x_raw, None)
        normalizer = norm_dict
    else:
        x_cont = normalize_x(x_raw, normalizer)

    n_shop = len(shop_ids)
    n_public = len(public_ids)
    N = n_shop + n_public

    # 全局节点索引: 0..n_shop-1 shop, n_shop..N-1 public
    def global_idx(nid: int):
        if nid in node_id_to_idx_shop:
            return node_id_to_idx_shop[nid]
        return n_shop + node_id_to_idx_public[nid]

    edge_src, edge_dst = [], []
    for e in edges_raw:
        u, v = e["src"], e["dst"]
        open_val = e.get("open", 1)
        if open_val != 1 and (shop_to_public is None or (u != shop_to_public and v != shop_to_public)):
            continue
        if u not in global_id_to_idx or v not in global_id_to_idx:
            continue
        edge_src.append(global_idx(u))
        edge_dst.append(global_idx(v))
        edge_src.append(global_idx(v))
        edge_dst.append(global_idx(u))

    key_ts = (day, slot_idx) if (use_slot and slot_idx is not None) else (day, hour)
    flow_dict = flows_map.get(key_ts, {})
    y_list = []
    mask_list = []
    for nid in public_ids:
        if nid in flow_dict:
            flow = flow_dict[nid]
            if label_transform == "log1p":
                y_list.append(math.log1p(flow))
            elif label_transform == "remap_1_10":
                y_list.append(float(flow))
            else:
                y_list.append(float(flow))
            mask_list.append(1)
        else:
            y_list.append(0.0)
            mask_list.append(0)

    data = Data(
        x=x_cont,
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long) if edge_src else torch.empty(2, 0, dtype=torch.long),
    )
    data.num_nodes = N
    data.num_shop = n_shop
    data.num_public = n_public
    data.y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    data.mask = torch.tensor(mask_list, dtype=torch.bool)
    data.node_id = torch.tensor(shop_ids + public_ids, dtype=torch.long)

    return data, normalizer


def save_normalizer(normalizer: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalizer, f, indent=2)


def load_normalizer(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

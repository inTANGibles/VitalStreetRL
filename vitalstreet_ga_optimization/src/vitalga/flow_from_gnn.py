"""
活力接入 GNN：按每个 public_space 构建子图，输入 GNN 预测客流量，写回 state.space_units.flow_prediction。

依赖：streetflow_hetero_time_gnn（需将该项目根路径加入 sys.path 或通过 streetflow_root 传入）。
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import math
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

from .state import WorldState


def _compute_shape_features(poly) -> dict:
    """从多边形计算 compactness, concavity, fractal_degree, extensibility, main_axis_dir, perimeter。"""
    if poly is None or getattr(poly, "is_empty", True):
        return dict(compactness=0.0, concavity=0.0, fractal_degree=0.0, extensibility=0.0, main_axis_dir=0.0, perimeter=0.0)
    area = float(poly.area)
    perimeter = float(poly.length)
    compactness = 4.0 * math.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
    hull = poly.convex_hull
    hull_area = float(hull.area)
    concavity = (1.0 - area / hull_area) if hull_area > 0 else 0.0
    fractal_degree = float(math.log(perimeter) / math.log(area)) if area > 1.0 and perimeter > 1.0 else 0.0
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    extensibility, main_axis_dir = 0.0, 0.0
    if len(coords) >= 4:
        edge_lengths, edge_dirs = [], []
        for i in range(len(coords) - 1):
            x1, y1, x2, y2 = *coords[i], *coords[i + 1]
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length <= 0:
                continue
            edge_lengths.append(length)
            edge_dirs.append(math.degrees(math.atan2(dy, dx)))
        if edge_lengths:
            unique_lengths = sorted(set(round(l, 6) for l in edge_lengths))
            short_side, long_side = unique_lengths[0], unique_lengths[-1]
            extensibility = float(long_side / short_side) if short_side > 0 else 0.0
            main_axis_dir = edge_dirs[edge_lengths.index(max(edge_lengths))] % 180.0
    return dict(
        compactness=compactness,
        concavity=concavity,
        fractal_degree=fractal_degree,
        extensibility=extensibility,
        main_axis_dir=main_axis_dir,
        perimeter=perimeter,
    )


def _build_adjacency_edges(gdf: gpd.GeoDataFrame) -> set:
    """多边形邻接：touches / overlaps / intersects。"""
    edges = set()
    try:
        sindex = gdf.sindex
    except Exception:
        return edges
    geometries = gdf.geometry.values
    for i, geom in enumerate(geometries):
        if geom is None or geom.is_empty:
            continue
        for j in sindex.intersection(geom.bounds):
            if j <= i:
                continue
            geom_j = geometries[j] if j < len(geometries) else None
            if geom_j is None or geom_j.is_empty:
                continue
            if not getattr(geom.envelope, "intersects", lambda _: False)(geom_j.envelope):
                continue
            if geom.touches(geom_j) or geom.overlaps(geom_j) or geom.intersects(geom_j):
                edges.add((i, j))
    return edges


def build_rows_and_edges_from_state(state: WorldState) -> Tuple[List[Dict], List[Dict], List[Any]]:
    """
    从 WorldState 的 space_units 构建 GNN 所需的 rows 与 edges，以及 node_id -> 单元标识 的映射。
    节点顺序：shop 在前，public_space 在后；node_id = 0..N-1（即行下标）。
    Returns:
        rows: 每项为 dict(node_id, node_type, x, y, compactness, extensibility, concavity,
              fractal_degree, street_length, closeness, betweenness, main_axis_dir_deg)
        edges_raw: 每项为 dict(src, dst, open)
        unit_order: unit_order[node_id] 为 state.space_units.get_all_space_units() 的 index（uid）
    """
    gdf_all = state.space_units.get_all_space_units()
    if gdf_all.empty:
        return [], [], []

    shop_mask = gdf_all["unit_type"] == "shop"
    public_mask = gdf_all["unit_type"] == "public_space"
    gdf_shop = gdf_all[shop_mask].copy()
    gdf_public = gdf_all[public_mask].copy()
    gdf_ordered = gpd.GeoDataFrame(
        pd.concat([gdf_shop, gdf_public], ignore_index=False),
        geometry="geometry",
        crs=gdf_all.crs,
    )
    gdf_ordered = gdf_ordered.reset_index(drop=False)
    gdf_ordered["_uid"] = gdf_ordered.index
    n = len(gdf_ordered)
    gdf_ordered["node_id"] = np.arange(n, dtype=int)
    gdf_ordered["node_type"] = gdf_ordered["unit_type"].map(lambda t: "shop" if t == "shop" else "public")

    centroids = gdf_ordered.geometry.centroid
    gdf_ordered["x"] = centroids.x.astype(float)
    gdf_ordered["y"] = centroids.y.astype(float)
    shape_feats = [_compute_shape_features(geom) for geom in gdf_ordered.geometry]
    for k in ["compactness", "extensibility", "concavity", "fractal_degree", "main_axis_dir", "perimeter"]:
        gdf_ordered[k] = [f[k] for f in shape_feats]
    gdf_ordered["street_length"] = [f["perimeter"] for f in shape_feats]
    gdf_ordered["main_axis_dir_deg"] = [f["main_axis_dir"] for f in shape_feats]

    edges_set = _build_adjacency_edges(gdf_ordered)
    edges_list = sorted({(int(min(u, v)), int(max(u, v))) for u, v in edges_set})
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges_list)
        closeness = nx.closeness_centrality(G)
        betweenness = nx.betweenness_centrality(G, normalized=True)
    except Exception:
        closeness = {i: 0.0 for i in range(n)}
        betweenness = {i: 0.0 for i in range(n)}
    gdf_ordered["closeness"] = gdf_ordered["node_id"].map(closeness).astype(float)
    gdf_ordered["betweenness"] = gdf_ordered["node_id"].map(betweenness).astype(float)

    unit_order = list(gdf_ordered["_uid"].values)

    rows = []
    for _, row in gdf_ordered.iterrows():
        rows.append({
            "node_id": int(row["node_id"]),
            "node_type": row["node_type"],
            "x": float(row["x"]),
            "y": float(row["y"]),
            "compactness": float(row["compactness"]),
            "extensibility": float(row["extensibility"]),
            "concavity": float(row["concavity"]),
            "fractal_degree": float(row["fractal_degree"]),
            "street_length": float(row["street_length"]),
            "closeness": float(row["closeness"]),
            "betweenness": float(row["betweenness"]),
            "main_axis_dir_deg": float(row["main_axis_dir_deg"]),
        })

    edges_raw = [{"src": u, "dst": v, "open": 1} for u, v in edges_list]
    return rows, edges_raw, unit_order


def predict_flows_gnn(
    state: WorldState,
    checkpoint_dir: Path,
    normalizer_path: Optional[Path] = None,
    streetflow_root: Optional[Path] = None,
    num_hops: int = 2,
    device=None,
    checkpoint_name: str = "best.pt",
    use_slot: bool = False,
) -> None:
    """
    为 state 中每个 public_space 构建子图并调用 GNN 预测客流量，写回 state.space_units 的 flow_prediction。
    - checkpoint_dir: 存放 best.pt（或 checkpoint_name）与 normalizer.json 的目录
    - normalizer_path: 若为 None 则使用 checkpoint_dir / "normalizer.json"
    - streetflow_root: streetflow_hetero_time_gnn 项目根目录，用于 import；若为 None 则尝试默认相对路径
    - num_hops: 1 或 2，子图跳数
    - use_slot: 与 GNN 训练时一致（log1p 回归时通常 False）
    """
    try:
        import torch
    except ImportError:
        raise ImportError("flow_from_gnn 需要 PyTorch，请安装 torch。")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_dir)
    if normalizer_path is None:
        normalizer_path = checkpoint_dir / "normalizer.json"

    if streetflow_root is not None:
        streetflow_root = Path(streetflow_root)
        if str(streetflow_root) not in sys.path:
            sys.path.insert(0, str(streetflow_root))

    rows, edges_raw, unit_order = build_rows_and_edges_from_state(state)
    if not rows:
        return
    n_shop = sum(1 for r in rows if r.get("node_type") == "shop")
    n_public = len(rows) - n_shop
    if n_public == 0:
        return

    from geo.tool.build_graph import build_graph_data_from_rows, load_normalizer
    from data.subgraph_utils import extract_1hop_subgraph, extract_2hop_subgraph
    from models.sage import SAGE

    normalizer = load_normalizer(normalizer_path) if normalizer_path.exists() else None
    data, _ = build_graph_data_from_rows(rows, edges_raw, normalizer=normalizer, device=device)
    extract_fn = extract_1hop_subgraph if num_hops == 1 else extract_2hop_subgraph

    ckpt_path = checkpoint_dir / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"GNN 权重不存在: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    hidden = int(ckpt.get("hidden_channels", 64))
    model = SAGE(in_channels=11, hidden_channels=hidden, out_channels=1, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_units = state.space_units.get_all_space_units()
    shop_mask = all_units["unit_type"] == "shop"
    if shop_mask.any():
        all_units.loc[shop_mask, "flow_prediction"] = 0.0

    with torch.no_grad():
        for center_public_idx in range(n_public):
            center_global = n_shop + center_public_idx
            sub, center_local = extract_fn(data, center_global)
            sub = sub.to(device)
            out = model(sub.x, sub.edge_index).squeeze(1)
            yhat_log = out[center_local].item()
            yhat_flow = float(np.expm1(yhat_log)) if not use_slot else float(yhat_log)
            uid = unit_order[center_global]
            all_units.loc[uid, "flow_prediction"] = max(0.0, yhat_flow)

    state.space_units._SpaceUnitCollection__unit_gdf = all_units

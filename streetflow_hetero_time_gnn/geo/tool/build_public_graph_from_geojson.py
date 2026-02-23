#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 0222_flow_mapped_public.geojson 构建公共空间图：约 40 个节点，边由邻接关系构建。
open 定义：公共空间–公共空间 恒为 1（本脚本仅含 public，故所有边 open=1）。
客流从 geojson 的 pass_count / flow_5min 提取，无时间片概念。
"""
import argparse
import math
import warnings
from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import networkx as nx
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="从公共空间 GeoJSON 构建 nodes/edges/flows")
    p.add_argument("--geojson", required=True, help="公共空间 GeoJSON（如 0222_flow_mapped_public.geojson）")
    p.add_argument("--out_dir", required=True, help="输出目录：nodes.csv, edges.csv, flows.csv")
    p.add_argument("--flow_field", default="flow_5min", choices=["flow_5min", "pass_count", "flow_prediction"],
                   help="用作流量的属性字段")
    return p.parse_args()


def ensure_valid_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf[gdf.geometry.notnull()].copy()
    if gdf.empty:
        raise ValueError("所有几何为空。")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")
        gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[gdf.geometry.notnull()].copy()
    return gdf


def compute_shape_features(poly) -> dict:
    """compactness, concavity, fractal_degree, extensibility, main_axis_dir."""
    area = float(poly.area)
    perimeter = float(poly.length)
    compactness = (4.0 * math.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    hull = poly.convex_hull
    hull_area = float(hull.area)
    concavity = (1.0 - area / hull_area) if hull_area > 0 else 0.0
    fractal_degree = float(math.log(perimeter) / math.log(area)) if (area > 1.0 and perimeter > 1.0) else 0.0
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    extensibility, main_axis_dir = 0.0, 0.0
    if len(coords) >= 4:
        edge_lengths = []
        edge_dirs = []
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
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
            idx = edge_lengths.index(max(edge_lengths))
            main_axis_dir = edge_dirs[idx] % 180.0
    return dict(
        compactness=compactness, concavity=concavity, fractal_degree=fractal_degree,
        extensibility=extensibility, main_axis_dir=main_axis_dir, perimeter=perimeter,
    )


def build_adjacency_edges(gdf: gpd.GeoDataFrame) -> List[Tuple[int, int]]:
    """多边形邻接：touches / intersects，返回 (i, j) 且 i < j。"""
    edges = []
    sindex = gdf.sindex
    geometries = gdf.geometry.values
    for i, geom in enumerate(geometries):
        if geom is None or geom.is_empty:
            continue
        for j in sindex.intersection(geom.bounds):
            if j <= i:
                continue
            geom_j = geometries[j]
            if geom_j is None or geom_j.is_empty:
                continue
            if geom.touches(geom_j) or geom.intersects(geom_j):
                edges.append((i, j))
    return edges


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 公共空间
    gdf = gpd.read_file(args.geojson)
    if "unit_type" in gdf.columns:
        gdf = gdf[gdf["unit_type"] == "public_space"].copy()
    gdf = ensure_valid_geometries(gdf)
    gdf = gdf.reset_index(drop=True)
    n_nodes = len(gdf)

    # 形状特征 + 质心
    shape_feats = [compute_shape_features(geom) for geom in gdf.geometry]
    shape_df = pd.DataFrame(shape_feats)
    centroids = gdf.geometry.centroid
    gdf["x"] = centroids.x.astype(float)
    gdf["y"] = centroids.y.astype(float)
    gdf["compactness"] = shape_df["compactness"]
    gdf["extensibility"] = shape_df["extensibility"]
    gdf["concavity"] = shape_df["concavity"]
    gdf["fractal_degree"] = shape_df["fractal_degree"]
    gdf["main_axis_dir_deg"] = shape_df["main_axis_dir"]
    gdf["street_length"] = shape_df["perimeter"]

    # 4. 邻接边
    edges_list = build_adjacency_edges(gdf)

    # 5. 中心性（基于邻接图）
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(edges_list)
    try:
        closeness_dict = nx.closeness_centrality(G)
    except Exception:
        closeness_dict = {i: 0.0 for i in range(n_nodes)}
    try:
        betweenness_dict = nx.betweenness_centrality(G, normalized=True)
    except Exception:
        betweenness_dict = {i: 0.0 for i in range(n_nodes)}
    gdf["closeness"] = gdf.index.map(lambda i: closeness_dict.get(i, 0.0)).astype(float)
    gdf["betweenness"] = gdf.index.map(lambda i: betweenness_dict.get(i, 0.0)).astype(float)

    # 6. 边的 open：本图仅公共空间，public–public 恒为 1
    open_list = [1] * len(edges_list)

    # 7. nodes.csv（node_id 用行号 0..N-1）
    gdf["node_id"] = gdf.index.astype(int)
    nodes_df = gdf[
        [
            "node_id", "x", "y", "compactness", "extensibility", "concavity",
            "fractal_degree", "street_length", "closeness", "betweenness", "main_axis_dir_deg",
        ]
    ].copy()
    nodes_df.insert(1, "node_type", "public")
    nodes_path = out_dir / "nodes.csv"
    nodes_df.to_csv(nodes_path, index=False)

    # 8. edges.csv: src, dst, open
    edges_df = pd.DataFrame(
        [
            {"src": i, "dst": j, "open": o}
            for (i, j), o in zip(edges_list, open_list)
        ]
    )
    edges_path = out_dir / "edges.csv"
    edges_df.to_csv(edges_path, index=False)

    # 9. flows.csv：单一时刻，day=0, hour=0
    flow_col = args.flow_field
    if flow_col not in gdf.columns:
        flow_col = "flow_5min" if "flow_5min" in gdf.columns else "pass_count"
    rows = []
    for idx in gdf.index:
        node_id = int(gdf.loc[idx, "node_id"])
        val = gdf.loc[idx, flow_col]
        flow = float(val) if pd.notna(val) else 0.0
        rows.append({"node_id": node_id, "day": 0, "hour": 0, "flow": max(0.0, flow)})
    flows_path = out_dir / "flows.csv"
    pd.DataFrame(rows).to_csv(flows_path, index=False)

    print(f"节点数: {n_nodes}, 边数: {len(edges_list)}（均为 public–public，open=1）")
    print(f"nodes.csv -> {nodes_path}")
    print(f"edges.csv -> {edges_path}")
    print(f"flows.csv -> {flows_path}")


if __name__ == "__main__":
    main()

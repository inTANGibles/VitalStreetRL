#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_nodes_edges.py（仿照 VitalStreetRL/geo/tool/build_nodes_edges.py）

从 street.geojson（或分开的 shop/public 图层）构建 nodes.csv 与 edges.csv：
- 形状特征：compactness, concavity, fractal_degree, extensibility, main_axis_dir（最小外接矩形）
- 边：多边形邻接（touches/overlaps/intersects），可选近邻补边
- 中心性：closeness, betweenness
- 输出列：node_id, node_type(shop|public), x, y, compactness, extensibility, concavity,
  fractal_degree, street_length, closeness, betweenness, main_axis_dir_deg
- edges.csv: src, dst, open. 所有相邻图块的边都输出；open=1 当且仅当两端点中至少有一个为 public_space，
  否则 open=0。构建图时仅使用 open=1 的边；GA 将 shop 改为 public 时可直接用同一 edge 集构建子图。
"""

import argparse
import math
import warnings
from pathlib import Path
from typing import Optional

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="从空间多边形构建 nodes.csv 和 edges.csv（streetflow 格式）")
    p.add_argument("--geojson", default=None, help="单一 GeoJSON（含 unit_type: shop/public_space），与 --shop/--public 二选一")
    p.add_argument("--shop", default=None, help="店铺 polygon 路径（与 --geojson 二选一）")
    p.add_argument("--public", default=None, help="公共空间 polygon 路径（可选，与 --geojson 二选一）")
    p.add_argument("--layer_shop", default=None)
    p.add_argument("--layer_public", default=None)
    p.add_argument("--d_thresh", type=float, default=0.0, help="近邻连边质心距离阈值，0 表示不用")
    p.add_argument("--adjacency_only", type=int, choices=[0, 1], default=0, help="1: 仅邻接；0: 可近邻补边")
    p.add_argument("--out_dir", required=True, help="输出目录，生成 nodes.csv、edges.csv")
    p.add_argument("--flows_csv", action="store_true", help="若设，从 properties.flow_prediction 生成 flows.csv（42 time slices）")
    return p.parse_args()


def read_layer(path: Path, layer: Optional[str]) -> gpd.GeoDataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}")
    if path.suffix.lower() == ".gpkg" and layer:
        return gpd.read_file(path, layer=layer)
    return gpd.read_file(path)


def ensure_valid_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf[gdf.geometry.notnull()].copy()
    if gdf.empty:
        raise ValueError("所有几何为空。")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")
        gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[gdf.geometry.notnull()].copy()
    if gdf.empty:
        raise ValueError("几何修复后仍为空。")
    return gdf


def compute_shape_features(poly) -> dict:
    """与 VitalStreetRL 一致：compactness, concavity, fractal_degree, extensibility, main_axis_dir。"""
    area = float(poly.area)
    perimeter = float(poly.length)

    if perimeter > 0:
        compactness = 4.0 * math.pi * area / (perimeter ** 2)
    else:
        compactness = 0.0

    hull = poly.convex_hull
    hull_area = float(hull.area)
    concavity = (1.0 - area / hull_area) if hull_area > 0 else 0.0

    if area > 1.0 and perimeter > 1.0:
        fractal_degree = float(math.log(perimeter) / math.log(area))
    else:
        fractal_degree = 0.0

    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    extensibility = 0.0
    main_axis_dir = 0.0
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
            short_side = unique_lengths[0]
            long_side = unique_lengths[-1]
            extensibility = float(long_side / short_side) if short_side > 0 else 0.0
            idx = edge_lengths.index(max(edge_lengths))
            main_axis_dir = edge_dirs[idx] % 180.0

    return dict(
        compactness=compactness,
        concavity=concavity,
        fractal_degree=fractal_degree,
        extensibility=extensibility,
        main_axis_dir=main_axis_dir,
        perimeter=perimeter,
    )


def build_adjacency_edges(gdf: gpd.GeoDataFrame) -> set:
    """多边形邻接：touches / overlaps / intersects。"""
    edges = set()
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
            if not geom.envelope.intersects(geom_j.envelope):
                continue
            if geom.touches(geom_j) or geom.overlaps(geom_j) or geom.intersects(geom_j):
                edges.add((i, j))
    return edges


def build_knn_edges(gdf: gpd.GeoDataFrame, base_edges: set, d_thresh: float) -> set:
    if d_thresh <= 0:
        return base_edges
    centroids = gdf.geometry.centroid
    coords = np.array([[p.x, p.y] for p in centroids])
    sindex = centroids.sindex
    edges = set(base_edges)
    for i, (xi, yi) in enumerate(coords):
        box = (xi - d_thresh, yi - d_thresh, xi + d_thresh, yi + d_thresh)
        for j in sindex.intersection(box):
            if j <= i:
                continue
            dist = math.hypot(coords[j, 0] - xi, coords[j, 1] - yi)
            if dist < d_thresh:
                edges.add((i, j))
    return edges


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.geojson:
        gdf_all = read_layer(Path(args.geojson), None)
        gdf_all = ensure_valid_geometries(gdf_all)
        if "unit_type" not in gdf_all.columns:
            gdf_all["unit_type"] = "shop"
        shop_gdf = gdf_all[gdf_all["unit_type"] == "shop"].copy()
        public_gdf = gdf_all[gdf_all["unit_type"] == "public_space"].copy()
        if len(shop_gdf) == 0:
            shop_gdf = gdf_all.copy()
        if len(public_gdf) == 0:
            public_gdf = gdf_all.head(0).copy()
        shop_gdf["__geom_type__"] = "shop"
        public_gdf["__geom_type__"] = "public"
        gdf = pd.concat([shop_gdf, public_gdf], ignore_index=True)
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=gdf_all.crs)
    else:
        if not args.shop:
            raise ValueError("请指定 --geojson 或 --shop")
        shop_gdf = read_layer(Path(args.shop), args.layer_shop)
        shop_gdf = ensure_valid_geometries(shop_gdf)
        shop_gdf["__geom_type__"] = "shop"
        if args.public:
            public_gdf = read_layer(Path(args.public), args.layer_public)
            public_gdf = ensure_valid_geometries(public_gdf)
            public_gdf["__geom_type__"] = "public"
            if public_gdf.crs != shop_gdf.crs:
                public_gdf = public_gdf.to_crs(shop_gdf.crs)
            gdf = pd.concat([shop_gdf, public_gdf], ignore_index=True)
        else:
            gdf = shop_gdf.copy()
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=shop_gdf.crs)

    gdf = ensure_valid_geometries(gdf)
    gdf = gdf.reset_index(drop=True)
    gdf["node_id"] = gdf.index.astype(int)
    n_nodes = len(gdf)

    # 形状特征 + 质心 + street_length
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

    # 边
    adj_edges = build_adjacency_edges(gdf)
    if args.adjacency_only == 0 and args.d_thresh > 0:
        edges_set = build_knn_edges(gdf, adj_edges, args.d_thresh)
    else:
        edges_set = adj_edges
    edges_set = {(int(min(u, v)), int(max(u, v))) for u, v in edges_set}
    edges_list = sorted(edges_set)

    # 中心性
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
    gdf["closeness"] = gdf["node_id"].map(closeness_dict).astype(float)
    gdf["betweenness"] = gdf["node_id"].map(betweenness_dict).astype(float)

    # node_type: shop / public
    gdf["node_type"] = gdf["__geom_type__"].map(lambda t: "shop" if t == "shop" else "public")
    nid_to_type = gdf.set_index("node_id")["node_type"].to_dict()

    # nodes.csv（streetflow 所需列名）
    nodes_df = gdf[
        [
            "node_id", "node_type", "x", "y", "compactness", "extensibility", "concavity",
            "fractal_degree", "street_length", "closeness", "betweenness", "main_axis_dir_deg",
        ]
    ].copy()
    nodes_path = out_dir / "nodes.csv"
    nodes_df.to_csv(nodes_path, index=False)

    # edges.csv: src, dst, open. 所有相邻边都保留；open=1 当且仅当两端至少有一个为 public_space
    open_list = []
    for (u, v) in edges_list:
        tu, tv = nid_to_type.get(u, "shop"), nid_to_type.get(v, "shop")
        open_list.append(1 if (tu == "public" or tv == "public") else 0)
    edges_df = pd.DataFrame(
        list(zip([e[0] for e in edges_list], [e[1] for e in edges_list], open_list)),
        columns=["src", "dst", "open"],
    )
    edges_path = out_dir / "edges.csv"
    edges_df.to_csv(edges_path, index=False)

    print(f"节点数: {n_nodes}, 边数: {len(edges_list)}")
    print(f"nodes.csv -> {nodes_path}")
    print(f"edges.csv -> {edges_path}")

    # 可选：flows.csv（42 time slices，仅 public，用 flow_prediction）
    if args.flows_csv and "flow_prediction" in gdf.columns:
        DAYS = list(range(7))
        HOURS = list(range(12, 18))
        rows = []
        for idx in gdf.index:
            if gdf.loc[idx, "node_type"] != "public":
                continue
            node_id = int(gdf.loc[idx, "node_id"])
            flow_val = float(gdf.loc[idx, "flow_prediction"]) if pd.notna(gdf.loc[idx, "flow_prediction"]) else 0.0
            for day in DAYS:
                for hour in HOURS:
                    rows.append({"node_id": node_id, "day": day, "hour": hour, "flow": max(0.0, flow_val)})
        if rows:
            flows_path = out_dir / "flows.csv"
            pd.DataFrame(rows).to_csv(flows_path, index=False)
            print(f"flows.csv -> {flows_path} ({len(rows)} 行)")


if __name__ == "__main__":
    main()

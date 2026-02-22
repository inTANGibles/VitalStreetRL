#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_nodes_edges.py

从店铺 + 公共空间多边形数据构建节点/边表，用于后续图神经网络客流预测。
- 输出 nodes.csv 与 edges.csv 到指定目录。
"""

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import Optional

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="从空间多边形构建 nodes.csv 和 edges.csv"
    )
    parser.add_argument("--shop", required=True, help="店铺 polygon 数据路径（shp/geojson/gpkg）")
    parser.add_argument("--public", required=False, help="公共空间 polygon 数据路径（可选）")
    parser.add_argument("--layer_shop", required=False, help="店铺 gpkg 图层名（可选）")
    parser.add_argument("--layer_public", required=False, help="公共空间 gpkg 图层名（可选）")
    parser.add_argument(
        "--func_field",
        required=False,
        default=None,
        help="功能字段名（例如 landuse/func），不存在则 func_type 全部为 0",
    )
    parser.add_argument(
        "--d_thresh",
        required=False,
        type=float,
        default=0.0,
        help="近邻连边质心距离阈值，默认 0（表示不使用近邻边）",
    )
    parser.add_argument(
        "--adjacency_only",
        required=False,
        type=int,
        choices=[0, 1],
        default=0,
        help="1: 只用面邻接连边；0: 可用近邻补边",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="输出目录，将生成 nodes.csv 与 edges.csv",
    )
    return parser.parse_args()


def read_layer(path: Path, layer: Optional[str]) -> gpd.GeoDataFrame:
    """读取矢量图层（支持 gpkg 的 layer 指定）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}")
    if path.suffix.lower() == ".gpkg" and layer:
        return gpd.read_file(path, layer=layer)
    return gpd.read_file(path)


def ensure_valid_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """删除空几何并尝试 buffer(0) 修复无效几何。"""
    gdf = gdf[gdf.geometry.notnull()].copy()
    if gdf.empty:
        raise ValueError("所有几何为空，请检查输入数据。")
    # 尝试修复无效几何
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")
        gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[gdf.geometry.notnull()].copy()
    if gdf.empty:
        raise ValueError("几何修复后仍为空，请检查输入数据。")
    return gdf


def warn_if_geographic(crs):
    """若为经纬度坐标系给出警告。"""
    if crs is None:
        warnings.warn("输入数据无 CRS 信息，几何度量结果将按原坐标单位计算。", UserWarning)
        return
    crs_str = str(crs).lower()
    try:
        epsg = crs.to_epsg()
    except Exception:
        epsg = None
    if epsg == 4326 or "4326" in crs_str:
        warnings.warn(
            "检测到坐标系为 EPSG:4326（经纬度）。"
            " 建议在外部重投影到米制坐标（如 EPSG:3857/4547 等）后再运行本脚本，"
            "否则面积/长度等度量将以“度”为单位，仅供相对比较。",
            UserWarning,
        )


def compute_shape_features(poly) -> dict:
    """根据要求计算单个多边形的形状特征。"""
    # 统一为 Polygon / MultiPolygon 已在 gdf 层保证
    area = float(poly.area)
    perimeter = float(poly.length)

    # 1) compactness
    if perimeter > 0:
        compactness = 4.0 * math.pi * area / (perimeter ** 2)
    else:
        compactness = 0.0

    # 2) concavity
    hull = poly.convex_hull
    hull_area = float(hull.area)
    if hull_area > 0:
        concavity = 1.0 - area / hull_area
    else:
        concavity = 0.0

    # 3) fractal_degree = log(P) / log(A)
    if area > 1.0 and perimeter > 1.0:
        fractal_degree = float(math.log(perimeter) / math.log(area))
    else:
        fractal_degree = 0.0

    # 4) extensibility & main_axis_dir 基于最小外接旋转矩形
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    if len(coords) < 4:
        extensibility = 0.0
        main_axis_dir = 0.0
    else:
        # 取连续边的长度
        edge_lengths = []
        edge_dirs = []
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length <= 0:
                continue
            edge_lengths.append(length)
            angle_deg = math.degrees(math.atan2(dy, dx))
            edge_dirs.append(angle_deg)

        if not edge_lengths:
            extensibility = 0.0
            main_axis_dir = 0.0
        else:
            # 矩形边只有两种长度，取最大/最小
            unique_lengths = sorted(set(round(l, 6) for l in edge_lengths))
            if len(unique_lengths) == 1:
                long_side = short_side = unique_lengths[0]
            else:
                short_side = unique_lengths[0]
                long_side = unique_lengths[-1]
            if short_side > 0:
                extensibility = float(long_side / short_side)
            else:
                extensibility = 0.0

            # 取最长边的方向角，归一化到 [0, 180)
            max_len = max(edge_lengths)
            idx = edge_lengths.index(max_len)
            angle = edge_dirs[idx] % 180.0
            main_axis_dir = float(angle)

    return dict(
        area=area,
        perimeter=perimeter,
        compactness=compactness,
        concavity=concavity,
        fractal_degree=fractal_degree,
        extensibility=extensibility,
        main_axis_dir=main_axis_dir,
    )


def compute_street_length(poly) -> float:
    """沿街长度：用多边形周长作为 street_length。"""
    if poly is None or poly.is_empty:
        return 0.0
    return float(poly.length)


def build_adjacency_edges(gdf: gpd.GeoDataFrame) -> set[tuple[int, int]]:
    """基于多边形邻接（touches / overlaps / intersects）构建无向边，使用空间索引加速。"""
    edges: set[tuple[int, int]] = set()
    sindex = gdf.sindex
    geometries = gdf.geometry.values

    for i, geom in enumerate(geometries):
        if geom is None or geom.is_empty:
            continue
        # bbox 查询候选
        candidate_idx = list(sindex.intersection(geom.bounds))
        for j in candidate_idx:
            if j <= i:
                continue
            geom_j = geometries[j]
            if geom_j is None or geom_j.is_empty:
                continue
            # 先快速排除，不相交则跳过
            if not geom.envelope.intersects(geom_j.envelope):
                continue
            # touches / overlaps / intersects 任一视为邻接（与 notebook 一致）
            if geom.touches(geom_j) or geom.overlaps(geom_j) or geom.intersects(geom_j):
                edges.add((i, j))
    return edges


def build_knn_edges(
    gdf: gpd.GeoDataFrame,
    base_edges: set[tuple[int, int]],
    d_thresh: float,
) -> set[tuple[int, int]]:
    """在已有邻接边基础上，用质心距离 < d_thresh 补充近邻边，避免孤立节点。"""
    if d_thresh <= 0:
        return base_edges

    centroids = gdf.geometry.centroid
    coords = np.array([[p.x, p.y] for p in centroids])
    sindex = centroids.sindex
    edges = set(base_edges)

    for i, (xi, yi) in enumerate(coords):
        # 若已在图中有至少一条边，可选是否跳过补边（这里仍允许补充更多边）
        # bbox 扩展 d_thresh
        minx = xi - d_thresh
        maxx = xi + d_thresh
        miny = yi - d_thresh
        maxy = yi + d_thresh
        candidate_idx = list(sindex.intersection((minx, miny, maxx, maxy)))
        for j in candidate_idx:
            if j <= i:
                continue
            xj, yj = coords[j]
            dist = math.hypot(xj - xi, yj - yi)
            if dist < d_thresh:
                edges.add((i, j))

    return edges


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读取 shop / public，并合并
    shop_gdf = read_layer(Path(args.shop), args.layer_shop)
    shop_gdf = ensure_valid_geometries(shop_gdf)

    if args.public:
        public_gdf = read_layer(Path(args.public), args.layer_public)
        public_gdf = ensure_valid_geometries(public_gdf)
    else:
        # 若未提供 public，则直接复用 shop 数据
        public_gdf = shop_gdf.copy()

    # 对于两者 CRS 不一致的情况，统一到 shop 的 CRS
    if public_gdf.crs != shop_gdf.crs:
        public_gdf = public_gdf.to_crs(shop_gdf.crs)

    warn_if_geographic(shop_gdf.crs)

    shop_gdf = shop_gdf.copy()
    shop_gdf["__geom_type__"] = "shop"
    public_gdf = public_gdf.copy()
    public_gdf["__geom_type__"] = "public"

    gdf = pd.concat([shop_gdf, public_gdf], ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=shop_gdf.crs)
    gdf = ensure_valid_geometries(gdf)

    # 2) 分配 node_id
    gdf = gdf.reset_index(drop=True)
    gdf["node_id"] = gdf.index.astype(int)
    n_nodes = len(gdf)

    # 3) 功能类型编码 func_type
    if args.func_field and args.func_field in gdf.columns:
        vals = gdf[args.func_field].astype("category")
        gdf["func_type"] = vals.cat.codes.astype(int)
    else:
        if args.func_field and args.func_field not in gdf.columns:
            warnings.warn(
                f"func_field='{args.func_field}' 不存在，所有 func_type 将设为 0。",
                UserWarning,
            )
        gdf["func_type"] = 0

    # 4) 计算几何特征 + 街长 + 质心
    shape_feats = []
    street_lengths = []
    centroids = gdf.geometry.centroid

    for geom in gdf.geometry:
        feats = compute_shape_features(geom)
        shape_feats.append(feats)
        street_len = compute_street_length(geom)
        street_lengths.append(street_len)

    shape_df = pd.DataFrame(shape_feats)
    gdf["x"] = centroids.x.astype(float)
    gdf["y"] = centroids.y.astype(float)
    gdf["compactness"] = shape_df["compactness"]
    gdf["extensibility"] = shape_df["extensibility"]
    gdf["concavity"] = shape_df["concavity"]
    gdf["fractal_degree"] = shape_df["fractal_degree"]
    gdf["main_axis_dir"] = shape_df["main_axis_dir"]
    gdf["street_length"] = street_lengths

    # 5) 构建边：先面邻接，再按需要补近邻
    adj_edges = build_adjacency_edges(gdf)
    if args.adjacency_only == 0 and args.d_thresh > 0:
        edges = build_knn_edges(gdf, adj_edges, d_thresh=args.d_thresh)
    else:
        edges = adj_edges

    # 确保 u < v
    edges = {(int(min(u, v)), int(max(u, v))) for (u, v) in edges}
    edges_list = sorted(list(edges))

    # 6) 计算中心性指标（无向图）
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(edges_list)

    # 若完全无边，NetworkX 仍可计算，但 closeness 全为 0
    try:
        closeness_dict = nx.closeness_centrality(G)
    except Exception as e:
        warnings.warn(f"计算 closeness_centrality 失败，全部设为 0：{e}", UserWarning)
        closeness_dict = {i: 0.0 for i in range(n_nodes)}

    try:
        betweenness_dict = nx.betweenness_centrality(G, normalized=True)
    except Exception as e:
        warnings.warn(f"计算 betweenness_centrality 失败，全部设为 0：{e}", UserWarning)
        betweenness_dict = {i: 0.0 for i in range(n_nodes)}

    gdf["closeness"] = gdf["node_id"].map(closeness_dict).astype(float)
    gdf["betweenness"] = gdf["node_id"].map(betweenness_dict).astype(float)

    # 7) 生成 nodes.csv
    nodes_df = pd.DataFrame(
        {
            "node_id": gdf["node_id"].astype(int),
            "func_type": gdf["func_type"].astype(int),
            "x": gdf["x"].astype(float),
            "y": gdf["y"].astype(float),
            "compactness": gdf["compactness"].astype(float),
            "extensibility": gdf["extensibility"].astype(float),
            "concavity": gdf["concavity"].astype(float),
            "fractal_degree": gdf["fractal_degree"].astype(float),
            "main_axis_dir": gdf["main_axis_dir"].astype(float),
            "street_length": gdf["street_length"].astype(float),
            "closeness": gdf["closeness"].astype(float),
            "betweenness": gdf["betweenness"].astype(float),
            # flow 先留空，后续填充
            "flow": np.nan,
        }
    )

    nodes_path = out_dir / "nodes.csv"
    nodes_df.to_csv(nodes_path, index=False)

    # 8) 生成 edges.csv
    # open=1: shop-public_space, public_space-shop, public_space-public_space
    # open=0: shop-shop
    node_type = gdf["__geom_type__"].values  # "shop" or "public"
    edges_df = pd.DataFrame(edges_list, columns=["u", "v"])

    u_types = node_type[edges_df["u"].to_numpy()]
    v_types = node_type[edges_df["v"].to_numpy()]
    shop_shop = (u_types == "shop") & (v_types == "shop")
    edges_df["open"] = np.where(shop_shop, 0, 1)
    edges_path = out_dir / "edges.csv"
    edges_df.to_csv(edges_path, index=False)

    # 9) 打印统计信息
    n_edges = len(edges_list)
    n_open = int(edges_df["open"].sum())
    n_components = nx.number_connected_components(G)
    print(f"节点数: {n_nodes}")
    print(f"边数: {n_edges} (open=1: {n_open}, open=0: {n_edges - n_open})")
    print(f"连通分量数: {n_components}")
    print(f"nodes.csv 已写入: {nodes_path}")
    print(f"edges.csv 已写入: {edges_path}")


if __name__ == "__main__":
    main()
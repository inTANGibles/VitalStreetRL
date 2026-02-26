#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 0222_flow_mapped_public.geojson 生成 flows.csv，与 0222.geojson 生成的 nodes.csv 对应。
通过几何匹配：0222 中每个 public 节点与 flow_mapped 中最近质心的 feature 匹配，取其 flow。
若 flow_geojson 含 node_id 列，则 flows.csv 的 node_id 使用该列（与 flow_mapping.ipynb、nodes.csv 一致）；
否则退化为 0222 合并 gdf 的行号。
nodes.csv/edges.csv 由 build_nodes_edges --geojson 0222.geojson 生成，node_id 为 0222 行号 0..N-1。
"""
import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="从 flow_mapped 公共空间 GeoJSON 生成 flows.csv，与 0222 节点对应")
    p.add_argument("--geojson", required=True, help="0222.geojson，用于得到与 nodes 一致的 public 几何与顺序")
    p.add_argument("--flow_geojson", required=True, help="0222_flow_mapped_public.geojson，含 flow_5min 等")
    p.add_argument("--out_dir", required=True, help="输出 flows.csv 的目录")
    p.add_argument("--flow_field", default="flow_5min", choices=["flow_5min", "pass_count", "flow_prediction"])
    return p.parse_args()


def ensure_valid_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf[gdf.geometry.notnull()].copy()
    if gdf.empty:
        return gdf
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")
        gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf[gdf.geometry.notnull()].copy()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 0222：与 build_nodes_edges 相同的 gdf 顺序（shop + public）
    gdf_full = gpd.read_file(args.geojson)
    if "unit_type" not in gdf_full.columns:
        raise ValueError("0222.geojson 需含 unit_type 列")
    gdf_full = ensure_valid_geometries(gdf_full)
    shop_gdf = gdf_full[gdf_full["unit_type"] == "shop"].copy()
    public_gdf = gdf_full[gdf_full["unit_type"] == "public_space"].copy()
    if len(shop_gdf) == 0:
        shop_gdf = gdf_full.head(0).copy()
    if len(public_gdf) == 0:
        public_gdf = gdf_full.head(0).copy()
    shop_gdf["__type__"] = "shop"
    public_gdf["__type__"] = "public"
    gdf = pd.concat([shop_gdf, public_gdf], ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=gdf_full.crs)
    gdf = gdf.reset_index(drop=True)
    # node_id 与 nodes.csv 一致：行号 0..N-1
    public_indices = gdf.index[gdf["__type__"] == "public"].tolist()

    # 2. flow_mapped 公共空间
    gdf_flow = gpd.read_file(args.flow_geojson)
    if gdf_flow.crs != gdf.crs:
        gdf_flow = gdf_flow.to_crs(gdf.crs)
    gdf_flow = ensure_valid_geometries(gdf_flow)
    if "unit_type" in gdf_flow.columns:
        gdf_flow = gdf_flow[gdf_flow["unit_type"] == "public_space"].copy()
    flow_col = args.flow_field
    if flow_col not in gdf_flow.columns:
        flow_col = "flow_5min" if "flow_5min" in gdf_flow.columns else "pass_count"

    # 3. 为每个 0222 的 public 节点匹配 flow_mapped 中的 feature（质心最近），用 flow_mapped 的 node_id 写入，与 flow_mapping.ipynb / nodes.csv 一致
    use_flow_node_id = "node_id" in gdf_flow.columns
    flows_by_node = {}  # node_id (from flow_mapped if present, else gdf row index) -> flow
    for idx in public_indices:
        geom = gdf.geometry.iloc[idx]
        if geom is None or (getattr(geom, "is_empty", False)):
            if not use_flow_node_id:
                flows_by_node[idx] = 0.0
            continue
        centroid = geom.centroid
        best_j = None
        best_dist = float("inf")
        for j, row in gdf_flow.iterrows():
            g = row.geometry
            if g is None or (getattr(g, "is_empty", False)):
                continue
            try:
                d = centroid.distance(g.centroid)
                if d < best_dist:
                    best_dist = d
                    best_j = j
            except Exception:
                continue
        if best_j is not None:
            val = gdf_flow.loc[best_j, flow_col]
            flow_val = float(val) if pd.notna(val) else 0.0
            out_id = int(gdf_flow.loc[best_j, "node_id"]) if use_flow_node_id else idx
            flows_by_node[out_id] = flow_val
        else:
            if not use_flow_node_id:
                flows_by_node[idx] = 0.0

    # 4. 将原始 flow 映射到 [1,10]（非零），写出 flows.csv：仅 public 节点，单时刻 day=0, hour=0
    raw_values = [flow for flow in flows_by_node.values() if flow > 0.0]
    if raw_values:
        fmin = float(min(raw_values))
        fmax = float(max(raw_values))
    else:
        fmin, fmax = 0.0, 1.0

    def to_1_10(flow: float) -> float:
        """与 build_graph.py 中 linear_1_10 一致：非零 flow 线性映射到 [1,10]，零保持 0。"""
        if flow <= 0.0:
            return 0.0
        if fmax <= fmin:
            return 1.0
        t = (flow - fmin) / (fmax - fmin)
        t = max(0.0, min(1.0, t))
        return 1.0 + t * 9.0

    rows = [
        {
            "node_id": node_id,
            "day": 0,
            "hour": 0,
            "flow": to_1_10(max(0.0, flow)),
        }
        for node_id, flow in flows_by_node.items()
    ]
    flows_path = out_dir / "flows.csv"
    pd.DataFrame(rows).to_csv(flows_path, index=False)
    print(f"flows.csv -> {flows_path}（{len(rows)} 个 public 节点）")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 0222_flow_mapped_public.geojson 生成 flows.csv，与 0222.geojson 生成的 nodes.csv 对应。
通过几何匹配：0222 中每个 public 节点与 flow_mapped 中最近/重叠的 feature 匹配，取其 flow。
nodes.csv/edges.csv 由 build_nodes_edges --geojson 0222.geojson 生成，node_id 为行号 0..N-1。
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

    # 3. 为每个 0222 的 public 节点匹配 flow_mapped 中的 feature（质心最近）
    flows_by_node = {}
    for idx in public_indices:
        geom = gdf.geometry.iloc[idx]
        if geom is None or (getattr(geom, "is_empty", False)):
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
            flows_by_node[idx] = float(val) if pd.notna(val) else 0.0
        else:
            flows_by_node[idx] = 0.0

    # 4. 写出 flows.csv：仅 public 节点，单时刻 day=0, hour=0
    rows = [{"node_id": node_id, "day": 0, "hour": 0, "flow": max(0.0, flow)} for node_id, flow in flows_by_node.items()]
    flows_path = out_dir / "flows.csv"
    pd.DataFrame(rows).to_csv(flows_path, index=False)
    print(f"flows.csv -> {flows_path}（{len(rows)} 个 public 节点）")


if __name__ == "__main__":
    main()

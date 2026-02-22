# -*- coding: utf-8 -*-
"""
Flow 客流可视化：输入 flow 文件，输出原始客流与还原客流的 choropleth 图。
用于在 notebook 中批量可视化多天/一周的 flow。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Point


# ---------- 图与邻接 ----------
def build_adjacency_edges(gdf, only_public=False, buffer_eps=0):
    """空间单元邻接（touches/intersects）。only_public 时仅 public_space；buffer_eps>0 时用 buffer 吸收缝隙。"""
    edges = set()
    geoms = gdf.geometry.values
    sindex = gdf.sindex
    unit_type = gdf["unit_type"].values if only_public else None
    for i, geom in enumerate(geoms):
        if geom is None or geom.is_empty:
            continue
        if only_public and unit_type[i] != "public_space":
            continue
        geom_check = geom.buffer(buffer_eps) if (only_public and buffer_eps > 0) else geom
        for j in sindex.intersection(geom_check.bounds):
            if j <= i:
                continue
            if only_public and unit_type[j] != "public_space":
                continue
            gj = geoms[j]
            if gj is None or gj.is_empty:
                continue
            gj_check = gj.buffer(buffer_eps) if (only_public and buffer_eps > 0) else gj
            if geom_check.touches(gj_check) or geom_check.intersects(gj_check):
                edges.add((i, j))
    return edges


def all_pairs_shortest_paths(G, weight="weight"):
    """全源最短路径，返回 {(i,j): [length, path_list]}，path 为 node_id 列表。"""
    paths = {}
    for i in G.nodes():
        for j in G.nodes():
            if i >= j:
                continue
            try:
                path = nx.shortest_path(G, i, j, weight=weight)
                length = nx.shortest_path_length(G, i, j, weight=weight)
                paths[(i, j)] = [float(length), [int(x) for x in path]]
            except nx.NetworkXNoPath:
                pass
    return paths


def path_key(i, j):
    return tuple(sorted((i, j)))


# ---------- 上下文：一次加载 GeoJSON + 探针，构建图与路径 ----------
def load_context(
    geojson_path: Path | str,
    devices_axis_path: Path | str,
    buffer_eps: float = 1.0,
):
    """
    加载空间底图与探针映射，构建 public_space 邻接图与最短路径。
    返回的 ctx 用于多次调用 compute_flow_and_plot，避免重复建图。
    """
    geojson_path = Path(geojson_path)
    devices_axis_path = Path(devices_axis_path)

    gdf = gpd.read_file(geojson_path)
    gdf = gdf[gdf.geometry.notnull()].copy().reset_index(drop=True)
    gdf["node_id"] = gdf.index.astype(int)
    centroids = gdf.geometry.centroid
    gdf["cx"] = centroids.x.astype(float)
    gdf["cy"] = centroids.y.astype(float)
    is_public = (gdf["unit_type"] == "public_space").values
    n_spaces = len(gdf)

    adj_edges = build_adjacency_edges(gdf, only_public=True, buffer_eps=buffer_eps)
    G = nx.Graph()
    G.add_nodes_from(range(n_spaces))
    coords = gdf[["cx", "cy"]].values
    for (i, j) in adj_edges:
        d = np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
        G.add_edge(i, j, weight=d)
    space_shortest_paths = all_pairs_shortest_paths(G)

    def get_path_between_spaces(si, sj):
        key = path_key(si, sj)
        if key in space_shortest_paths:
            return space_shortest_paths[key][1]
        if si == sj:
            return [si]
        return None

    if devices_axis_path.exists():
        with open(devices_axis_path, encoding="utf-8") as f:
            devices_axis = {int(k): v for k, v in json.load(f).items()}
    else:
        devices_axis = {i: [gdf.at[i, "cx"], gdf.at[i, "cy"]] for i in range(min(20, n_spaces))}

    probe_to_space = {}
    pts = gpd.GeoDataFrame(
        {"probe_id": list(devices_axis.keys())},
        geometry=[Point(devices_axis[d][0], devices_axis[d][1]) for d in devices_axis],
        crs=gdf.crs,
    )
    joined = gpd.sjoin(pts, gdf[["node_id", "geometry"]], how="left", predicate="within")
    for _, row in joined.iterrows():
        pid, nid = row["probe_id"], row["node_id"]
        if pd.notna(nid):
            probe_to_space[pid] = int(nid)
    spaces_with_probe = set(probe_to_space.values())

    class Ctx:
        pass

    ctx = Ctx()
    ctx.gdf = gdf
    ctx.is_public = is_public
    ctx.n_spaces = n_spaces
    ctx.probe_to_space = probe_to_space
    ctx.spaces_with_probe = spaces_with_probe
    ctx.get_path_between_spaces = get_path_between_spaces
    return ctx


# ---------- 单次 flow 计算与绘图 ----------
INTERVAL_FLOW = 300  # 5 分钟
CMAP_FLOW = LinearSegmentedColormap.from_list(
    "ylorrd_0white",
    ["white", "#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026"],
    N=256,
)


def compute_flow_and_plot(
    ctx,
    flow_json_path: Path | str,
    output_dir: Optional[Path | str] = None,
    title_suffix: str = "",
    dpi: int = 150,
):
    """
    读取 flow 文件，计算 flow_5min（还原）与 flow_5min_direct（原始），
    绘制两张 choropleth 图；若 output_dir 给定则保存为 PNG。

    flow 格式：每行一个 JSON，{"m": [...], "a": 探针号, "t": unix 秒}。

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        带 flow_5min、flow_5min_direct 列的 gdf（仅当有记录时写入）。
    """
    flow_json_path = Path(flow_json_path)
    gdf = ctx.gdf.copy()
    is_public = ctx.is_public
    n_spaces = ctx.n_spaces
    probe_to_space = ctx.probe_to_space
    get_path = ctx.get_path_between_spaces

    if flow_json_path.exists():
        with open(flow_json_path, encoding="utf-8") as f:
            records_flow = [json.loads(line) for line in f if line.strip()]
    else:
        records_flow = []

    flow_5min_count = defaultdict(float)
    flow_5min_direct_count = defaultdict(float)
    for mac in set(tuple(r["m"]) for r in records_flow):
        mac_records = [r for r in records_flow if tuple(r["m"]) == mac]
        if len(mac_records) < 2:
            continue
        mac_records.sort(key=lambda r: r["t"])
        for k in range(len(mac_records) - 1):
            t1, t2 = mac_records[k]["t"], mac_records[k + 1]["t"]
            a1, a2 = mac_records[k]["a"], mac_records[k + 1]["a"]
            if a1 not in probe_to_space or a2 not in probe_to_space:
                continue
            s1, s2 = probe_to_space[a1], probe_to_space[a2]
            flow_5min_direct_count[s1] += 1.0
            flow_5min_direct_count[s2] += 1.0
            path = get_path(s1, s2)
            if not path:
                continue
            n_slots = max(1, int((t2 - t1) / INTERVAL_FLOW))
            weight_per_node = n_slots / len(path)
            for node_id in path:
                if is_public[node_id]:
                    flow_5min_count[node_id] += weight_per_node

    gdf["flow_5min"] = [flow_5min_count.get(i, 0) for i in range(n_spaces)]
    gdf["flow_5min_direct"] = [flow_5min_direct_count.get(i, 0) for i in range(n_spaces)]

    gdf_public = gdf[is_public]
    suffix = f" {title_suffix}" if title_suffix else ""

    # 图1：还原客流（全路径 5min 均分）
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 10))
    gdf[~is_public].plot(ax=ax1, facecolor="lightgray", edgecolor="gray", linewidth=0.3)
    gdf_public.plot(ax=ax1, column="flow_5min", legend=True, cmap=CMAP_FLOW, vmin=0, edgecolor="gray", linewidth=0.3)
    ax1.scatter(gdf_public["cx"], gdf_public["cy"], s=15, c="black", alpha=0.5)
    ax1.set_title(f"还原客流（5min 颗粒 全路径）{suffix}")
    ax1.set_aspect("equal")
    plt.tight_layout()
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = flow_json_path.stem
        fig1.savefig(out_dir / f"{base}_reconstructed.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig1)

    # 图2：原始客流（仅起止点）
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))
    gdf[~is_public].plot(ax=ax2, facecolor="lightgray", edgecolor="gray", linewidth=0.3)
    gdf_public.plot(ax=ax2, column="flow_5min_direct", legend=True, cmap=CMAP_FLOW, vmin=0, edgecolor="gray", linewidth=0.3)
    ax2.scatter(gdf_public["cx"], gdf_public["cy"], s=15, c="black", alpha=0.5)
    ax2.set_title(f"原始客流（仅探针起止点）{suffix}")
    ax2.set_aspect("equal")
    plt.tight_layout()
    if output_dir:
        out_dir = Path(output_dir)
        fig2.savefig(out_dir / f"{base}_raw.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig2)

    return gdf


# ---------- 命令行 / 单文件 ----------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Flow 客流可视化：输入 flow 文件，输出原始/还原客流图")
    parser.add_argument("flow_json", type=Path, help="flow 文件路径（每行 JSON: m, a, t）")
    parser.add_argument("--geojson", type=Path, default=None, help="空间 GeoJSON（默认：flow 同目录上级的 data/0222.geojson）")
    parser.add_argument("--devices-axis", type=Path, default=None, help="devices_axis.json（默认：flow 同目录下 devices_axis.json）")
    parser.add_argument("--out-dir", "-o", type=Path, default=None, help="输出目录，不指定则不保存")
    parser.add_argument("--title", type=str, default="", help="图标题后缀")
    args = parser.parse_args()

    base = args.flow_json.resolve().parent
    geojson = args.geojson or (base.parent.parent / "0222.geojson")
    devices_axis = args.devices_axis or (base / "devices_axis.json")
    if not geojson.exists():
        geojson = base.parent.parent / "data" / "0222.geojson"
    if not devices_axis.exists():
        devices_axis = base.parent.parent / "data" / "FlowData" / "0222" / "devices_axis.json"

    ctx = load_context(geojson, devices_axis)
    compute_flow_and_plot(ctx, args.flow_json, output_dir=args.out_dir, title_suffix=args.title)
    print("Done.")


if __name__ == "__main__":
    main()

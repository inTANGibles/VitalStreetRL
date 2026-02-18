#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_flow_15min.py

从 clean_*.json 构建 15 分钟时间片客流，输出 flows.csv。
- 时间片：12–17 时，每 15 分钟，共 20 个 slot/天
- 去重：同一设备、同一 MAC、同一 15 分钟只计 1 人次
- 空间映射：设备 sjoin 到 street，uid 映射到 node_id（与 build_nodes_edges 顺序一致）
- 标签：Box-Cox + 线性映射到 1–10（每个时间片内独立映射）
- 输出：node_id, day, slot_idx, flow（1–10）
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

# 直接导入 remap_flow，避免 geo 包的链式依赖
_THIS = Path(__file__).resolve().parent
_PROJECT = _THIS.parent.parent.parent  # VitalStreetRL
_GEO_TOOL = _PROJECT / "geo" / "tool"
if str(_GEO_TOOL) not in sys.path:
    sys.path.insert(0, str(_GEO_TOOL))
from remap_flow import remap_flow

HOUR_START = 12
HOUR_END = 17  # 不含 17，即 12:00–16:59
SLOTS_PER_DAY = 20  # 12:00, 12:15, ..., 16:45


def build_gdf_same_order_as_nodes(street_path: Path) -> gpd.GeoDataFrame:
    """与 build_nodes_edges 相同顺序：shop 在前，public_space 在后。"""
    gdf = gpd.read_file(street_path)
    if "unit_type" not in gdf.columns:
        raise ValueError("street.geojson 需包含 unit_type 列")
    shop = gdf[gdf["unit_type"] == "shop"].copy()
    public = gdf[gdf["unit_type"] == "public_space"].copy()
    merged = pd.concat([shop, public], ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=gdf.crs)


def compute_uid_flow_per_slot(
    flow_path: Path,
    devices_path: Path,
    street_path: Path,
    day_idx: int,
) -> dict:
    """
    单日单文件：按 (day_idx, slot_idx) 计算 uid -> flow。
    去重：(device_id, mac, slot_idx) 只计 1 人次。
    返回 {(day_idx, slot_idx): {uid: count}}
    """
    with open(devices_path, encoding="utf-8") as f:
        devices = {int(k): v for k, v in json.load(f).items()}

    with open(flow_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # 时段 12–17，slot_idx = (hour-12)*4 + minute//15
    device_mac_slot = set()
    for r in records:
        try:
            dt = datetime.fromtimestamp(r["t"])
            if dt.hour < HOUR_START or dt.hour >= HOUR_END:
                continue
            slot_idx = (dt.hour - HOUR_START) * 4 + dt.minute // 15
            device_mac_slot.add((r["a"], tuple(r["m"]), slot_idx))
        except (KeyError, OSError, OverflowError):
            continue

    device_flow_slot = defaultdict(lambda: defaultdict(int))
    for did, _, slot_idx in device_mac_slot:
        device_flow_slot[slot_idx][did] += 1

    gdf = gpd.read_file(street_path)
    uid_col = "uid" if "uid" in gdf.columns else "index"
    out = {}
    for slot_idx in range(SLOTS_PER_DAY):
        device_flow = device_flow_slot.get(slot_idx, {})
        pts = gpd.GeoDataFrame(
            {
                "device_id": list(devices.keys()),
                "flow": [device_flow.get(d, 0) for d in devices.keys()],
            },
            geometry=[Point(devices[d][0], devices[d][1]) for d in devices.keys()],
            crs=gdf.crs,
        )
        joined = gpd.sjoin(
            pts, gdf[[uid_col, "geometry"]], how="left", predicate="within"
        )
        joined_valid = joined[joined[uid_col].notna()]
        uid_flow = joined_valid.groupby(uid_col, observed=True)["flow"].sum().to_dict()
        out[(day_idx, slot_idx)] = uid_flow
    return out


def parse_day_from_filename(path: Path, day_offset: int) -> int:
    """clean_N.json -> day_idx = N - day_offset"""
    m = re.search(r"clean_(\d+)\.json", path.name, re.I)
    if not m:
        raise ValueError(f"无法从文件名解析日期: {path.name}")
    return int(m.group(1)) - day_offset


def main():
    p = argparse.ArgumentParser(description="从 clean_*.json 构建 15 分钟 flows.csv")
    p.add_argument("--flow_dir", required=True, help="Flow 目录（含 clean_*.json）")
    p.add_argument("--devices", required=True, help="devices_axis.json")
    p.add_argument("--street", required=True, help="street.geojson")
    p.add_argument("--nodes", required=True, help="nodes.csv（用于 uid->node_id）")
    p.add_argument("--out", required=True, help="输出 flows.csv")
    p.add_argument("--day-offset", type=int, default=19, help="clean_N -> day = N - offset")
    args = p.parse_args()

    flow_dir = Path(args.flow_dir)
    flow_paths = sorted(flow_dir.glob("clean_*.json"))
    if not flow_paths:
        raise FileNotFoundError(f"目录下无 clean_*.json: {flow_dir}")

    gdf_build = build_gdf_same_order_as_nodes(Path(args.street))
    uid_to_node_id = {uid: i for i, uid in enumerate(gdf_build["uid"].values)}

    all_rows = []
    for fp in flow_paths:
        day_idx = parse_day_from_filename(fp, args.day_offset)
        uid_flow_per_slot = compute_uid_flow_per_slot(
            fp, Path(args.devices), Path(args.street), day_idx
        )
        for (d, s), uid_flow in uid_flow_per_slot.items():
            for uid, count in uid_flow.items():
                if uid not in uid_to_node_id:
                    continue
                node_id = uid_to_node_id[uid]
                all_rows.append({"node_id": node_id, "day": d, "slot_idx": s, "flow_raw": count})

    if not all_rows:
        raise ValueError("无有效客流记录")

    df = pd.DataFrame(all_rows)
    df_agg = df.groupby(["node_id", "day", "slot_idx"], as_index=False)["flow_raw"].sum()

    # 每个 (day, slot_idx) 内独立 remap
    flow_mapped = []
    for (day, slot_idx), grp in df_agg.groupby(["day", "slot_idx"]):
        raw = grp["flow_raw"].values
        mapped = remap_flow(raw, group_ids=None)
        flow_mapped.extend(mapped)

    df_agg["flow"] = flow_mapped
    out_df = df_agg[["node_id", "day", "slot_idx", "flow"]]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    n_slots = out_df[["day", "slot_idx"]].drop_duplicates().shape[0]
    n_labeled = out_df["node_id"].nunique()
    print(f"已保存 flows.csv: {out_path} ({len(out_df)} 行)")
    print(f"  时间片: {n_slots} 个 (day, slot_idx)")
    print(f"  有标签节点: {n_labeled} 个")


if __name__ == "__main__":
    main()

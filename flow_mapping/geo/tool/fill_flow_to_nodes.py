#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fill_flow_to_nodes.py

参考 flow_mapping.ipynb 的人流映射逻辑，用客流数据填充 nodes.csv 的 flow 列。
- 读取 Flow/clean_*.json、Axis/devices_axis.json、street.geojson
- 按 (device_id, mac, interval) 去重统计设备客流，默认 15 分钟间隔
- 可选过滤时段（如 12-18 时）
- 支持单日或目录下多日聚合（sum）
- 可选：两阶段重映射（Box-Cox λ=1/6 + 线性映射到 1-10）并输出 flow.csv
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

# 导入重映射模块
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from geo.tool.remap_flow import remap_flow  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="用客流数据填充 nodes.csv 的 flow 列，可选输出 flow.csv"
    )
    parser.add_argument(
        "--flow",
        required=False,
        help="单日客流 JSON 路径（如 Flow/clean_19.json）",
    )
    parser.add_argument(
        "--flow_dir",
        required=False,
        help="客流 JSON 目录，将聚合所有 clean_*.json（与 --flow 二选一）",
    )
    parser.add_argument(
        "--aggregate",
        choices=["sum", "mean"],
        default="sum",
        help="多日聚合方式，默认 sum",
    )
    parser.add_argument(
        "--devices",
        required=True,
        help="设备坐标 JSON 路径（如 Axis/devices_axis.json）",
    )
    parser.add_argument(
        "--street",
        required=True,
        help="空间单元 GeoJSON 路径（如 street.geojson）",
    )
    parser.add_argument(
        "--nodes",
        required=True,
        help="nodes.csv 路径，将被更新",
    )
    parser.add_argument(
        "--remap",
        action="store_true",
        help="对 flow 进行两阶段重映射（Box-Cox + 线性映射 1-10）",
    )
    parser.add_argument(
        "--flow_csv",
        default=None,
        help="输出 flow.csv 路径，含 node_id, flow_raw, flow_remapped",
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=1.0 / 6.0,
        dest="lam",
        help="Box-Cox 变换参数，默认 1/6",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=900,
        help="去重时间间隔（秒），默认 900（15 分钟）",
    )
    parser.add_argument(
        "--hour-start",
        type=int,
        default=None,
        dest="hour_start",
        help="仅统计该小时及之后（0-23），默认不限制",
    )
    parser.add_argument(
        "--hour-end",
        type=int,
        default=None,
        dest="hour_end",
        help="仅统计该小时之前（不含），如 18 表示 12-17 时，默认不限制",
    )
    return parser.parse_args()


def compute_uid_flow_single(
    flow_path: Path,
    devices_path: Path,
    street_path: Path,
    interval_seconds: int = 900,
    hour_start: Optional[int] = None,
    hour_end: Optional[int] = None,
) -> dict:
    """
    按 flow_mapping.ipynb 逻辑计算单日 uid -> flow。
    去重规则：同一设备、同一 MAC、同一时间间隔（默认 15 分钟）只计 1 人次。
    可选：仅统计 hour_start <= hour < hour_end 的记录（本地时间）。

    返回 {uid: flow_count}。
    """
    with open(devices_path, encoding="utf-8") as f:
        devices = {int(k): v for k, v in json.load(f).items()}

    with open(flow_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # 时段过滤：仅保留 hour_start <= hour < hour_end
    if hour_start is not None or hour_end is not None:
        filtered = []
        for r in records:
            try:
                hour = datetime.fromtimestamp(r["t"]).hour
            except (KeyError, OSError, OverflowError):
                continue
            if hour_start is not None and hour < hour_start:
                continue
            if hour_end is not None and hour >= hour_end:
                continue
            filtered.append(r)
        records = filtered

    # (device_id, mac, interval_slot) 去重，interval_slot = t // interval_seconds
    device_mac_slot = {
        (r["a"], tuple(r["m"]), r["t"] // interval_seconds) for r in records
    }
    device_flow = defaultdict(int)
    for did, _, _ in device_mac_slot:
        device_flow[did] += 1

    gdf = gpd.read_file(street_path)
    pts = gpd.GeoDataFrame(
        {
            "device_id": list(devices.keys()),
            "flow": [device_flow.get(d, 0) for d in devices.keys()],
        },
        geometry=[Point(devices[d][0], devices[d][1]) for d in devices.keys()],
        crs=gdf.crs,
    )
    joined = gpd.sjoin(
        pts, gdf[["uid", "index", "geometry"]], how="left", predicate="within"
    )
    joined_valid = joined[joined["uid"].notna()]
    n_outside = len(joined) - len(joined_valid)
    if n_outside > 0:
        warnings.warn(
            f"有 {n_outside} 个探测点未落在任何空间图块内，已忽略",
            UserWarning,
        )
    uid_flow = joined_valid.groupby("uid", observed=True)["flow"].sum().to_dict()
    return uid_flow


def compute_uid_flow(
    flow_paths: list[Path],
    devices_path: Path,
    street_path: Path,
    aggregate: str = "sum",
    interval_seconds: int = 900,
    hour_start: Optional[int] = None,
    hour_end: Optional[int] = None,
) -> dict:
    """
    聚合多日 uid -> flow。aggregate: sum 或 mean。
    """
    if not flow_paths:
        raise ValueError("至少需要一个客流文件")
    all_uid_flow = []
    for fp in flow_paths:
        uid_flow = compute_uid_flow_single(
            fp,
            devices_path,
            street_path,
            interval_seconds=interval_seconds,
            hour_start=hour_start,
            hour_end=hour_end,
        )
        all_uid_flow.append(uid_flow)
    all_uids = set()
    for uf in all_uid_flow:
        all_uids.update(uf.keys())
    if aggregate == "sum":
        out = {uid: sum(uf.get(uid, 0) for uf in all_uid_flow) for uid in all_uids}
    else:
        out = {
            uid: np.mean([uf.get(uid, 0) for uf in all_uid_flow])
            for uid in all_uids
        }
    return out


def build_gdf_same_order_as_nodes(street_path: Path) -> gpd.GeoDataFrame:
    """
    构建与 build_nodes_edges 相同顺序的 gdf：shop 在前，public_space 在后。
    """
    gdf = gpd.read_file(street_path)
    if "unit_type" not in gdf.columns:
        raise ValueError("street.geojson 需包含 unit_type 列")
    shop = gdf[gdf["unit_type"] == "shop"].copy()
    public = gdf[gdf["unit_type"] == "public_space"].copy()
    merged = pd.concat([shop, public], ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=gdf.crs)


def main():
    args = parse_args()
    if args.flow and args.flow_dir:
        raise ValueError("--flow 与 --flow_dir 只能指定其一")
    if not args.flow and not args.flow_dir:
        raise ValueError("请指定 --flow 或 --flow_dir")

    devices_path = Path(args.devices)
    street_path = Path(args.street)
    nodes_path = Path(args.nodes)

    if args.flow:
        flow_paths = [Path(args.flow)]
    else:
        flow_dir = Path(args.flow_dir)
        flow_paths = sorted(flow_dir.glob("clean_*.json"))
        if not flow_paths:
            raise FileNotFoundError(f"目录下无 clean_*.json: {flow_dir}")

    for p in flow_paths + [devices_path, street_path, nodes_path]:
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {p}")

    uid_flow = compute_uid_flow(
        flow_paths,
        devices_path,
        street_path,
        aggregate=args.aggregate,
        interval_seconds=args.interval,
        hour_start=args.hour_start,
        hour_end=args.hour_end,
    )
    gdf_build = build_gdf_same_order_as_nodes(street_path)
    flow_by_node = gdf_build["uid"].map(uid_flow)

    nodes_df = pd.read_csv(nodes_path)
    if "flow" not in nodes_df.columns:
        nodes_df["flow"] = np.nan
    if len(flow_by_node) != len(nodes_df):
        raise ValueError(
            f"节点数不一致: gdf={len(flow_by_node)}, nodes.csv={len(nodes_df)}"
        )
    flow_raw = flow_by_node.values
    nodes_df["flow"] = flow_raw
    nodes_df.to_csv(nodes_path, index=False)

    n_labeled = nodes_df["flow"].notna().sum()
    names = ", ".join(p.name for p in flow_paths[:3])
    if len(flow_paths) > 3:
        names += f" ... (+{len(flow_paths)-3})"
    print(f"已用 {names} 填充 flow")
    print(f"  总节点: {len(nodes_df)}, 有标签节点: {n_labeled}")
    print(f"  已保存: {nodes_path}")

    # 可选：重映射并输出 flow.csv
    flow_csv_path = Path(args.flow_csv) if args.flow_csv else None
    if args.remap and flow_csv_path is None:
        flow_csv_path = nodes_path.parent / "flow.csv"
    if flow_csv_path is not None or args.remap:
        flow_raw_arr = np.array(flow_raw, dtype=float)
        flow_raw_arr = np.where(
            pd.isna(flow_raw_arr) | (flow_raw_arr < 0), np.nan, flow_raw_arr
        )
        flow_remapped = remap_flow(flow_raw_arr, lam=args.lam, group_ids=None)
        flow_df = pd.DataFrame(
            {
                "node_id": nodes_df["node_id"].values,
                "flow_raw": flow_raw_arr,
                "flow_remapped": flow_remapped,
            }
        )
        if flow_csv_path is not None:
            flow_csv_path = Path(flow_csv_path)
            flow_csv_path.parent.mkdir(parents=True, exist_ok=True)
            flow_df.to_csv(flow_csv_path, index=False)
            print(f"  已保存 flow.csv: {flow_csv_path}")


if __name__ == "__main__":
    main()

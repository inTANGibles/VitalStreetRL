#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_flow_csv.py

从客流数据生成 flow.csv（含两阶段重映射）。
调用 fill_flow_to_nodes 并启用 --remap 与 --flow_csv。
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="从客流数据生成 flow.csv（Box-Cox + 线性映射 1-10）"
    )
    parser.add_argument(
        "--flow_dir",
        default=None,
        help="客流 JSON 目录（默认: data/FlowData/Jul/Flow）",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help="设备坐标 JSON（默认: data/FlowData/Jul/Axis/devices_axis.json）",
    )
    parser.add_argument(
        "--street",
        default=None,
        help="空间单元 GeoJSON（默认: data/street.geojson）",
    )
    parser.add_argument(
        "--nodes",
        default=None,
        help="nodes.csv 路径（默认: data/FlowData/Jul/nodes.csv）",
    )
    parser.add_argument(
        "--flow_csv",
        default=None,
        help="输出 flow.csv 路径（默认: data/FlowData/Jul/flow.csv）",
    )
    parser.add_argument(
        "--aggregate",
        choices=["sum", "mean"],
        default="sum",
        help="多日聚合方式",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    flow_dir = Path(args.flow_dir or str(root / "data" / "FlowData" / "Jul" / "Flow"))
    devices = args.devices or str(root / "data" / "FlowData" / "Jul" / "Axis" / "devices_axis.json")
    street = args.street or str(root / "data" / "street.geojson")
    nodes = args.nodes or str(root / "data" / "FlowData" / "Jul" / "nodes.csv")
    flow_csv = args.flow_csv or str(root / "data" / "FlowData" / "Jul" / "flow.csv")

    script = root / "geo" / "tool" / "fill_flow_to_nodes.py"
    cmd = [
        sys.executable,
        str(script),
        "--flow_dir", str(flow_dir),
        "--devices", devices,
        "--street", street,
        "--nodes", nodes,
        "--flow_csv", flow_csv,
        "--remap",
        "--aggregate", args.aggregate,
    ]
    subprocess.run(cmd, cwd=str(root), check=True)
    print(f"flow.csv 已生成: {flow_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 data/FlowData/Jul/Flow 原始客流 JSON 生成带 slot_idx 的 flows.csv。

依赖：
- data/FlowData/Jul/Flow/clean_*.json
- data/FlowData/Jul/Axis/devices_axis.json
- streetflow_hetero_time_gnn/data/street.geojson
- streetflow_hetero_time_gnn/data_demo/nodes.csv（用于校验 public 节点）

输出：streetflow_hetero_time_gnn/data_demo/flows.csv（node_id, day, slot_idx, flow）
"""
import subprocess
import sys
from pathlib import Path

# 项目根目录（VitalStreetRL）
ROOT = Path(__file__).resolve().parent.parent.parent
STREETFLOW = ROOT / "streetflow_hetero_time_gnn"

FLOW_DIR = ROOT / "data" / "FlowData" / "Flow"
DEVICES = ROOT / "data" / "FlowData" / "devices_axis.json"
STREET = STREETFLOW / "data" / "street.geojson"
NODES = STREETFLOW / "data_demo" / "nodes.csv"
OUT = STREETFLOW / "data_demo" / "flows.csv"

BUILD_SCRIPT = STREETFLOW / "geo" / "tool" / "build_flow_15min.py"


def main():
    if not FLOW_DIR.exists():
        print(f"错误: Flow 目录不存在: {FLOW_DIR}")
        sys.exit(1)
    if not DEVICES.exists():
        print(f"错误: devices_axis.json 不存在: {DEVICES}")
        sys.exit(1)
    if not STREET.exists():
        print(f"错误: street.geojson 不存在: {STREET}")
        sys.exit(1)
    if not NODES.exists():
        print(f"错误: nodes.csv 不存在: {NODES}")
        sys.exit(1)

    cmd = [
        sys.executable,
        str(BUILD_SCRIPT),
        "--flow_dir", str(FLOW_DIR),
        "--devices", str(DEVICES),
        "--street", str(STREET),
        "--nodes", str(NODES),
        "--out", str(OUT),
        "--day-offset", "19",
    ]
    print("执行:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        sys.exit(result.returncode)
    print(f"\n已生成: {OUT}")


if __name__ == "__main__":
    main()

"""
Inference: given day/hour, output predictions.csv (node_id, node_type, yhat_log, yhat_flow).
"""
import csv
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from geo.tool.build_graph import build_hetero_data, load_normalizer
from models.hetero_sage import HeteroSAGE
from data.subgraph_utils import extract_2hop_subgraph


def predict(
    data_dir: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    day: int,
    hour: int,
    slot_idx: Optional[int] = None,
    use_slot: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[Path, torch.Tensor, "HeteroData"]:
    """
    Run prediction for (day, hour) or (day, slot_idx).
    use_slot: 若 True，hour 视为 slot_idx，label 为 1-10。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nodes_path = data_dir / "nodes.csv"
    edges_path = data_dir / "edges.csv"
    flows_path = data_dir / "flows.csv"
    if not flows_path.exists():
        flows_path = None
    normalizer_path = checkpoint_dir / "normalizer.json"
    normalizer = load_normalizer(normalizer_path) if normalizer_path.exists() else None
    label_tf = "remap_1_10" if use_slot else "log1p"
    s = slot_idx if use_slot and slot_idx is not None else hour
    data, _ = build_hetero_data(
        nodes_path, edges_path, flows_path,
        day=day, hour=hour if not use_slot else 12,
        slot_idx=s if use_slot else None,
        use_slot=use_slot,
        normalizer=normalizer,
        label_transform=label_tf,
        device=device,
    )
    data = data.to(device)
    model = HeteroSAGE(in_channels=11, hidden_channels=64, out_channels=1).to(device)
    ckpt = torch.load(checkpoint_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
    yhat_shop = out["shop"].squeeze(1).cpu()
    yhat_public = out["public"].squeeze(1).cpu()
    shop_ids = data["shop"].node_id.cpu().tolist()
    public_ids = data["public"].node_id.cpu().tolist()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "predictions.csv"
    if use_slot:
        yhat_flow_shop = yhat_shop
        yhat_flow_public = yhat_public
    else:
        yhat_flow_shop = torch.expm1(yhat_shop)
        yhat_flow_public = torch.expm1(yhat_public)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "node_type", "yhat_log", "yhat_flow"])
        for nid, yl, yf in zip(shop_ids, yhat_shop.tolist(), yhat_flow_shop.tolist()):
            w.writerow([nid, "shop", round(yl, 6), round(float(yf), 4)])
        for nid, yl, yf in zip(public_ids, yhat_public.tolist(), yhat_flow_public.tolist()):
            w.writerow([nid, "public", round(yl, 6), round(float(yf), 4)])
    yhat_flow_all = torch.cat([yhat_flow_shop, yhat_flow_public])
    return out_csv, yhat_flow_all, data.cpu()


def predict_shop_converted_to_public(
    data_dir: Path,
    checkpoint_dir: Path,
    shop_node_id: int,
    day: int = 0,
    hour: int = 12,
    slot_idx: Optional[int] = None,
    use_slot: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[float, "HeteroData", int]:
    """
    将某个 SHOP 模拟改造成 public space，构建 2-hop 子图，预测该“新 public”的客流量。
    子图以转换后的节点为中心，其与周边 shop 的边来自 edges.csv（需包含 shop-shop 邻接）。
    Returns: (yhat_flow, subgraph_data, center_idx_in_subgraph)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nodes_path = data_dir / "nodes.csv"
    edges_path = data_dir / "edges.csv"
    flows_path = data_dir / "flows.csv"
    if not flows_path.exists():
        flows_path = None
    normalizer_path = checkpoint_dir / "normalizer.json"
    normalizer = load_normalizer(normalizer_path) if normalizer_path.exists() else None
    label_tf = "remap_1_10" if use_slot else "log1p"
    s = slot_idx if use_slot and slot_idx is not None else hour
    data, _ = build_hetero_data(
        nodes_path, edges_path, flows_path,
        day=day, hour=hour if not use_slot else 12,
        slot_idx=s if use_slot else None,
        use_slot=use_slot,
        normalizer=normalizer,
        label_transform=label_tf,
        device=device,
        shop_to_public=shop_node_id,
    )
    public_ids = data["public"].node_id.cpu().tolist()
    if shop_node_id not in public_ids:
        raise ValueError(f"shop_node_id {shop_node_id} 转换后未出现在 public 中")
    center_idx = public_ids.index(shop_node_id)
    sub, center_new = extract_2hop_subgraph(data, "public", center_idx)
    sub = sub.to(device)
    model = HeteroSAGE(in_channels=11, hidden_channels=64, out_channels=1).to(device)
    ckpt = torch.load(checkpoint_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        out = model(sub.x_dict, sub.edge_index_dict)
    yhat_public = out["public"].squeeze(1).cpu()
    yhat_flow = float(yhat_public[center_new].item()) if use_slot else float(torch.expm1(yhat_public[center_new]).item())
    return yhat_flow, sub.cpu(), center_new


def main():
    import argparse
    p = argparse.ArgumentParser(description="Inference: output predictions.csv for given day/hour or day/slot.")
    p.add_argument("--data_dir", type=Path, default=ROOT / "data_demo", help="Directory with nodes.csv, edges.csv, flows.csv")
    p.add_argument("--checkpoint_dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--output_dir", type=Path, default=ROOT / "outputs")
    p.add_argument("--day", type=int, default=0)
    p.add_argument("--hour", type=int, default=12, help="hour or slot_idx when --use_slot")
    p.add_argument("--use_slot", action="store_true", help="15min slot mode")
    args = p.parse_args()
    out_csv, _, _ = predict(args.data_dir, args.checkpoint_dir, args.output_dir, args.day, args.hour,
        slot_idx=args.hour if args.use_slot else None, use_slot=args.use_slot)
    print(f"Predictions written to {out_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full pipeline script mirroring 01_pipeline_visualization.ipynb.
Saves all figures and logs to --out_dir (default: outputs/run_<timestamp>).
"""
# Avoid OMP Error #15 when PyTorch and other libs both ship OpenMP (libiomp5md.dll).
# Must be set before importing torch/matplotlib/numpy.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
try:
    import geopandas as gpd
except ImportError:
    gpd = None
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import SubgraphTimeSliceDataset, get_train_val_time_slices
from geo.tool.build_graph import build_hetero_data, load_normalizer, save_normalizer
from models.hetero_sage import HeteroSAGE
from utils.metrics import mae, rmse
from utils.seed import set_seed
from utils.viz import (
    plot_train_curves,
    plot_pred_vs_true,
    plot_error_histogram,
    plot_nodes_by_value,
    plot_heatmap_choropleth,
    plot_hetero_graph,
)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n_batch = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict)
        pred = out["public"].squeeze(1)
        center_idx = batch.center_idx
        pred_center = pred[center_idx : center_idx + 1]
        y_center = batch["public"].y
        loss = F.l1_loss(pred_center, y_center)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batch += 1
    return total_loss / max(n_batch, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y, all_pred, all_mask = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict)
        pred = out["public"].squeeze(1)
        center_idx = batch.center_idx
        pred_center = pred[center_idx : center_idx + 1]
        all_y.append(batch["public"].y.squeeze(1).cpu())
        all_pred.append(pred_center.cpu())
        all_mask.append(torch.ones(1, dtype=torch.bool))
    y = torch.cat(all_y)
    pred = torch.cat(all_pred)
    mask = torch.cat(all_mask)
    return mae(y, pred, mask), rmse(y, pred, mask), y, pred, mask


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline and save figures/logs")
    parser.add_argument("--data_dir", type=Path, default=ROOT / "data_demo", help="nodes/edges/flows CSV dir")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output dir; default: outputs/run_<timestamp>")
    parser.add_argument("--checkpoint_dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--project_root", type=Path, default=ROOT.parent, help="VitalStreetRL root for FlowData")
    parser.add_argument("--skip_csv", action="store_true", help="Skip A. Generate CSV (use existing data_dir)")
    parser.add_argument("--run_optuna", action="store_true", help="Run Optuna before training")
    parser.add_argument("--optuna_trials", type=int, default=5)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shop_to_public", type=int, default=None, help="If set, run F: predict converted shop")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = ROOT / "outputs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    log_dir = out_dir / "logs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    project_root = Path(args.project_root)
    street_geojson = ROOT / "data" / "street.geojson"
    flow_dir = project_root / "data" / "FlowData" / "Jul" / "Flow"
    devices_path = project_root / "data" / "FlowData" / "Jul" / "Axis" / "devices_axis.json"

    # ---------- A. Generate CSV ----------
    if not args.skip_csv:
        print("A. Generating CSV...")
        subprocess.run(
            [sys.executable, str(ROOT / "geo" / "tool" / "build_nodes_edges.py"),
             "--geojson", str(street_geojson), "--out_dir", str(data_dir)],
            check=True, cwd=str(ROOT),
        )
        if flow_dir.exists() and devices_path.exists():
            subprocess.run(
                [sys.executable, str(ROOT / "geo" / "tool" / "build_flow_15min.py"),
                 "--flow_dir", str(flow_dir), "--devices", str(devices_path),
                 "--street", str(street_geojson), "--nodes", str(data_dir / "nodes.csv"),
                 "--out", str(data_dir / "flows.csv"), "--day-offset", "19"],
                check=True, cwd=str(project_root),
            )
        else:
            print("  FlowData not found, skipping flows.csv")
    else:
        print("A. Skipping CSV generation")

    # ---------- B. Time slices & datasets ----------
    print("B. Building datasets...")
    DAYS = list(range(11))
    SLOTS = list(range(20))
    time_slices = [(d, s) for d in DAYS for s in SLOTS]
    train_slices, val_slices = get_train_val_time_slices(time_slices, val_ratio=0.2, seed=args.seed)
    label_tf = "remap_1_10"
    train_ds = SubgraphTimeSliceDataset(data_dir, train_slices, use_slot=True, label_transform=label_tf)
    val_ds = SubgraphTimeSliceDataset(data_dir, val_slices, use_slot=True, normalizer=train_ds.normalizer, label_transform=label_tf)
    if train_ds.normalizer is not None:
        save_normalizer(train_ds.normalizer, checkpoint_dir / "normalizer.json")
    train_loader = PyGDataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = PyGDataLoader(val_ds, batch_size=1, shuffle=False)
    feat_dim = train_ds[0]["shop"].x.shape[1]
    print(f"  train={len(train_ds)}, val={len(val_ds)}, feat_dim={feat_dim}")

    # ---------- Optional: Optuna ----------
    optuna_best_params = None
    if args.run_optuna:
        print("D.0 Running Optuna...")
        from scripts.optuna_tune import run_study
        run_study(data_dir=data_dir, checkpoint_dir=checkpoint_dir, n_trials=args.optuna_trials, time_slices_ratio=0.3, seed=args.seed)
        optuna_path = checkpoint_dir / "optuna_best_params.json"
        if optuna_path.exists():
            with open(optuna_path, "r", encoding="utf-8") as f:
                optuna_best_params = json.load(f)
            print("  Loaded optuna_best_params:", optuna_best_params)

    # ---------- D. Training ----------
    if optuna_best_params is None and (checkpoint_dir / "optuna_best_params.json").exists():
        with open(checkpoint_dir / "optuna_best_params.json", "r", encoding="utf-8") as f:
            optuna_best_params = json.load(f)
    hidden_channels = (optuna_best_params or {}).get("hidden_channels", 64)
    lr = (optuna_best_params or {}).get("lr", 1e-3)
    weight_decay = (optuna_best_params or {}).get("weight_decay", 0.0)
    n_epochs = (optuna_best_params or {}).get("n_epochs", args.n_epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeteroSAGE(feat_dim, hidden_channels, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_mae = float("inf")
    log_epochs, log_train_loss, log_val_mae = [], [], []

    print("D. Training...")
    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_mae_, val_rmse_, _, _, _ = evaluate(model, val_loader, device)
        log_epochs.append(epoch)
        log_train_loss.append(train_loss)
        log_val_mae.append(val_mae_)
        if val_mae_ < best_val_mae:
            best_val_mae = val_mae_
            torch.save({"model": model.state_dict(), "epoch": epoch, "hidden_channels": hidden_channels}, checkpoint_dir / "best.pt")
        if epoch % 20 == 0:
            print(f"  Epoch {epoch} loss={train_loss:.4f} val_mae={val_mae_:.4f} val_rmse={val_rmse_:.4f}")

    train_log = {"epoch": log_epochs, "train_loss": log_train_loss, "val_mae": log_val_mae, "best_val_mae": best_val_mae}
    with open(log_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2)
    print(f"  Best val_mae = {best_val_mae:.4f}")

    # ---------- E. Plots: train curves, pred vs true, error hist ----------
    print("E. Saving figures...")
    plot_train_curves(log_epochs, log_train_loss, log_val_mae, out_path=fig_dir / "train_curves.png")
    plt.close("all")

    ckpt = torch.load(checkpoint_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    _, _, y_true, y_pred, mask = evaluate(model, val_loader, device)
    plot_pred_vs_true(y_true, y_pred, mask, out_path=fig_dir / "pred_vs_true.png")
    plt.close("all")
    plot_error_histogram(y_true, y_pred, mask, out_path=fig_dir / "error_histogram.png")
    plt.close("all")

    # ---------- E2. Full-graph prediction + node map + heatmap ----------
    if gpd is not None:
        day, slot = 0, 0
        nodes_path = data_dir / "nodes.csv"
        edges_path = data_dir / "edges.csv"
        flows_path = data_dir / "flows.csv" if (data_dir / "flows.csv").exists() else None
        normalizer = load_normalizer(checkpoint_dir / "normalizer.json") if (checkpoint_dir / "normalizer.json").exists() else None
        data_pred, _ = build_hetero_data(nodes_path, edges_path, flows_path, day=day, hour=12, slot_idx=slot, use_slot=True, normalizer=normalizer, label_transform=label_tf)
        data_pred = data_pred.to(device)
        with torch.no_grad():
            out = model(data_pred.x_dict, data_pred.edge_index_dict)
        yhat_public = out["public"].squeeze(1).cpu()
        yhat_shop = out["shop"].squeeze(1).cpu()
        yhat_flow_all = torch.cat([yhat_shop, yhat_public])
        public_vals = yhat_public
        vmin, vmax = float(public_vals.min()), float(public_vals.max())
        gdf = gpd.read_file(street_geojson) if street_geojson.exists() else None
        if gdf is not None:
            plot_nodes_by_value(data_pred.cpu(), public_vals, title=f"Predicted flow (day={day}, slot={slot})", value_label="yhat_flow", gdf=gdf, vmin=vmin, vmax=vmax, out_path=fig_dir / "nodes_by_value.png")
            plt.close("all")
            plot_heatmap_choropleth(gdf, yhat_flow_all.numpy(), title=f"Predicted flow heatmap (day={day}, slot={slot})", value_label="yhat_flow", smooth_with_neighbors=True, vmin=vmin, vmax=vmax, out_path=fig_dir / "heatmap_choropleth.png")
            plt.close("all")
    else:
        print("  E2 (node map/heatmap) skipped: geopandas not available")

    # ---------- F. Shop -> Public prediction ----------
    if args.shop_to_public is not None:
        print("F. Shop -> Public prediction...")
        from predict import predict_shop_converted_to_public
        yhat_flow, sub, center_idx = predict_shop_converted_to_public(
            data_dir, checkpoint_dir, args.shop_to_public,
            day=0, hour=12, slot_idx=0, use_slot=True,
        )
        gdf = gpd.read_file(street_geojson) if (gpd is not None and street_geojson.exists()) else None
        fig = plot_hetero_graph(sub, title=f"Subgraph: Shop {args.shop_to_public} -> Public (predicted flow {yhat_flow:.2f})", center_node=("public", center_idx), gdf=gdf)
        fig.savefig(fig_dir / "shop_to_public_subgraph.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        with open(log_dir / "shop_to_public_result.json", "w", encoding="utf-8") as f:
            json.dump({"shop_node_id": args.shop_to_public, "yhat_flow": yhat_flow}, f, indent=2)

    # ---------- Summary ----------
    summary = {
        "out_dir": str(out_dir),
        "fig_dir": str(fig_dir),
        "log_dir": str(log_dir),
        "best_val_mae": best_val_mae,
        "n_epochs": n_epochs,
        "hidden_channels": hidden_channels,
        "lr": lr,
    }
    with open(out_dir / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Outputs saved to {out_dir}")
    print(f"  figures: {fig_dir}")
    print(f"  logs:    {log_dir}")


if __name__ == "__main__":
    main()

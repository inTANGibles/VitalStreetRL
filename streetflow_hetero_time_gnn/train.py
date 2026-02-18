"""
Training entry: masked regression on public nodes.
- 整图模式：TimeSliceDataset，按时间片训练
- 子图模式：SubgraphTimeSliceDataset，4400 样本（220×20），2-hop 子图
Saves best.pt (by val_mae), outputs/train_log.json.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# project root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.dataset import (
    TimeSliceDataset,
    SubgraphTimeSliceDataset,
    get_train_val_time_slices,
)
from geo.tool.build_graph import save_normalizer
from models.hetero_sage import HeteroSAGE
from utils.metrics import mae, rmse
from utils.seed import set_seed


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batch = 0
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
    return mae(y, pred, mask), rmse(y, pred, mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subgraph", action="store_true", help="使用 2-hop 子图数据集（4400 样本）")
    parser.add_argument("--use_slot", action="store_true", help="15 分钟时间片（220 片），需 flows.csv 含 slot_idx")
    args = parser.parse_args()

    set_seed(42)
    data_dir = ROOT / "data_demo"
    checkpoint_dir = ROOT / "checkpoints"
    output_dir = ROOT / "outputs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_slot:
        # 220 时间片：11 天 × 20 slot/天
        DAYS = list(range(11))
        SLOTS = list(range(20))
        time_slices = [(d, s) for d in DAYS for s in SLOTS]
        label_transform = "remap_1_10"
    else:
        # 42 时间片：7 天 × 6 小时
        DAYS = list(range(7))
        HOURS = list(range(12, 18))
        time_slices = [(d, h) for d in DAYS for h in HOURS]
        label_transform = "log1p"

    train_slices, val_slices = get_train_val_time_slices(time_slices, val_ratio=0.2)

    if args.subgraph:
        train_ds = SubgraphTimeSliceDataset(
            data_dir, train_slices, use_slot=args.use_slot,
            label_transform=label_transform,
        )
        val_ds = SubgraphTimeSliceDataset(
            data_dir, val_slices, use_slot=args.use_slot,
            normalizer=train_ds.normalizer,
            label_transform=label_transform,
        )
        print(f"Subgraph 模式: train={len(train_ds)}, val={len(val_ds)}")
    else:
        train_ds = TimeSliceDataset(
            data_dir, train_slices, use_slot=args.use_slot,
            label_transform=label_transform,
        )
        val_ds = TimeSliceDataset(
            data_dir, val_slices, use_slot=args.use_slot,
            normalizer=train_ds.normalizer,
            label_transform=label_transform,
        )

    if train_ds.normalizer is not None:
        save_normalizer(train_ds.normalizer, checkpoint_dir / "normalizer.json")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeteroSAGE(in_channels=11, hidden_channels=64, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_mae = float("inf")
    log_epochs = []
    log_train_loss = []
    log_val_mae = []

    n_epochs = 200
    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_mae_, val_rmse_ = evaluate(model, val_loader, device)
        log_epochs.append(epoch)
        log_train_loss.append(train_loss)
        log_val_mae.append(val_mae_)
        if val_mae_ < best_val_mae:
            best_val_mae = val_mae_
            torch.save({"model": model.state_dict(), "epoch": epoch}, checkpoint_dir / "best.pt")
        if epoch % 20 == 0:
            print(f"Epoch {epoch} train_loss={train_loss:.4f} val_mae={val_mae_:.4f} val_rmse={val_rmse_:.4f}")

    with open(output_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump({
            "epoch": log_epochs,
            "train_loss": log_train_loss,
            "val_mae": log_val_mae,
        }, f, indent=2)
    print(f"Saved best.pt and train_log.json. Best val_mae={best_val_mae:.4f}")


if __name__ == "__main__":
    main()

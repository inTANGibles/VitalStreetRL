#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optuna hyperparameter tuning for StreetFlow Hetero GNN.
Minimizes validation MAE. Saves best params to checkpoints/optuna_best_params.json.
Usage:
  python scripts/optuna_tune.py [--data_dir DATA_DIR] [--n_trials N] [--time_slices_ratio R]
  Or import and run_study() from notebook.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import optuna
from optuna.trial import Trial

from data.dataset import SubgraphTimeSliceDataset, get_train_val_time_slices
from geo.tool.build_graph import save_normalizer
from models.sage import SAGE
from utils.metrics import mae, rmse
from utils.seed import set_seed


def _collate_batch_size1(batch):
    """batch_size=1 时直接返回唯一样本（PyG Data）。"""
    return batch[0]


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n_batch = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index).squeeze(1)
        center_idx = batch.center_idx
        pred_center = pred[center_idx : center_idx + 1]
        y_center = batch.y.squeeze(1)
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
        pred = model(batch.x, batch.edge_index).squeeze(1)
        center_idx = batch.center_idx
        pred_center = pred[center_idx : center_idx + 1]
        all_y.append(batch.y.squeeze(1).cpu())
        all_pred.append(pred_center.cpu())
        all_mask.append(torch.ones(1, dtype=torch.bool))
    y = torch.cat(all_y)
    pred = torch.cat(all_pred)
    mask = torch.cat(all_mask)
    return mae(y, pred, mask), rmse(y, pred, mask)


def create_objective(
    data_dir: Path,
    train_slices: list,
    val_slices: list,
    label_tf: str = "remap_1_10",
    seed: int = 42,
    device=None,
):
    """Build an objective function for Optuna that uses the given data splits."""

    def objective(trial: Trial) -> float:
        set_seed(seed)
        if device is None:
            device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_ = device

        # Hyperparameters to tune
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        n_epochs = trial.suggest_int("n_epochs", 40, 120, step=20)

        train_ds = SubgraphTimeSliceDataset(
            data_dir, train_slices, use_slot=True, label_transform=label_tf
        )
        val_ds = SubgraphTimeSliceDataset(
            data_dir, val_slices, use_slot=True,
            normalizer=train_ds.normalizer, label_transform=label_tf,
        )
        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True, collate_fn=_collate_batch_size1
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, collate_fn=_collate_batch_size1
        )

        in_channels = 11
        out_channels = 1
        model = SAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
        ).to(device_)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        best_val_mae = float("inf")
        for epoch in range(1, n_epochs + 1):
            train_one_epoch(model, train_loader, optimizer, device_)
            val_mae_, _ = evaluate(model, val_loader, device_)
            if val_mae_ < best_val_mae:
                best_val_mae = val_mae_
            trial.report(best_val_mae, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return best_val_mae

    return objective


def run_study(
    data_dir: Path = None,
    checkpoint_dir: Path = None,
    n_trials: int = 20,
    timeout: float = None,
    time_slices_ratio: float = 1.0,
    seed: int = 42,
) -> optuna.Study:
    """
    Run Optuna study and save best params to checkpoint_dir/optuna_best_params.json.
    time_slices_ratio: use a fraction of time_slices for faster tuning (e.g. 0.3).
    """
    if data_dir is None:
        data_dir = ROOT / "data_demo"
    if checkpoint_dir is None:
        checkpoint_dir = ROOT / "checkpoints"
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    DAYS = list(range(11))
    SLOTS = list(range(20))
    time_slices_full = [(d, s) for d in DAYS for s in SLOTS]
    n_use = max(1, int(len(time_slices_full) * time_slices_ratio))
    time_slices = time_slices_full[:n_use]
    train_slices, val_slices = get_train_val_time_slices(time_slices, val_ratio=0.2, seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objective = create_objective(
        Path(data_dir), train_slices, val_slices,
        label_tf="remap_1_10", seed=seed, device=device,
    )

    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=10)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best = study.best_params
    best["best_val_mae"] = study.best_value
    out_path = checkpoint_dir / "optuna_best_params.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print(f"Best params saved to {out_path}")
    print(f"Best val_mae = {study.best_value:.4f}")
    print("Best params:", best)
    return study


def main():
    import argparse
    p = argparse.ArgumentParser(description="Optuna hyperparameter tuning for StreetFlow GNN")
    p.add_argument("--data_dir", type=Path, default=ROOT / "data_demo")
    p.add_argument("--checkpoint_dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--timeout", type=float, default=None, help="Max seconds for the study")
    p.add_argument("--time_slices_ratio", type=float, default=0.3, help="Use fraction of time slices for faster tuning")
    args = p.parse_args()
    run_study(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        n_trials=args.n_trials,
        timeout=args.timeout,
        time_slices_ratio=args.time_slices_ratio,
    )


if __name__ == "__main__":
    main()

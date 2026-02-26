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
from torch.utils.data import DataLoader, Subset

import numpy as np
import optuna
from optuna.trial import Trial

from data.dataset import SubgraphTimeSliceDataset, FullSubgraphTimeSliceDataset, get_train_val_time_slices
from geo.tool.build_graph import save_normalizer
from models.sage import SAGE
from models.gcn import GCN
from utils.metrics import mae, rmse
from utils.seed import set_seed


def _collate_batch_size1(batch):
    """batch_size=1 时直接返回唯一样本（PyG Data）。"""
    return batch[0]


def train_one_epoch(model, loader, optimizer, device, is_cls: bool = False):
    model.train()
    total_loss, n_batch = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index)
        center_idx = batch.center_idx
        pred_center = pred[center_idx : center_idx + 1]
        y_center = batch.y.squeeze(1)
        if is_cls:
            pred_center = pred_center.squeeze(-1) if pred_center.dim() > 2 else pred_center
            loss = F.cross_entropy(pred_center, y_center.long())
        else:
            pred_center = pred_center.squeeze(1)
            loss = F.l1_loss(pred_center, y_center)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batch += 1
    return total_loss / max(n_batch, 1)


def train_one_epoch_semisupervised(model, loader, optimizer, device, is_cls: bool = False):
    """半监督：仅对 mask=True 的样本计算损失。is_cls 时用 CrossEntropy，否则 L1。"""
    model.train()
    total_loss, n_batch = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        if not batch.mask.any():
            continue
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index)
        center_idx = batch.center_idx
        pred_center = pred[center_idx : center_idx + 1]
        y_center = batch.y.squeeze(1)
        if is_cls:
            pred_center = pred_center.squeeze(-1) if pred_center.dim() > 2 else pred_center.squeeze(1)
            loss = F.cross_entropy(pred_center, y_center.long())
        else:
            pred_center = pred_center.squeeze(1)
            loss = F.l1_loss(pred_center, y_center)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batch += 1
    return total_loss / max(n_batch, 1)


@torch.no_grad()
def evaluate_semisupervised(model, loader, device, is_cls: bool = False):
    """评估：mask=True 的样本。is_cls 时返回 (1-acc, acc)，否则 (MAE, RMSE)。"""
    model.eval()
    all_y, all_pred, all_mask = [], [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index)
        center_idx = batch.center_idx
        pred_center = pred[center_idx : center_idx + 1]
        all_y.append(batch.y.squeeze(1).cpu())
        all_pred.append(pred_center.cpu())
        all_mask.append(batch.mask.cpu())
    y = torch.cat(all_y)
    pred = torch.cat(all_pred)
    mask = torch.cat(all_mask)
    if is_cls:
        y_long = y.long()
        correct = (pred.argmax(1) == y_long)[mask]
        acc = correct.float().mean().item() if mask.sum() > 0 else 0.0
        return 1.0 - acc, acc  # 第一个供 minimize，第二个供打印
    return mae(y, pred, mask), rmse(y, pred, mask)


@torch.no_grad()
def evaluate(model, loader, device, is_cls: bool = False):
    model.eval()
    all_y, all_pred, all_mask = [], [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index)
        center_idx = batch.center_idx
        pred_center = pred[center_idx : center_idx + 1]
        y_center = batch.y.squeeze(1)
        if is_cls:
            all_y.append(y_center.long().cpu())
            all_pred.append(pred_center.cpu())
            all_mask.append(torch.ones(1, dtype=torch.bool))
        else:
            all_y.append(y_center.cpu())
            all_pred.append(pred_center.squeeze(1).cpu())
            all_mask.append(torch.ones(1, dtype=torch.bool))
    y = torch.cat(all_y)
    pred = torch.cat(all_pred)
    mask = torch.cat(all_mask)
    if is_cls:
        acc = (pred.argmax(1) == y)[mask].float().mean().item()
        return 1.0 - acc, acc  # 返回 (1-acc) 用于 minimize，以及 acc 供打印
    return mae(y, pred, mask), rmse(y, pred, mask)


def create_objective(
    data_dir: Path,
    train_slices: list,
    val_slices: list,
    label_tf: str = "remap_1_10",
    use_slot: bool = True,
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
        dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)
        n_epochs = trial.suggest_int("n_epochs", 40, 120, step=20)

        train_ds = SubgraphTimeSliceDataset(
            data_dir, train_slices, use_slot=use_slot, label_transform=label_tf
        )
        val_ds = SubgraphTimeSliceDataset(
            data_dir, val_slices, use_slot=use_slot,
            normalizer=train_ds.normalizer, label_transform=label_tf,
        )
        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True, collate_fn=_collate_batch_size1
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, collate_fn=_collate_batch_size1
        )

        in_channels = train_ds[0].x.shape[1]
        out_channels = 1
        model = SAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
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


def create_objective_from_datasets(
    train_ds,
    val_ds,
    model_cls=None,
    label_tf: str = "log1p",
    seed: int = 42,
    device=None,
):
    """Build objective using pre-built train_ds and val_ds. model_cls: SAGE or GCN (default SAGE).
    label_tf=remap_1_10 时使用分类模式（out_channels=10, CrossEntropy, 优化 accuracy）。"""
    if model_cls is None:
        model_cls = SAGE
    is_cls = label_tf == "remap_1_10"

    def objective(trial: Trial) -> float:
        set_seed(seed)
        if device is None:
            device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_ = device

        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)
        n_epochs = trial.suggest_int("n_epochs", 40, 120, step=20)

        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True, collate_fn=_collate_batch_size1
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, collate_fn=_collate_batch_size1
        )

        in_channels = train_ds[0].x.shape[1]
        out_channels = 10 if is_cls else 1
        model = model_cls(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
        ).to(device_)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        best_val = 0.0 if is_cls else float("inf")
        for epoch in range(1, n_epochs + 1):
            train_one_epoch(model, train_loader, optimizer, device_, is_cls=is_cls)
            val_metric, val_acc = evaluate(model, val_loader, device_, is_cls=is_cls)
            if is_cls:
                if val_acc > best_val:
                    best_val = val_acc
                trial.report(best_val, epoch)
            else:
                if val_metric < best_val:
                    best_val = val_metric
                trial.report(best_val, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return best_val

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


def run_study_38_subgraphs(
    data_dir: Path = None,
    checkpoint_dir: Path = None,
    time_slices: list = None,
    use_slot: bool = False,
    label_tf: str = "log1p",
    val_ratio: float = 0.2,
    n_trials: int = 20,
    timeout: float = None,
    seed: int = 42,
    model_cls=None,
    optuna_params_path: str = None,
    num_hops: int = 2,
) -> optuna.Study:
    """
    38 子图模式：1 时间片 × 38 public 节点，按子图索引划分 train/val。
    time_slices: 从 flows.csv 解析，如 [(0,0)]；若为 None 则自动解析。
    """
    if data_dir is None:
        data_dir = ROOT / "data_demo"
    if checkpoint_dir is None:
        checkpoint_dir = ROOT / "checkpoints"
    data_dir = Path(data_dir)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if time_slices is None:
        import pandas as pd
        flows_path = data_dir / "flows.csv"
        if not flows_path.exists():
            raise FileNotFoundError(f"flows.csv not found in {data_dir}")
        flows_df = pd.read_csv(flows_path)
        if "slot_idx" in flows_df.columns:
            df_slices = flows_df[["day", "slot_idx"]].drop_duplicates()
            use_slot = True
        else:
            df_slices = flows_df[["day", "hour"]].drop_duplicates()
            use_slot = False
        time_slices = [tuple(r) for r in df_slices.values]

    full_ds = SubgraphTimeSliceDataset(
        data_dir, time_slices, use_slot=use_slot, label_transform=label_tf, num_hops=num_hops
    )
    n_sub = len(full_ds)
    rng = np.random.RandomState(seed)
    idx = np.arange(n_sub)
    rng.shuffle(idx)
    n_val = max(1, int(n_sub * val_ratio))
    train_indices = idx[n_val:].tolist()
    val_indices = idx[:n_val].tolist()
    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)
    if full_ds.normalizer is not None:
        save_normalizer(full_ds.normalizer, checkpoint_dir / "normalizer.json")

    if model_cls is None:
        model_cls = SAGE
    model_name = "gcn" if model_cls is GCN else "sage"
    print(f"38 subgraph mode ({model_name}): n={n_sub}, train={len(train_ds)}, val={len(val_ds)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cls = label_tf == "remap_1_10"
    objective = create_objective_from_datasets(
        train_ds, val_ds, model_cls=model_cls, label_tf=label_tf, seed=seed, device=device
    )

    pruner = optuna.pruners.MedianPruner(n_startup_trials=min(3, n_trials), n_warmup_steps=10)
    study = optuna.create_study(
        direction="maximize" if is_cls else "minimize",
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best = study.best_params
    best["best_val_mae" if not is_cls else "best_val_acc"] = study.best_value
    out_path = Path(optuna_params_path) if optuna_params_path else checkpoint_dir / ("optuna_best_params_gcn.json" if model_name == "gcn" else "optuna_best_params.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print(f"Best params saved to {out_path}")
    print(f"Best val_{'acc' if is_cls else 'mae'} = {study.best_value:.4f}")
    return study


def run_study_full_subgraphs(
    data_dir: Path = None,
    checkpoint_dir: Path = None,
    time_slices: list = None,
    use_slot: bool = False,
    label_tf: str = "log1p",
    val_ratio: float = 0.2,
    n_trials: int = 20,
    timeout: float = None,
    seed: int = 42,
    model_cls=None,
    num_hops: int = 1,
    use_shop_pseudo: bool = True,
) -> optuna.Study:
    """
    105+38 子图半监督模式：FullSubgraphTimeSliceDataset，仅 mask=True 样本参与损失。
    """
    if data_dir is None:
        data_dir = ROOT / "data_demo"
    if checkpoint_dir is None:
        checkpoint_dir = ROOT / "checkpoints"
    data_dir = Path(data_dir)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if time_slices is None:
        import pandas as pd
        flows_path = data_dir / "flows.csv"
        if not flows_path.exists():
            raise FileNotFoundError(f"flows.csv not found in {data_dir}")
        flows_df = pd.read_csv(flows_path)
        if "slot_idx" in flows_df.columns:
            df_slices = flows_df[["day", "slot_idx"]].drop_duplicates()
            use_slot = True
        else:
            df_slices = flows_df[["day", "hour"]].drop_duplicates()
            use_slot = False
        time_slices = [tuple(r) for r in df_slices.values]

    full_ds = FullSubgraphTimeSliceDataset(
        data_dir, time_slices,
        use_slot=use_slot,
        label_transform=label_tf,
        num_hops=num_hops,
        use_shop_pseudo=use_shop_pseudo,
    )
    n_sub = len(full_ds)
    rng = np.random.RandomState(seed)
    idx = np.arange(n_sub)
    rng.shuffle(idx)
    n_val = max(1, int(n_sub * val_ratio))
    train_indices = idx[n_val:].tolist()
    val_indices = idx[:n_val].tolist()
    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)
    if full_ds.normalizer is not None:
        save_normalizer(full_ds.normalizer, checkpoint_dir / "normalizer.json")

    if model_cls is None:
        model_cls = GCN
    is_cls = label_tf == "remap_1_10"
    print(f"105+38 subgraph semi-supervised ({model_cls.__name__}): n={n_sub}, train={len(train_ds)}, val={len(val_ds)}, mode={'classification' if is_cls else 'regression'}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial: Trial) -> float:
        set_seed(seed)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)
        n_epochs = trial.suggest_int("n_epochs", 40, 120, step=20)

        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True, collate_fn=_collate_batch_size1
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, collate_fn=_collate_batch_size1
        )

        in_channels = train_ds[0].x.shape[1]
        out_channels = 10 if is_cls else 1
        model = model_cls(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val = 0.0 if is_cls else float("inf")
        for epoch in range(1, n_epochs + 1):
            train_one_epoch_semisupervised(model, train_loader, optimizer, device, is_cls=is_cls)
            val_metric, val_acc = evaluate_semisupervised(model, val_loader, device, is_cls=is_cls)
            if is_cls:
                if val_acc > best_val:
                    best_val = val_acc
                trial.report(best_val, epoch)
            else:
                if val_metric < best_val:
                    best_val = val_metric
                trial.report(best_val, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return best_val

    pruner = optuna.pruners.MedianPruner(n_startup_trials=min(3, n_trials), n_warmup_steps=10)
    study = optuna.create_study(
        direction="maximize" if is_cls else "minimize",
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best = study.best_params
    best["best_val_acc" if is_cls else "best_val_mae"] = study.best_value
    out_path = checkpoint_dir / "optuna_best_params_semisupervised.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print(f"Best params saved to {out_path}")
    print(f"Best val_{'acc' if is_cls else 'mae'} = {study.best_value:.4f}")
    return study


def train_one_epoch_consistency(model, loader, optimizer, device, lambda_consist: float = 0.2):
    """监督损失（仅 mask=True）+ 一致性损失（全部子图，两次前向）。"""
    model.train()
    total_loss, n_batch = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out1 = model(batch.x, batch.edge_index)
        out2 = model(batch.x, batch.edge_index)
        center_idx = batch.center_idx
        pred1 = out1[center_idx : center_idx + 1].squeeze(1)
        pred2 = out2[center_idx : center_idx + 1].squeeze(1)
        y_center = batch.y.squeeze(1)
        mask = batch.mask

        loss_consist = F.mse_loss(pred1, pred2)
        if mask.any():
            loss_sup = F.l1_loss(pred1, y_center)
            loss = loss_sup + lambda_consist * loss_consist
        else:
            loss = lambda_consist * loss_consist
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batch += 1
    return total_loss / max(n_batch, 1)


def run_study_consistency(
    data_dir: Path = None,
    checkpoint_dir: Path = None,
    time_slices: list = None,
    use_slot: bool = False,
    label_tf: str = "log1p",
    val_ratio: float = 0.2,
    n_trials: int = 20,
    timeout: float = None,
    seed: int = 42,
    model_cls=None,
    num_hops: int = 1,
) -> optuna.Study:
    """
    105+38 子图 + 一致性正则：use_shop_pseudo=False，监督仅 public；一致性对全部 143 子图。
    搜索 lr, hidden_channels, weight_decay, dropout, n_epochs, lambda_consist。
    """
    if data_dir is None:
        data_dir = ROOT / "data_demo"
    if checkpoint_dir is None:
        checkpoint_dir = ROOT / "checkpoints"
    data_dir = Path(data_dir)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if time_slices is None:
        import pandas as pd
        flows_path = data_dir / "flows.csv"
        if not flows_path.exists():
            raise FileNotFoundError(f"flows.csv not found in {data_dir}")
        flows_df = pd.read_csv(flows_path)
        if "slot_idx" in flows_df.columns:
            df_slices = flows_df[["day", "slot_idx"]].drop_duplicates()
            use_slot = True
        else:
            df_slices = flows_df[["day", "hour"]].drop_duplicates()
            use_slot = False
        time_slices = [tuple(r) for r in df_slices.values]

    full_ds = FullSubgraphTimeSliceDataset(
        data_dir, time_slices,
        use_slot=use_slot,
        label_transform=label_tf,
        num_hops=num_hops,
        use_shop_pseudo=False,
    )
    n_sub = len(full_ds)
    rng = np.random.RandomState(seed)
    idx = np.arange(n_sub)
    rng.shuffle(idx)
    n_val = max(1, int(n_sub * val_ratio))
    train_indices = idx[n_val:].tolist()
    val_indices = idx[:n_val].tolist()
    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)
    if full_ds.normalizer is not None:
        save_normalizer(full_ds.normalizer, checkpoint_dir / "normalizer.json")

    if model_cls is None:
        model_cls = GCN
    print(f"105+38 consistency ({model_cls.__name__}): n={n_sub}, train={len(train_ds)}, val={len(val_ds)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial: Trial) -> float:
        set_seed(seed)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)
        n_epochs = trial.suggest_int("n_epochs", 40, 120, step=20)
        lambda_consist = trial.suggest_float("lambda_consist", 0.05, 0.5, step=0.05)

        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True, collate_fn=_collate_batch_size1
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, collate_fn=_collate_batch_size1
        )

        in_channels = train_ds[0].x.shape[1]
        model = model_cls(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=1,
            dropout=dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_mae = float("inf")
        for epoch in range(1, n_epochs + 1):
            train_one_epoch_consistency(model, train_loader, optimizer, device, lambda_consist=lambda_consist)
            val_mae_, _ = evaluate_semisupervised(model, val_loader, device)
            if val_mae_ < best_val_mae:
                best_val_mae = val_mae_
            trial.report(best_val_mae, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return best_val_mae

    pruner = optuna.pruners.MedianPruner(n_startup_trials=min(3, n_trials), n_warmup_steps=10)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best = study.best_params
    best["best_val_mae"] = study.best_value
    out_path = checkpoint_dir / "optuna_best_params_consistency.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print(f"Best params saved to {out_path}")
    print(f"Best val_mae = {study.best_value:.4f}")
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

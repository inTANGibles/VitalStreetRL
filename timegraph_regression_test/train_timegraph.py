"""
时间片整图节点回归：7-fold by day，GCN 与 MLP baseline。
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from build_time_dataset import build_42_graph_views
from models import GCNTimeNodeRegressor, MLPTimeBaseline
from utils import set_seed, masked_mae, EarlyStopping


def load_data(graph_pt: str, y_pt: str, mask_pt: str) -> tuple:
    """加载 graph_cache_static.pt, y_time.pt, mask_time.pt"""
    graph = torch.load(graph_pt, map_location="cpu", weights_only=True)
    y_time = torch.load(y_pt, map_location="cpu", weights_only=True)
    mask_time = torch.load(mask_pt, map_location="cpu", weights_only=True)
    return graph, y_time, mask_time


def get_fold_indices_by_day(num_days: int = 7, samples_per_day: int = 6):
    """
    按天划分 7-fold。
    返回 [(train_idx, test_idx), ...]，每个 test_idx 为某天的 6 个样本索引。
    """
    folds = []
    for test_day in range(num_days):
        test_start = test_day * samples_per_day
        test_end = test_start + samples_per_day
        test_idx = list(range(test_start, test_end))
        train_idx = [i for i in range(42) if i not in test_idx]
        folds.append((train_idx, test_idx))
    return folds


def train_one_fold(
    model,
    train_data,
    test_data,
    args,
    device,
    use_mask_loss: bool = True,
):
    """训练一个 fold，返回 test MAE(log)。"""
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.L1Loss(reduction="none")
    early_stop = EarlyStopping(patience=args.patience)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred = model(batch)
            loss_per_node = criterion(pred, batch.y)
            if use_mask_loss and hasattr(batch, "mask") and batch.mask is not None:
                if batch.mask.sum() > 0:
                    loss = loss_per_node[batch.mask].mean()
                else:
                    continue
            else:
                loss = loss_per_node.mean()
            loss.backward()
            opt.step()

        # 验证：用 train 的 mask 做早停（或可用 train 子集做 val，此处简化用 train）
        model.eval()
        with torch.no_grad():
            train_maes = []
            for d in train_data:
                d = d.to(device)
                pred = model(d)
                mae = masked_mae(pred, d.y, d.mask)
                if not (mae != mae):  # not nan
                    train_maes.append(mae)
            val_mae = sum(train_maes) / len(train_maes) if train_maes else float("inf")

        if early_stop(val_mae, lower_is_better=True):
            break

    # 评估 test
    model.eval()
    test_preds = []
    test_targets = []
    test_masks = []
    with torch.no_grad():
        for d in test_data:
            d = d.to(device)
            pred = model(d)
            test_preds.append(pred)
            test_targets.append(d.y)
            test_masks.append(d.mask)

    pred_cat = torch.cat(test_preds)
    target_cat = torch.cat(test_targets)
    mask_cat = torch.cat(test_masks)
    test_mae = masked_mae(pred_cat, target_cat, mask_cat)
    return test_mae


def run_cv_gcn(dataset, args, device):
    """GCN 7-fold by day"""
    folds = get_fold_indices_by_day()
    maes = []
    time_dim = 4  # use_day_of_week=True
    for fold, (train_idx, test_idx) in enumerate(folds):
        set_seed(42 + fold)
        train_data = [dataset[i] for i in train_idx]
        test_data = [dataset[i] for i in test_idx]

        model = GCNTimeNodeRegressor(
            in_cont=11,
            num_func_types=2,
            time_dim=time_dim,
            emb_dim=8,
            hidden=64,
        ).to(device)

        mae = train_one_fold(model, train_data, test_data, args, device, use_mask_loss=True)
        maes.append(mae)
        print(f"  Fold {fold + 1} (test_day={fold}): test MAE(log) = {mae:.6f}")
    return maes


def run_cv_mlp(dataset, args, device):
    """MLP baseline 7-fold by day"""
    folds = get_fold_indices_by_day()
    maes = []
    time_dim = 4
    for fold, (train_idx, test_idx) in enumerate(folds):
        set_seed(42 + fold)
        train_data = [dataset[i] for i in train_idx]
        test_data = [dataset[i] for i in test_idx]

        model = MLPTimeBaseline(
            in_cont=11,
            num_func_types=2,
            time_dim=time_dim,
            emb_dim=8,
            hidden=64,
        ).to(device)

        mae = train_one_fold(model, train_data, test_data, args, device, use_mask_loss=True)
        maes.append(mae)
        print(f"  Fold {fold + 1} (test_day={fold}): test MAE(log) = {mae:.6f}")
    return maes


def generate_mock_cache_if_missing(graph_pt: str, y_pt: str, mask_pt: str):
    """若 pt 文件不存在，生成 mock 数据。"""
    if Path(graph_pt).exists() and Path(y_pt).exists() and Path(mask_pt).exists():
        return

    torch.manual_seed(42)
    num_nodes = 101
    num_edges = 340

    x_cont = torch.randn(num_nodes, 11).float()
    x_cont = (x_cont - x_cont.mean(dim=0)) / (x_cont.std(dim=0) + 1e-8)
    func_type = torch.randint(0, 2, (num_nodes,)).long()

    seen = set()
    edge_list = []
    for i in range(num_nodes - 1):
        a, b = i, i + 1
        seen.add((min(a, b), max(a, b)))
        edge_list.extend([[a, b], [b, a]])
    while len(seen) < num_edges:
        u, v = torch.randint(0, num_nodes, (2,)).tolist()
        if u != v:
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                edge_list.extend([[u, v], [v, u]])
    edge_index = torch.tensor(edge_list, dtype=torch.long).T

    labeled = torch.randperm(num_nodes)[:20]
    y_time = torch.zeros(42, num_nodes).float()
    mask_time = torch.zeros(42, num_nodes, dtype=torch.bool)
    for t in range(42):
        mask_time[t, labeled] = True
        y_time[t, labeled] = torch.log1p(torch.rand(20).float() * 10 + 0.1)

    Path(graph_pt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x_cont": x_cont, "func_type": func_type, "edge_index": edge_index}, graph_pt)
    torch.save(y_time, y_pt)
    torch.save(mask_time, mask_pt)
    print(f"已生成 mock 数据: {graph_pt}, {y_pt}, {mask_pt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--kfold_by", type=str, default="day")
    parser.add_argument("--graph_pt", type=str, default="graph_cache_static.pt")
    parser.add_argument("--y_pt", type=str, default="y_time.pt")
    parser.add_argument("--mask_pt", type=str, default="mask_time.pt")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cpu")

    base = Path(__file__).parent
    graph_pt = base / args.graph_pt
    y_pt = base / args.y_pt
    mask_pt = base / args.mask_pt

    generate_mock_cache_if_missing(str(graph_pt), str(y_pt), str(mask_pt))

    graph, y_time, mask_time = load_data(str(graph_pt), str(y_pt), str(mask_pt))
    dataset = build_42_graph_views(graph, y_time, mask_time, use_day_of_week=True)
    print(f"数据集大小: {len(dataset)} 个时间片")

    if args.kfold_by != "day":
        print("当前仅支持 --kfold_by day")
        args.kfold_by = "day"

    print("\n--- GCN 7-fold by day ---")
    gcn_maes = run_cv_gcn(dataset, args, device)
    gcn_mean = sum(gcn_maes) / len(gcn_maes)
    gcn_std = (sum((m - gcn_mean) ** 2 for m in gcn_maes) / len(gcn_maes)) ** 0.5
    print(f"GCN MAE(log): {gcn_mean:.6f} ± {gcn_std:.6f}")

    print("\n--- MLP Baseline 7-fold by day ---")
    mlp_maes = run_cv_mlp(dataset, args, device)
    mlp_mean = sum(mlp_maes) / len(mlp_maes)
    mlp_std = (sum((m - mlp_mean) ** 2 for m in mlp_maes) / len(mlp_maes)) ** 0.5
    print(f"MLP MAE(log): {mlp_mean:.6f} ± {mlp_std:.6f}")

    print("\n--- 对比 ---")
    print(f"GCN vs MLP: {gcn_mean:.6f} vs {mlp_mean:.6f}")


if __name__ == "__main__":
    main()

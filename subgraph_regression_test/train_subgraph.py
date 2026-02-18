"""
子图回归训练：5-fold CV，GCN 与 MLP baseline，早停。
"""
import argparse

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold

from data_io import load_graph
from build_subgraphs import build_dataset
from models import GCNNodeRegressor, MLPCenterBaseline
from utils import set_seed, metric_mae


def train_one_fold_gcn(
    dataset, train_labeled_ids, val_labeled_ids, args, device
):
    """
    Masked supervision: 仅对有标签样本（center in train_labeled_ids）计算回归损失。
    全部 101 个子图参与前向，但 loss 只对 train_labeled_ids 中的样本计算。
    """
    emb_dim = getattr(args, "emb_dim", 8)
    hidden = getattr(args, "hidden", 64)
    model = GCNNodeRegressor(
        in_cont=11, num_func_types=2, emb_dim=emb_dim, hidden=hidden
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.HuberLoss() if args.use_huber else nn.L1Loss()
    train_set = set(train_labeled_ids.tolist())

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    val_data = [d for d in dataset if d.center_global_id.item() in val_labeled_ids.tolist()]
    best_val_mae = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred = model(batch)
            center_preds = []
            for i in range(batch.num_graphs):
                idx = batch.ptr[i].item() + batch.center_idx[i].item()
                center_preds.append(pred[idx])
            center_preds = torch.stack(center_preds)
            # Masked: 仅对 train_labeled_ids 中的样本计算 loss
            train_mask = torch.tensor(
                [batch.center_global_id[i].item() in train_set for i in range(batch.num_graphs)],
                device=device, dtype=torch.bool,
            )
            if train_mask.any():
                loss = criterion(center_preds[train_mask], batch.y_center[train_mask])
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            val_preds, val_targets = [], []
            for d in val_data:
                d = d.to(device)
                pred = model(Batch.from_data_list([d]))
                idx = d.center_idx.item()
                val_preds.append(pred[idx].item())
                val_targets.append(d.y_center.item())
            val_mae = metric_mae(torch.tensor(val_preds), torch.tensor(val_targets))

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    return best_val_mae


def run_cv_gcn(dataset, labeled_node_ids, args, device):
    """5-fold CV 按 20 个有标签节点划分，非按 101 个子图划分。"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    maes = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(labeled_node_ids)))):
        set_seed(42 + fold)
        train_labeled_ids = labeled_node_ids[train_idx]
        val_labeled_ids = labeled_node_ids[val_idx]
        mae = train_one_fold_gcn(dataset, train_labeled_ids, val_labeled_ids, args, device)
        maes.append(mae)
        print(f"  Fold {fold + 1}: best val MAE = {mae:.6f}")
    return maes


def run_cv_mlp(graph, labeled_indices, args, device):
    """MLP: 按子图索引划分，每个子图对应一个 labeled 中心；中心特征从整图取。"""
    x_cont = graph["x_cont"]
    func_type = graph["func_type"]
    y = graph["y"]
    labeled_node_ids = graph["labeled_node_ids"]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    maes = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(labeled_node_ids)))):
        set_seed(42 + fold)
        emb_dim = getattr(args, "emb_dim", 8)
        hidden = getattr(args, "hidden", 64)
        model = MLPCenterBaseline(in_cont=11, num_func_types=2, emb_dim=emb_dim, hidden=hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.HuberLoss() if args.use_huber else nn.L1Loss()

        train_ids = labeled_node_ids[train_idx]
        val_ids = labeled_node_ids[val_idx]

        best_val_mae = float("inf")
        patience_counter = 0

        for epoch in range(args.epochs):
            model.train()
            x = x_cont[train_ids].to(device)
            ft = func_type[train_ids].to(device)
            tgt = y[train_ids].to(device)
            opt.zero_grad()
            pred = model(x, ft)
            loss = criterion(pred, tgt)
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                xv = x_cont[val_ids].to(device)
                ftv = func_type[val_ids].to(device)
                yv = y[val_ids].to(device)
                predv = model(xv, ftv)
                val_mae = metric_mae(predv, yv)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    break

        maes.append(best_val_mae)
        print(f"  Fold {fold + 1}: best val MAE = {best_val_mae:.6f}")
    return maes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=2, help="k-hop ego-subgraph")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--use_huber", type=int, default=1, help="1=Huber, 0=L1")
    parser.add_argument("--lr", type=float, default=0.009988079371192426)
    parser.add_argument("--weight_decay", type=float, default=0.009918206498117957)
    parser.add_argument("--patience", type=int, default=41)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--emb_dim", type=int, default=8, help="embedding dim for func_type")
    parser.add_argument("--hidden", type=int, default=32, help="hidden dim")
    parser.add_argument("--pt_path", type=str, default="graph_cache.pt")
    parser.add_argument("--nodes", type=str, default=None, help="nodes.csv (with --edges --flow to load from CSV)")
    parser.add_argument("--edges", type=str, default=None, help="edges.csv")
    parser.add_argument("--flow", type=str, default=None, help="flow.csv")
    args = parser.parse_args()
    args.use_huber = bool(args.use_huber)

    set_seed(42)
    device = torch.device("cpu")

    graph = load_graph(
        nodes_csv=args.nodes,
        edges_csv=args.edges,
        flow_csv=args.flow,
        pt_path=args.pt_path,
    )
    labeled_node_ids = graph["labeled_node_ids"]
    dataset = build_dataset(graph, k=args.k)
    print(f"子图数量: {len(dataset)}, k={args.k}, 有标签中心: {len(labeled_node_ids)}")

    print("\n--- GCN 5-fold CV (masked supervision) ---")
    gcn_maes = run_cv_gcn(dataset, labeled_node_ids, args, device)
    gcn_mean = sum(gcn_maes) / len(gcn_maes)
    gcn_std = (sum((m - gcn_mean) ** 2 for m in gcn_maes) / len(gcn_maes)) ** 0.5
    print(f"GCN MAE(log): {gcn_mean:.6f} ± {gcn_std:.6f}")

    print("\n--- MLP Baseline 5-fold CV ---")
    mlp_maes = run_cv_mlp(graph, labeled_node_ids, args, device)
    mlp_mean = sum(mlp_maes) / len(mlp_maes)
    mlp_std = (sum((m - mlp_mean) ** 2 for m in mlp_maes) / len(mlp_maes)) ** 0.5
    print(f"MLP MAE(log): {mlp_mean:.6f} ± {mlp_std:.6f}")


if __name__ == "__main__":
    main()

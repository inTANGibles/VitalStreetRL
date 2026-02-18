"""
Optuna 超参数搜索：对 GCN 子图回归做 5-fold CV 调参。
"""
import argparse
from types import SimpleNamespace

import numpy as np
import optuna
import torch

from data_io import load_graph
from build_subgraphs import build_dataset
from train_subgraph import run_cv_gcn, run_cv_mlp
from utils import set_seed


def make_args(trial, base_args):
    """从 trial 采样超参数，构造 args 对象"""
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    hidden = trial.suggest_categorical("hidden", [32, 64, 128])
    emb_dim = trial.suggest_categorical("emb_dim", [4, 8, 16])
    patience = trial.suggest_int("patience", 20, 80)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    return SimpleNamespace(
        lr=lr,
        weight_decay=weight_decay,
        hidden=hidden,
        emb_dim=emb_dim,
        patience=patience,
        batch_size=batch_size,
        epochs=base_args.epochs,
        use_huber=base_args.use_huber,
        k=base_args.k,
    )


def objective_gcn(trial, dataset, labeled_node_ids, base_args, device):
    """GCN 目标函数：5-fold CV 平均 MAE"""
    args = make_args(trial, base_args)
    maes = run_cv_gcn(dataset, labeled_node_ids, args, device)
    return np.mean(maes)


def objective_mlp(trial, graph, labeled_node_ids, base_args, device):
    """MLP 目标函数：5-fold CV 平均 MAE"""
    args = make_args(trial, base_args)
    maes = run_cv_mlp(graph, labeled_node_ids, args, device)
    return np.mean(maes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "mlp"])
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--use_huber", type=int, default=1)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--pt_path", type=str, default="graph_cache.pt")
    parser.add_argument("--study_name", type=str, default="subgraph_regression")
    parser.add_argument("--storage", type=str, default=None, help="SQLite path for study persistence")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()
    args.use_huber = bool(args.use_huber)

    set_seed(42)
    device = torch.device("cpu")

    graph = load_graph(pt_path=args.pt_path)
    labeled_node_ids = graph["labeled_node_ids"]
    dataset = build_dataset(graph, k=args.k)
    base_args = SimpleNamespace(epochs=args.epochs, use_huber=args.use_huber, k=args.k)

    print(f"子图数量: {len(dataset)}, 有标签中心: {len(labeled_node_ids)}")
    print(f"Optuna 搜索: model={args.model}, n_trials={args.n_trials}\n")

    optuna.logging.set_verbosity(optuna.logging.INFO if args.verbose else optuna.logging.WARNING)

    storage = f"sqlite:///{args.storage}" if args.storage else None
    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        storage=storage,
        load_if_exists=bool(args.storage),
    )

    if args.model == "gcn":
        study.optimize(
            lambda t: objective_gcn(t, dataset, labeled_node_ids, base_args, device),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )
    else:
        study.optimize(
            lambda t: objective_mlp(t, graph, labeled_node_ids, base_args, device),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )

    print("\n" + "=" * 50)
    print("最佳超参数:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"\n最佳 5-fold 平均 MAE: {study.best_value:.6f}")


if __name__ == "__main__":
    main()

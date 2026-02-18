#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单测试脚本：验证 Optuna 与 matplotlib savefig 是否正常（含 PyTorch 同进程时的 OMP 兼容）。
用法: python scripts/test_optuna_savefig.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import optuna


def test_optuna():
    """跑一个最小 Optuna  study（2 个 trial）。"""
    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -2.0, 2.0)
        return x ** 2

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2, show_progress_bar=False)
    best = study.best_value
    assert best >= 0, "best value should be non-negative"
    print("  Optuna: 2 trials done, best value =", round(best, 6))
    return True


def test_savefig(out_dir: Path) -> bool:
    """画一张简单图并保存，模拟 run_pipeline 的 savefig 路径。"""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([1, 2, 3], [1, 4, 2], "o-")
    ax.set_title("test_optuna_savefig")
    out_path = out_dir / "test_figure.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    assert out_path.exists(), f"expected file: {out_path}"
    print("  savefig: saved", out_path)
    return True


def main():
    out_dir = ROOT / "outputs" / "test_optuna_savefig"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Output dir:", out_dir)
    print("PyTorch:", torch.__version__, "| Optuna:", optuna.__version__)

    print("\n1. Testing Optuna...")
    test_optuna()

    print("\n2. Testing savefig (with PyTorch already loaded)...")
    test_savefig(out_dir)

    print("\nDone. Optuna and savefig OK.")


if __name__ == "__main__":
    main()

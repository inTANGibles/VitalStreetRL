"""
工具函数：随机种子、masked MAE、早停。
"""
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """
    log1p(flow) 空间的 MAE，仅在 mask=True 的节点上计算。
    pred, target, mask: [N] 或 [B, N]
    """
    if mask.sum() == 0:
        return float("nan")
    diff = (pred - target).abs()
    return diff[mask].mean().item()


class EarlyStopping:
    """早停：val 指标不再改善时停止。"""

    def __init__(self, patience: int = 50, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def __call__(self, score: float, lower_is_better: bool = True) -> bool:
        """
        score: 当前验证指标（如 val MAE）
        lower_is_better: True 表示越小越好
        返回 True 表示应停止。
        """
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
            return False

        improved = (score < self.best_score - self.min_delta) if lower_is_better else (
            score > self.best_score + self.min_delta
        )
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

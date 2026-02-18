"""
工具函数：随机种子、MAE 指标。
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


def metric_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """log1p(flow) 空间的 MAE。"""
    return torch.nn.functional.l1_loss(pred, target).item()

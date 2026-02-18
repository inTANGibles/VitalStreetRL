"""MAE and RMSE for masked regression."""
import torch


def mae(y_true: torch.Tensor, y_pred: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean Absolute Error on masked elements. mask=True where valid."""
    if mask.sum() == 0:
        return float("nan")
    return (torch.abs(y_true - y_pred)[mask]).mean().item()


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor, mask: torch.Tensor) -> float:
    """Root Mean Squared Error on masked elements."""
    if mask.sum() == 0:
        return float("nan")
    return torch.sqrt(((y_true - y_pred)[mask] ** 2).mean()).item()

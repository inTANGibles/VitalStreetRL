#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
remap_flow.py

客流密度数据两阶段重映射（参考图 3-22）：
1. Box-Cox 变换（λ=1/6）：拉开小数值、缩小大数值，使分布更均衡
2. 线性映射：每个建筑内非零值映射到 [1, 10]，消除量纲差异

用于 GNN 等下游任务的客流效能指标。
"""

from typing import Optional

import numpy as np


def box_cox_transform(y: np.ndarray, lam: float = 1.0 / 6.0) -> np.ndarray:
    """
    Box-Cox 变换：y(λ) = ((y+1)^λ - 1) / λ
    使用 y+1 避免 y=0 时 0^λ 未定义，且保持 0 -> 0。
    NaN 保持为 NaN。

    Args:
        y: 原始客流密度（非负，可含 NaN）
        lam: 变换参数，默认 1/6

    Returns:
        变换后的值
    """
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y) & (y >= 0)
    y_adj = np.where(valid, y + 1.0, 1.0)
    out = (np.power(y_adj, lam) - 1.0) / lam
    return np.where(valid, out, np.nan)


def linear_map_to_1_10(
    y: np.ndarray,
    group_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    每个分组内，将非零值线性映射到 [1, 10]。
    零值保持为 0，NaN 保持为 NaN。

    Args:
        y: 变换后的客流值（可含 NaN）
        group_ids: 分组 ID（如建筑 ID），同长度。None 表示全部为一组。

    Returns:
        映射到 [1, 10] 的值，零保持 0，NaN 保持 NaN
    """
    y = np.asarray(y, dtype=float)
    out = np.where(np.isfinite(y), 0.0, np.nan)
    if group_ids is None:
        group_ids = np.zeros(len(y), dtype=int)
    groups = np.unique(group_ids)
    for g in groups:
        mask = (group_ids == g) & np.isfinite(y)
        vals = y[mask]
        nonzero = vals > 0
        if not np.any(nonzero):
            continue
        v_min = np.min(vals[nonzero])
        v_max = np.max(vals[nonzero])
        if v_max <= v_min:
            out[mask] = np.where(nonzero, 1.0, 0.0)
        else:
            mapped = 1.0 + 9.0 * (vals - v_min) / (v_max - v_min)
            out[mask] = np.where(nonzero, mapped, 0.0)
    return out


def remap_flow(
    flow_raw: np.ndarray,
    lam: float = 1.0 / 6.0,
    group_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    两阶段重映射：Box-Cox + 线性映射到 [1, 10]。
    NaN 保持为 NaN（无标签节点）。

    Args:
        flow_raw: 原始客流密度（可含 NaN）
        lam: Box-Cox 参数
        group_ids: 分组（如建筑），None 表示单组

    Returns:
        重映射后的客流效能指标 [NaN 或 0 或 1-10]
    """
    y_tr = box_cox_transform(flow_raw, lam=lam)
    return linear_map_to_1_10(y_tr, group_ids=group_ids)

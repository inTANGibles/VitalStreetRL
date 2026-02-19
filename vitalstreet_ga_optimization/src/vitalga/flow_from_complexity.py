"""基于周边图块功能复杂度计算 flow_prediction（迁移自 scripts/compute_flow_from_complexity）"""
from collections import Counter
from typing import Dict, List, Optional
import numpy as np
import geopandas as gpd

from .state import SpaceUnitCollection


def _shannon_entropy(categories: List[str]) -> float:
    if len(categories) == 0:
        return 0.0
    counter = Counter(categories)
    total = len(categories)
    entropy = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    max_entropy = np.log2(len(counter)) if len(counter) > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _category_weight(category: Optional[str]) -> float:
    if category is None:
        return 0.0
    weights = {
        'dining': 1.0, 'retail': 0.9, 'leisure': 0.8, 'cultural': 0.7,
        'service': 0.6, 'residential': 0.4, 'supporting': 0.3, 'undefined': 0.2,
    }
    return weights.get(category.lower(), 0.2)


def _compute_surrounding_complexity(
    target_unit: gpd.GeoSeries,
    all_units: gpd.GeoDataFrame,
    buffer_distance: float,
    exclude_self: bool = True,
) -> Dict[str, float]:
    geom = target_unit['geometry']
    if geom is None or geom.is_empty:
        return {'diversity': 0.0, 'weighted_sum': 0.0, 'total_area': 0.0, 'count': 0}
    try:
        buffer_geom = geom.buffer(buffer_distance)
        if buffer_geom is None or buffer_geom.is_empty:
            return {'diversity': 0.0, 'weighted_sum': 0.0, 'total_area': 0.0, 'count': 0}
    except Exception:
        return {'diversity': 0.0, 'weighted_sum': 0.0, 'total_area': 0.0, 'count': 0}
    try:
        overlapping = all_units[all_units.geometry.intersects(buffer_geom)].copy()
    except Exception:
        return {'diversity': 0.0, 'weighted_sum': 0.0, 'total_area': 0.0, 'count': 0}
    if exclude_self and 'uid' in target_unit.index:
        overlapping = overlapping[overlapping['uid'] != target_unit['uid']]
    if len(overlapping) == 0:
        return {'diversity': 0.0, 'weighted_sum': 0.0, 'total_area': 0.0, 'count': 0}
    categories, areas, weights = [], [], []
    for _, unit in overlapping.iterrows():
        if unit.get('unit_type', '') != 'shop':
            continue
        cat = unit.get('business_category', None)
        if cat is None:
            cat_str = 'undefined'
        elif isinstance(cat, str):
            cat_str = cat.lower()
        elif hasattr(cat, 'value'):
            cat_str = cat.value.lower()
        else:
            cat_str = str(cat).lower()
        categories.append(cat_str)
        areas.append(unit.get('area', 0.0))
        weights.append(_category_weight(cat_str))
    if len(categories) == 0:
        return {'diversity': 0.0, 'weighted_sum': 0.0, 'total_area': 0.0, 'count': 0}
    total_area = sum(areas)
    weighted_sum = sum(w * a for w, a in zip(weights, areas)) / total_area if total_area > 0 else 0.0
    return {
        'diversity': _shannon_entropy(categories),
        'weighted_sum': weighted_sum,
        'total_area': total_area,
        'count': len(categories),
    }


def _compute_surrounding_public_space_flow(
    target_unit: gpd.GeoSeries,
    all_units: gpd.GeoDataFrame,
    buffer_distance: float,
    exclude_self: bool = True,
) -> float:
    geom = target_unit['geometry']
    if geom is None or geom.is_empty:
        return 0.0
    buffer_geom = geom.buffer(buffer_distance)
    public_mask = all_units['unit_type'] == 'public_space'
    overlapping = all_units[public_mask & all_units.geometry.intersects(buffer_geom)].copy()
    if exclude_self and 'uid' in target_unit.index:
        overlapping = overlapping[overlapping['uid'] != target_unit['uid']]
    if len(overlapping) == 0:
        return 0.0
    total_wf, total_area = 0.0, 0.0
    for _, unit in overlapping.iterrows():
        flow = unit.get('flow_prediction', 0.0)
        area = unit.get('area', 0.0)
        if area > 0 and flow > 0:
            total_wf += flow * area
            total_area += area
    return total_wf / total_area if total_area > 0 else 0.0


def compute_flow_from_complexity(
    collection: SpaceUnitCollection,
    buffer_distance: float = 10.0,
    base_flow: float = 0.0,
    diversity_weight: float = 0.5,
    weighted_sum_weight: float = 0.5,
    normalize: bool = True,
    self_weight: float = 0.6,
    surrounding_weight: float = 0.4,
) -> SpaceUnitCollection:
    """只针对 public_space 更新 flow_prediction；原地修改 collection。"""
    all_units = collection._SpaceUnitCollection__unit_gdf
    if len(all_units) == 0:
        return collection
    shop_mask = all_units['unit_type'] == 'shop'
    if shop_mask.any():
        all_units.loc[shop_mask, 'flow_prediction'] = 0.0
    public_units = all_units[all_units['unit_type'] == 'public_space']
    if len(public_units) == 0:
        return collection
    complexity_scores = []
    public_indices = []
    for idx, unit in public_units.iterrows():
        c = _compute_surrounding_complexity(
            target_unit=unit, all_units=all_units,
            buffer_distance=buffer_distance, exclude_self=True,
        )
        score = diversity_weight * c['diversity'] + weighted_sum_weight * c['weighted_sum']
        complexity_scores.append(score)
        public_indices.append(idx)
    if normalize and len(complexity_scores) > 0:
        arr = np.array(complexity_scores)
        mn, mx = arr.min(), arr.max()
        complexity_scores = (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)
    initial_flows = {}
    for idx, score in zip(public_indices, complexity_scores):
        initial_flows[idx] = base_flow + float(score)
        all_units.loc[idx, 'flow_prediction'] = initial_flows[idx]
    final_flows = {}
    for idx, unit in public_units.iterrows():
        self_flow = initial_flows.get(idx, 0.0)
        sur = _compute_surrounding_public_space_flow(
            target_unit=unit, all_units=all_units,
            buffer_distance=buffer_distance, exclude_self=True,
        )
        final_flows[idx] = self_weight * self_flow + surrounding_weight * sur
    for idx, f in final_flows.items():
        all_units.loc[idx, 'flow_prediction'] = f
    if shop_mask.any():
        all_units.loc[shop_mask, 'flow_prediction'] = 0.0
    return collection

"""活力指标：VitalityMetrics.compute，三项分解 mixedness / vacancy / concentration（Shannon / vacancy / HHI）"""
from typing import Dict, Any
from collections import Counter
import numpy as np
from ..state import WorldState


class VitalityMetrics:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def compute(self, F_hat: np.ndarray, state: WorldState) -> np.ndarray:
        """返回 [mixedness, vacancy_rate, concentration]"""
        mixedness = self._compute_business_mixedness(state)
        vacancy_rate = self._compute_vacancy_rate(state)
        concentration = self._compute_concentration(state)
        return np.array([mixedness, vacancy_rate, concentration])

    def _compute_business_mixedness(self, state: WorldState) -> float:
        all_units = state.space_units.get_all_space_units()
        shops = all_units[all_units['unit_type'] == 'shop']
        if len(shops) == 0:
            return 0.0
        business_types = shops['business_type'].values
        type_counts = Counter(business_types)
        valid_types = {k: v for k, v in type_counts.items() if k not in ['UNDEFINED', 'N/A']}
        if len(valid_types) == 0:
            return 0.0
        total = sum(valid_types.values())
        proportions = [c / total for c in valid_types.values()]
        shannon = -sum(p * np.log(p) for p in proportions if p > 0)
        max_entropy = np.log(len(valid_types))
        return shannon / max_entropy if max_entropy > 0 else 0.0

    def _compute_vacancy_rate(self, state: WorldState) -> float:
        all_units = state.space_units.get_all_space_units()
        shops = all_units[all_units['unit_type'] == 'shop']
        if len(shops) == 0:
            return 0.0
        vacant = shops[
            (shops['business_type'].isin(['UNDEFINED', 'N/A'])) | (shops['enabled'] == False)
        ]
        return len(vacant) / len(shops)

    def _compute_concentration(self, state: WorldState) -> float:
        all_units = state.space_units.get_all_space_units()
        shops = all_units[all_units['unit_type'] == 'shop']
        if len(shops) == 0:
            return 0.0
        type_counts = Counter(shops['business_type'].values)
        valid = {k: v for k, v in type_counts.items() if k not in ['UNDEFINED', 'N/A']}
        if len(valid) == 0:
            return 1.0
        total = sum(valid.values())
        proportions = [c / total for c in valid.values()]
        hhi = sum(p ** 2 for p in proportions)
        n_types = len(valid)
        if n_types == 1:
            return 1.0
        min_hhi = 1.0 / n_types
        return (hhi - min_hhi) / (1.0 - min_hhi)

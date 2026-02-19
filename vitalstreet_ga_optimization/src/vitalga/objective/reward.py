"""奖励计算：_compute_vitality、reward = delta_v - mu_violation * violation"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
from ..state import WorldState
from ..action_space import Action
from .vitality_metrics import VitalityMetrics


class RewardCalculator:
    def __init__(self, config: Dict[str, Any]):
        self.mu_violation = config.get('mu_violation', 1.0)
        self.initial_vitality: Optional[float] = None
        vitality_config = config.get('vitality_metrics', {})
        self.vitality_metrics = VitalityMetrics(vitality_config)
        self.initial_mixedness: Optional[float] = None
        self.initial_vacancy_rate: Optional[float] = None
        self.initial_concentration: Optional[float] = None

    def reset(self, initial_state: WorldState):
        self.initial_vitality = self._compute_vitality(initial_state)
        v_vec = self.vitality_metrics.compute(F_hat=np.array([]), state=initial_state)
        self.initial_mixedness = float(v_vec[0])
        self.initial_vacancy_rate = float(v_vec[1])
        self.initial_concentration = float(v_vec[2])

    def compute(
        self,
        state: WorldState,
        action: Action,
        next_state: WorldState,
        V_prev: Optional[float] = None,
        V_next: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        if V_prev is None:
            V_prev = self._compute_vitality(state)
        if V_next is None:
            V_next = self._compute_vitality(next_state)
        delta_v = self._compute_vitality_change(V_prev, V_next)
        violation = self._compute_violation(next_state)
        reward = delta_v - self.mu_violation * violation
        current_v_vec = self.vitality_metrics.compute(F_hat=np.array([]), state=next_state)
        reward_terms = {
            'delta_vitality': delta_v,
            'vitality_current': V_next,
            'vitality_initial': self.initial_vitality or 0.0,
            'violation': violation,
            'mixedness_current': float(current_v_vec[0]),
            'mixedness_initial': self.initial_mixedness or 0.0,
            'vacancy_rate_current': float(current_v_vec[1]),
            'vacancy_rate_initial': self.initial_vacancy_rate or 0.0,
            'concentration_current': float(current_v_vec[2]),
            'concentration_initial': self.initial_concentration or 0.0,
            'total': reward,
        }
        return reward, reward_terms

    def _compute_vitality(self, state: WorldState) -> float:
        public_spaces = state.space_units.get_public_spaces()
        if len(public_spaces) == 0:
            return 0.0
        areas = public_spaces['area'].values
        flows = np.nan_to_num(public_spaces['flow_prediction'].values, nan=0.0)
        return float(np.sum(areas * flows))

    def _compute_vitality_change(self, V_prev: float, V_next: float) -> float:
        if self.initial_vitality is None:
            self.initial_vitality = V_prev
        delta_v = V_next - self.initial_vitality
        if abs(self.initial_vitality) > 1e-6:
            return delta_v / abs(self.initial_vitality)
        return delta_v / 1000.0

    def _compute_violation(self, state: WorldState) -> float:
        if (
            self.initial_mixedness is None
            or self.initial_vacancy_rate is None
            or self.initial_concentration is None
        ):
            return 0.0
        v_vec = self.vitality_metrics.compute(F_hat=np.array([]), state=state)
        dm = float(v_vec[0]) - self.initial_mixedness
        dv = float(v_vec[1]) - self.initial_vacancy_rate
        dc = float(v_vec[2]) - self.initial_concentration
        mixedness_penalty = max(0.0, -dm)
        vacancy_penalty = max(0.0, dv)
        concentration_penalty = max(0.0, dc)
        return mixedness_penalty + vacancy_penalty + concentration_penalty

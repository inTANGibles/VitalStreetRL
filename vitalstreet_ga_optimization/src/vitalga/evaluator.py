"""执行动作序列、累计 reward、输出 final_vitality / violation 分解 / 操作统计；方案A evaluate_genome 无时序。"""
from typing import Dict, Any, List, Tuple, Callable, Optional
import numpy as np
from .state import WorldState
from .action_space import ActionSpace, ActionType
from .transition import Transition
from .objective.reward import RewardCalculator
from .objective.vitality_metrics import VitalityMetrics


def evaluate_genome(
    genome: np.ndarray,
    initial_state: WorldState,
    editable_nodes: List[Any],
    decode_to_actions_fn: Callable[..., List],
    apply_fn: Callable[[WorldState, List], WorldState],
    objective_fn: Callable[[WorldState, WorldState, List], Dict[str, float]],
    constraints_fn: Optional[Callable[[WorldState], bool]] = None,
) -> Dict[str, float]:
    """方案A：Genome -> 动作列表 -> 应用 -> 目标值。objective_fn(initial_state, state_after, actions) 便于用 initial 算 violation。"""
    actions = decode_to_actions_fn(genome, editable_nodes)
    state_after = apply_fn(initial_state, actions)
    if constraints_fn is not None and not constraints_fn(state_after):
        return {"final_vitality": -1e9, "violation_total": 1e9, "cost_proxy": 1e9}
    return objective_fn(initial_state, state_after, actions)


class Evaluator:
    """给定初始状态与动作序列，逐步执行并汇总指标"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_steps = config.get('max_steps', 50)
        self.transition = Transition(config.get('transition', {}))
        reward_cfg = config.get('reward', {})
        self.reward_calc = RewardCalculator(reward_cfg)
        self.vitality_metrics = VitalityMetrics(reward_cfg.get('vitality_metrics', {}))
        self.action_space = ActionSpace(config.get('action_space', {}))

    def evaluate(
        self,
        initial_state: WorldState,
        action_sequence: np.ndarray,
    ) -> Dict[str, Any]:
        """
        action_sequence: (L, 3) 每行 [action_type, unit_index, type_id]，L <= max_steps。
        返回：total_reward, final_vitality, violation_total, violation_breakdown,
              n_change_business, n_shop_to_public, cost_proxy, done_reason, step_count, ...
        """
        state = initial_state.copy()
        self.reward_calc.reset(state)
        total_reward = 0.0
        n_change_business = 0
        n_shop_to_public = 0
        L = min(len(action_sequence), self.max_steps)
        done_reason = "max_steps"
        step_count = 0
        for t in range(L):
            if state.budget < 0:
                done_reason = "budget_exceeded"
                break
            encoded = action_sequence[t]
            if len(encoded) < 3:
                encoded = np.array([int(encoded[0]), int(encoded[1]) if len(encoded) > 1 else 0, 0])
            action = self.action_space.decode(encoded, state)
            if action.type == ActionType.NO_OP:
                state.step_idx += 1
                step_count += 1
                continue
            if not self.action_space.is_valid(state, action):
                state.step_idx += 1
                step_count += 1
                continue
            next_state, info = self.transition.step(state, action)
            V_prev = self.reward_calc._compute_vitality(state)
            V_next = self.reward_calc._compute_vitality(next_state)
            reward, terms = self.reward_calc.compute(state, action, next_state, V_prev=V_prev, V_next=V_next)
            total_reward += reward
            if action.type == ActionType.CHANGE_BUSINESS:
                n_change_business += 1
            elif action.type == ActionType.SHOP_TO_PUBLIC_SPACE:
                n_shop_to_public += 1
            state = next_state
            step_count += 1
        final_vitality = self.reward_calc._compute_vitality(state)
        v_vec = self.vitality_metrics.compute(np.array([]), state)
        violation_total = self.reward_calc._compute_violation(state)
        violation_breakdown = {
            'mixedness_penalty': max(0.0, self.reward_calc.initial_mixedness - float(v_vec[0]))
            if self.reward_calc.initial_mixedness is not None else 0.0,
            'vacancy_penalty': max(0.0, float(v_vec[1]) - self.reward_calc.initial_vacancy_rate)
            if self.reward_calc.initial_vacancy_rate is not None else 0.0,
            'concentration_penalty': max(0.0, float(v_vec[2]) - self.reward_calc.initial_concentration)
            if self.reward_calc.initial_concentration is not None else 0.0,
        }
        alpha = self.config.get('cost_alpha')
        beta = self.config.get('cost_beta')
        if alpha is None or beta is None:
            raise ValueError("config 必须包含 cost_alpha 与 cost_beta")
        cost_proxy = alpha * n_shop_to_public + beta * n_change_business
        initial_vitality = self.reward_calc.initial_vitality or 0.0
        delta_vitality = final_vitality - initial_vitality
        return {
            'total_reward': total_reward,
            'final_vitality': final_vitality,
            'initial_vitality': initial_vitality,
            'delta_vitality': delta_vitality,
            'violation_total': violation_total,
            'violation_breakdown': violation_breakdown,
            'n_change_business': n_change_business,
            'n_shop_to_public': n_shop_to_public,
            'cost_proxy': cost_proxy,
            'done_reason': done_reason,
            'step_count': step_count,
        }

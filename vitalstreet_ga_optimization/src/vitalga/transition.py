"""状态转移：Transition.step，含 SHOP_TO_PUBLIC_SPACE 后 flow_prediction 更新"""
from typing import Tuple, Dict, Any
from .state import WorldState
from .action_space import Action, ActionType
from .flow_from_complexity import compute_flow_from_complexity


class Transition:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def step(self, state: WorldState, action: Action) -> Tuple[WorldState, Dict[str, Any]]:
        next_state = state.copy()
        info = {}
        if action.type == ActionType.NO_OP:
            next_state.step_idx += 1
            return next_state, {'action': 'no_op'}
        if action.type == ActionType.CHANGE_BUSINESS:
            next_state, info = self._apply_change_business(next_state, action)
        elif action.type == ActionType.SHOP_TO_PUBLIC_SPACE:
            next_state, info = self._apply_shop_to_public_space(next_state, action)
        else:
            next_state.step_idx += 1
            return next_state, {'action': 'no_op_disabled'}
        next_state.step_idx += 1
        return next_state, info

    def _apply_change_business(self, state: WorldState, action: Action) -> Tuple[WorldState, Dict[str, Any]]:
        state.space_units.update_business_type(action.target_id, action.params['new_type'])
        return state, {'action': 'change_business'}

    def _apply_shop_to_public_space(self, state: WorldState, action: Action) -> Tuple[WorldState, Dict[str, Any]]:
        state.space_units.convert_to_public_space(action.target_id)
        state.graph = None
        flow_config = self.config.get('flow', {})
        compute_flow_from_complexity(
            collection=state.space_units,
            buffer_distance=flow_config.get('buffer_distance', 10.0),
            diversity_weight=flow_config.get('diversity_weight', 0.5),
            weighted_sum_weight=flow_config.get('weighted_sum_weight', 0.5),
        )
        return state, {'action': 'shop_to_public_space'}

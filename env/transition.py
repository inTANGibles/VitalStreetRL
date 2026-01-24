"""状态转移：动作执行与状态更新"""
from typing import Tuple, Dict, Any
from .world_state import WorldState
from .action_space import Action, ActionType
from scripts.compute_flow_from_complexity import compute_flow_from_complexity


class Transition:
    """状态转移引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化转移规则"""
        self.config = config
    
    def step(self, state: WorldState, action: Action) -> Tuple[WorldState, Dict[str, Any]]:
        """
        执行动作，返回新状态和信息
        
        Returns:
            next_state: 更新后的状态
            info: 包含更新详情、合法性检查等
        """
        next_state = state.copy()
        info = {}
        
        if action.type == ActionType.NO_OP:
            next_state.step_idx += 1
            return next_state, {'cost': 0, 'action': 'no_op'}
        
        # 策略1: 业态置换
        if action.type == ActionType.CHANGE_BUSINESS:
            next_state, info = self._apply_change_business(next_state, action)
        
        # 策略2: 街道打通与优化
        elif action.type == ActionType.SHOP_TO_PUBLIC_SPACE:
            next_state, info = self._apply_shop_to_public_space(next_state, action)
        elif action.type == ActionType.ATRIUM_TO_PUBLIC_SPACE:
            next_state, info = self._apply_atrium_to_public_space(next_state, action)
        elif action.type == ActionType.CLOSE_PUBLIC_SPACE_NODE:
            next_state, info = self._apply_close_public_space_node(next_state, action)
        
        # 策略3: 公共空间改造（暂时停用）
        elif action.type in [ActionType.WIDEN_PUBLIC_SPACE, ActionType.NARROW_PUBLIC_SPACE,
                            ActionType.GENERATE_POCKET_NODE, ActionType.DISSOLVE_POCKET_NODE,
                            ActionType.REGULARIZE_BOUNDARY]:
            next_state.step_idx += 1
            return next_state, {'cost': 0, 'action': 'no_op_disabled'}
        
        # 更新预算和步数
        next_state.step_idx += 1
        if 'cost' in info:
            next_state.budget -= info['cost']
        
        return next_state, info
    
    # 策略1: 业态置换
    def _apply_change_business(self, state: WorldState, action: Action) -> Tuple[WorldState, Dict[str, Any]]:
        """执行业态替换动作"""
        shop_uid = action.target_id
        new_type = action.params['new_type']
        state.space_units.update_business_type(shop_uid, new_type)
        
        # 成本作为动作惩罚（step penalty）
        cost = self.config.get('cost', {}).get('change_business', 3000.0)
        
        return state, {'cost': cost, 'action': 'change_business'}
    
    # 策略2: 街道打通与优化
    def _apply_shop_to_public_space(self, state: WorldState, action: Action) -> Tuple[WorldState, Dict[str, Any]]:
        """Shop → Public Space"""
        shop_uid = action.target_id
        state.space_units.convert_to_public_space(shop_uid)
        # 更新public_space图
        state.graph = None  # 触发重建
        # 更新所有public_space的flow_prediction（包括新转换的）
        flow_config = self.config.get('flow', {})
        compute_flow_from_complexity(
            collection=state.space_units,
            buffer_distance=flow_config.get('buffer_distance', 10.0),
            diversity_weight=flow_config.get('diversity_weight', 0.5),
            weighted_sum_weight=flow_config.get('weighted_sum_weight', 0.5)
        )
        return state, {'cost': self.config.get('cost_shop_to_public_space', 2000), 'action': 'shop_to_public_space'}
    
    def _apply_atrium_to_public_space(self, state: WorldState, action: Action) -> Tuple[WorldState, Dict[str, Any]]:
        """Atrium → Public Space"""
        atrium_uid = action.target_id
        state.space_units.convert_to_public_space(atrium_uid)
        state.graph = None
        # 更新所有public_space的flow_prediction（包括新转换的）
        flow_config = self.config.get('flow', {})
        compute_flow_from_complexity(
            collection=state.space_units,
            buffer_distance=flow_config.get('buffer_distance', 10.0),
            diversity_weight=flow_config.get('diversity_weight', 0.5),
            weighted_sum_weight=flow_config.get('weighted_sum_weight', 0.5)
        )
        return state, {'cost': self.config.get('cost_atrium_to_public_space', 3000), 'action': 'atrium_to_public_space'}
    
    def _apply_close_public_space_node(self, state: WorldState, action: Action) -> Tuple[WorldState, Dict[str, Any]]:
        """Public Space Node Closure"""
        node_uid = action.target_id
        state.space_units.close_public_space_node(node_uid)
        state.graph = None
        return state, {'cost': self.config.get('cost_close_public_space', 1000), 'action': 'close_public_space'}
    
    # 策略3: 公共空间改造
    def _apply_widen_public_space(self, state: WorldState, action: Action) -> Tuple[WorldState, Dict[str, Any]]:
        """Widen (Building Unit Transfer)"""
        public_space_uid = action.target_id
        adjacent_unit_uid = action.params['adjacent_unit_id']
        state.space_units.widen_public_space(public_space_uid, adjacent_unit_uid)
        state.graph = None
        return state, {'cost': self.config.get('cost_widen', 5000), 'action': 'widen_public_space'}
    
    def _apply_narrow_public_space(self, state: WorldState, action: Action) -> Tuple[WorldState, Dict[str, Any]]:
        """Narrow (Occupancy Zone Insertion)"""
        public_space_uid = action.target_id
        occupancy_zone_id = action.params['occupancy_zone_id']
        zone_type = action.params['zone_type']
        width_level = action.params['width_level']
        state.space_units.narrow_public_space(public_space_uid, occupancy_zone_id, zone_type, width_level)
        return state, {'cost': self.config.get('cost_narrow', 2000), 'action': 'narrow_public_space'}
    
    def _apply_generate_pocket_node(self, state: WorldState, action: Action) -> Tuple[WorldState, Dict[str, Any]]:
        """Node Generation"""
        public_space_uid = action.target_id
        adjacent_unit_uids = action.params['adjacent_unit_ids']
        state.space_units.generate_pocket_node(public_space_uid, adjacent_unit_uids)
        state.graph = None
        return state, {'cost': self.config.get('cost_generate_pocket', 3000), 'action': 'generate_pocket_node'}
    
    def _apply_dissolve_pocket_node(self, state: WorldState, action: Action) -> Tuple[WorldState, Dict[str, Any]]:
        """Node Dissolution"""
        pocket_node_uid = action.target_id
        state.space_units.dissolve_pocket_node(pocket_node_uid)
        state.graph = None
        return state, {'cost': self.config.get('cost_dissolve_pocket', 1000), 'action': 'dissolve_pocket_node'}
    
    def _apply_regularize_boundary(self, state: WorldState, action: Action) -> Tuple[WorldState, Dict[str, Any]]:
        """Boundary Regularization"""
        public_space_uid = action.target_id
        rule_level = action.params['rule_level']
        state.space_units.regularize_boundary(public_space_uid, rule_level)
        return state, {'cost': self.config.get('cost_regularize', 1500), 'action': 'regularize_boundary'}

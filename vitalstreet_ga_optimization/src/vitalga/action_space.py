"""动作空间：ActionType(0,1)、编码 [action_type, unit_index, type_id]、解码与合法动作生成"""
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass

from .state import WorldState
from .business_type import BusinessCategory


class ActionType(Enum):
    """仅保留两类动作"""
    CHANGE_BUSINESS = 0          # shop_i + type_j
    SHOP_TO_PUBLIC_SPACE = 1    # Shop → Public Space
    NO_OP = 9                   # fallback（无效动作时）


@dataclass
class Action:
    """统一动作表示。编码：encoded = [action_type, unit_index, type_id]，type_id 仅 CHANGE_BUSINESS 有效否则 0"""
    type: ActionType
    target_id: Any
    params: Dict[str, Any]  # CHANGE_BUSINESS: params={'new_type': type_str}

    def encode(self, state: WorldState) -> np.ndarray:
        all_units = state.space_units.get_all_space_units()
        unit_index = all_units.index.get_loc(self.target_id) if self.target_id in all_units.index else 0
        if self.type == ActionType.CHANGE_BUSINESS:
            type_id = self.params.get('new_type', 0)
            if isinstance(type_id, str):
                category_list = [c.value for c in BusinessCategory]
                type_id = category_list.index(type_id) if type_id in category_list else 0
            return np.array([self.type.value, unit_index, int(type_id)])
        return np.array([self.type.value, unit_index, 0])


class ActionSpace:
    """动作空间：decode、action_dim、合法动作生成；仅 shop 可执行 CHANGE_BUSINESS / SHOP_TO_PUBLIC_SPACE；protected/enabled 必须生效"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_shops = config.get('max_shops', 1000)
        self.max_business_types = config.get('max_business_types', 50)
        self.max_total_units = config.get('max_total_units', 1000)
        self.max_action_types = 2  # 0, 1

    def get_action_dim(self) -> Tuple[int, ...]:
        """(max_action_types, max_total_units, max_business_types)"""
        return (
            self.max_action_types,
            self.max_total_units,
            self.max_business_types,
        )

    def decode(self, encoded: np.ndarray, state: WorldState) -> Action:
        """encoded = [action_type, unit_index, type_id]。无效则返回 NO_OP。"""
        action_type_val = int(encoded[0])
        unit_index = int(encoded[1])
        param = int(encoded[2]) if len(encoded) > 2 else 0
        if action_type_val < 0 or action_type_val >= self.max_action_types:
            return Action(type=ActionType.NO_OP, target_id=None, params={})
        all_units = state.space_units.get_all_space_units()
        if unit_index >= len(all_units) or unit_index < 0:
            return Action(type=ActionType.NO_OP, target_id=None, params={})
        target_unit = all_units.iloc[unit_index]
        target_id = target_unit.name
        unit_type = target_unit.get('unit_type', '')
        if action_type_val == 0:  # CHANGE_BUSINESS
            if unit_type != 'shop':
                return Action(type=ActionType.NO_OP, target_id=None, params={})
            category_list = [c.value for c in BusinessCategory]
            new_type = category_list[param] if param < len(category_list) else 'undefined'
            return Action(type=ActionType.CHANGE_BUSINESS, target_id=target_id, params={'new_type': new_type})
        elif action_type_val == 1:  # SHOP_TO_PUBLIC_SPACE
            if unit_type != 'shop':
                return Action(type=ActionType.NO_OP, target_id=None, params={})
            return Action(type=ActionType.SHOP_TO_PUBLIC_SPACE, target_id=target_id, params={})
        return Action(type=ActionType.NO_OP, target_id=None, params={})

    def get_legal_actions(self, state: WorldState) -> List[np.ndarray]:
        """返回所有合法动作的编码列表；仅 shop、非 protected、enabled 可操作。"""
        legal = []
        all_units = state.space_units.get_all_space_units()
        replaceable = state.space_units.get_replaceable_shops()
        shops_for_public = state.space_units.get_shops_for_circulation()
        category_list = [c.value for c in BusinessCategory]
        n_types = min(len(category_list), self.max_business_types)
        for i in range(len(all_units)):
            if i >= self.max_total_units:
                break
            row = all_units.iloc[i]
            uid = row.name
            if row.get('protected', True):
                continue
            if not row.get('enabled', True):
                continue
            if row.get('unit_type', '') != 'shop':
                continue
            # CHANGE_BUSINESS：仅 replaceable 的 shop
            if uid in replaceable.index:
                for type_id in range(n_types):
                    legal.append(np.array([ActionType.CHANGE_BUSINESS.value, i, type_id]))
            # SHOP_TO_PUBLIC_SPACE：enabled 的 shop 即可
            if uid in shops_for_public.index:
                legal.append(np.array([ActionType.SHOP_TO_PUBLIC_SPACE.value, i, 0]))
        return legal

    def is_valid(self, state: WorldState, action: Action) -> bool:
        if action.type == ActionType.NO_OP:
            return False
        all_units = state.space_units.get_all_space_units()
        if action.target_id not in all_units.index:
            return False
        unit = all_units.loc[action.target_id]
        if unit.get('protected', True) or not unit.get('enabled', True):
            return False
        if unit.get('unit_type', '') != 'shop':
            return False
        if action.type == ActionType.CHANGE_BUSINESS:
            return action.target_id in state.space_units.get_replaceable_shops().index
        if action.type == ActionType.SHOP_TO_PUBLIC_SPACE:
            return action.target_id in state.space_units.get_shops_for_circulation().index
        return False

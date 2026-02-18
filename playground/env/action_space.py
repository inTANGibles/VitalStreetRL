"""动作空间：定义、编码、解码、mask"""
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass


class ActionType(Enum):
    """动作类型枚举 - 只保留两个动作"""
    # 策略1: 业态置换
    CHANGE_BUSINESS = 0  # Multi-Discrete: shop_i + type_j
    
    # 策略2: 街道打通与优化
    SHOP_TO_PUBLIC_SPACE = 1  # Shop → Public Space
    
    # ========== 以下动作类型已注释 ==========
    # ATRIUM_TO_PUBLIC_SPACE = 2  # Atrium → Public Space
    # CLOSE_PUBLIC_SPACE_NODE = 3  # Public Space Node Closure
    # 
    # # 策略3: 公共空间改造
    # WIDEN_PUBLIC_SPACE = 4  # Widen (Building Unit Transfer)
    # NARROW_PUBLIC_SPACE = 5  # Narrow (Occupancy Zone Insertion)
    # GENERATE_POCKET_NODE = 6  # Node Generation
    # DISSOLVE_POCKET_NODE = 7  # Node Dissolution
    # REGULARIZE_BOUNDARY = 8  # Boundary Regularization
    
    # NO_OP 保留作为 fallback（仅用于错误处理，不在正常动作空间中使用）
    NO_OP = 9


@dataclass
class Action:
    """统一动作表示"""
    type: ActionType
    target_id: Any  # 目标单元ID (UUID)
    params: Dict[str, Any]  # 动作参数
    # 对于CHANGE_BUSINESS: params={'new_type': type_j}
    
    def encode(self, state) -> np.ndarray:
        """编码为multi-discrete向量（需要state来获取单元索引）"""
        all_units = state.space_units.get_all_space_units()
        unit_index = all_units.index.get_loc(self.target_id) if self.target_id in all_units.index else 0
        
        if self.type == ActionType.CHANGE_BUSINESS:
            type_id = self.params.get('new_type', 0)
            return np.array([self.type.value, unit_index, type_id])
        else:
            return np.array([self.type.value, unit_index, 0])


class ActionSpace:
    """动作空间管理器 - 只支持动作类型 0 和 1"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化动作空间配置"""
        self.config = config
        self.max_shops = config.get('max_shops', 1000)
        self.max_business_types = config.get('max_business_types', 50)
        self.max_public_spaces = config.get('max_public_spaces', 100)
        self.max_public_space_nodes = config.get('max_public_space_nodes', 500)
        self.max_total_units = config.get('max_total_units', 1000)  # 所有单元的最大数量
        
        # 只支持2个动作类型：0 (CHANGE_BUSINESS) 和 1 (SHOP_TO_PUBLIC_SPACE)
        self.max_action_types = 2
    
    def get_action_mask(self, state) -> np.ndarray:
        """
        生成动作mask
        
        Returns:
            mask: shape=(action_dim,), True表示可执行
        """
        # TODO: 实现动作mask
        # 策略1: 业态置换 - 检查可替换的shop
        # 策略2: 街道打通 - 检查可转换的shop
        pass
    
    def sample(self, state, mask: Optional[np.ndarray] = None) -> Action:
        """在合法动作空间内采样"""
        # TODO: 实现合法动作采样
        pass
    
    def is_valid(self, state, action: Action) -> bool:
        """检查动作是否合法"""
        # TODO: 实现动作合法性检查
        pass
    
    def get_action_dim(self) -> Tuple[int, ...]:
        """
        返回multi-discrete动作维度
        - action[0]: ActionType (0-1，只有2个动作类型)
        - action[1]: 所有对象的索引 (0 to max_total_units-1)
        - action[2]: 参数（type_id等，仅用于CHANGE_BUSINESS）
        """
        return (
            self.max_action_types,  # action_type维度：只有2个动作类型 (0, 1)
            self.max_total_units,  # 所有对象的索引维度
            self.max_business_types,  # type_id维度（用于CHANGE_BUSINESS）
        )
    
    def decode(self, encoded: np.ndarray, state) -> Action:
        """
        从编码向量解码为Action对象
        只支持动作类型 0 (CHANGE_BUSINESS) 和 1 (SHOP_TO_PUBLIC_SPACE)
        
        Args:
            encoded: numpy数组 [action_type, unit_index, param]
            state: WorldState，用于获取单元列表
            
        Returns:
            Action对象（如果动作无效，返回NO_OP类型的Action作为fallback）
        """
        action_type_val = int(encoded[0])
        unit_index = int(encoded[1])
        param = int(encoded[2]) if len(encoded) > 2 else 0
        
        # 检查动作类型范围（只允许 0 和 1）
        if action_type_val < 0 or action_type_val >= self.max_action_types:
            # 动作类型超出范围，返回NO_OP作为fallback（仅用于错误处理）
            return Action(type=ActionType.NO_OP, target_id=None, params={})
        
        # 获取所有单元
        all_units = state.space_units.get_all_space_units()
        
        # 检查索引范围
        if unit_index >= len(all_units) or unit_index < 0:
            # 索引超出范围，返回NO_OP作为fallback
            return Action(type=ActionType.NO_OP, target_id=None, params={})
        
        # 获取目标单元
        target_unit = all_units.iloc[unit_index]
        target_id = target_unit.name  # UID (DataFrame的index)
        unit_type = target_unit.get('unit_type', '')
        
        # 根据动作类型解析（只处理动作类型 0 和 1）
        if action_type_val == 0:  # CHANGE_BUSINESS
            if unit_type != 'shop':
                # 单元类型不匹配，返回NO_OP作为fallback
                return Action(type=ActionType.NO_OP, target_id=None, params={})
            # 将type_id转换为业态类型字符串（使用BusinessCategory枚举值作为简化映射）
            from .geo.business_type import BusinessCategory
            category_list = list(BusinessCategory)
            if param < len(category_list):
                # 使用category的value作为new_type（简化处理）
                new_type = category_list[param].value
            else:
                new_type = 'undefined'
            return Action(type=ActionType.CHANGE_BUSINESS, target_id=target_id, params={'new_type': new_type})
        
        elif action_type_val == 1:  # SHOP_TO_PUBLIC_SPACE
            if unit_type != 'shop':
                # 单元类型不匹配，返回NO_OP作为fallback
                return Action(type=ActionType.NO_OP, target_id=None, params={})
            return Action(type=ActionType.SHOP_TO_PUBLIC_SPACE, target_id=target_id, params={})
        
        else:
            # 理论上不应该到达这里（因为已经检查了范围），返回NO_OP作为fallback
            return Action(type=ActionType.NO_OP, target_id=None, params={})
        
        # ========== 以下代码已注释（原动作类型 2-9）==========
        # elif action_type == ActionType.ATRIUM_TO_PUBLIC_SPACE:
        #     if unit_type != 'atrium':
        #         return Action(type=ActionType.NO_OP, target_id=None, params={})
        #     return Action(type=action_type, target_id=target_id, params={})
        # 
        # elif action_type == ActionType.CLOSE_PUBLIC_SPACE_NODE:
        #     if unit_type != 'public_space':
        #         return Action(type=ActionType.NO_OP, target_id=None, params={})
        #     return Action(type=action_type, target_id=target_id, params={})
        # 
        # else:
        #     # 其他动作类型暂时返回NO_OP
        #     return Action(type=ActionType.NO_OP, target_id=None, params={})

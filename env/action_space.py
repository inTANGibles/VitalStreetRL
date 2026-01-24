"""动作空间：定义、编码、解码、mask"""
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass


class ActionType(Enum):
    """动作类型枚举 - 对应三种更新策略"""
    # 策略1: 业态置换
    CHANGE_BUSINESS = 0  # Multi-Discrete: shop_i + type_j
    
    # 策略2: 街道打通与优化
    SHOP_TO_PUBLIC_SPACE = 1  # Shop → Public Space
    ATRIUM_TO_PUBLIC_SPACE = 2  # Atrium → Public Space
    CLOSE_PUBLIC_SPACE_NODE = 3  # Public Space Node Closure
    
    # 策略3: 公共空间改造
    WIDEN_PUBLIC_SPACE = 4  # Widen (Building Unit Transfer)
    NARROW_PUBLIC_SPACE = 5  # Narrow (Occupancy Zone Insertion)
    GENERATE_POCKET_NODE = 6  # Node Generation
    DISSOLVE_POCKET_NODE = 7  # Node Dissolution
    REGULARIZE_BOUNDARY = 8  # Boundary Regularization
    
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
    """动作空间管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化动作空间配置"""
        self.config = config
        self.max_shops = config.get('max_shops', 1000)
        self.max_business_types = config.get('max_business_types', 50)
        self.max_public_spaces = config.get('max_public_spaces', 100)
        self.max_public_space_nodes = config.get('max_public_space_nodes', 500)
        self.max_total_units = config.get('max_total_units', 1000)  # 所有单元的最大数量
    
    def get_action_mask(self, state) -> np.ndarray:
        """
        生成动作mask
        
        Returns:
            mask: shape=(action_dim,), True表示可执行
        """
        # 策略1: 业态置换 - 检查可替换的shop
        # 策略2: 街道打通 - 检查可转换的shop/atrium和可关闭的public_space节点
        # 策略3: 公共空间 - 检查可操作的public space和相邻单元
        pass
    
    def sample(self, state, mask: Optional[np.ndarray] = None) -> Action:
        """在合法动作空间内采样"""
        pass
    
    def is_valid(self, state, action: Action) -> bool:
        """检查动作是否合法"""
        pass
    
    def get_action_dim(self) -> Tuple[int, ...]:
        """
        返回multi-discrete动作维度
        - action[0]: ActionType (0-9)
        - action[1]: 所有对象的索引 (0 to max_total_units-1)
        - action[2]: 参数（type_id等）
        """
        return (
            len(ActionType),  # action_type维度
            self.max_total_units,  # 所有对象的索引维度
            self.max_business_types,  # type_id维度
        )
    
    def decode(self, encoded: np.ndarray, state) -> Action:
        """
        从编码向量解码为Action对象
        
        Args:
            encoded: numpy数组 [action_type, unit_index, param]
            state: WorldState，用于获取单元列表
            
        Returns:
            Action对象
        """
        action_type_val = int(encoded[0])
        unit_index = int(encoded[1])
        param = int(encoded[2]) if len(encoded) > 2 else 0
        
        # 获取所有单元
        all_units = state.space_units.get_all_space_units()
        
        # 检查索引范围
        if unit_index >= len(all_units) or unit_index < 0:
            return Action(type=ActionType.NO_OP, target_id=None, params={})
        
        # 获取目标单元
        target_unit = all_units.iloc[unit_index]
        target_id = target_unit.name  # UID (DataFrame的index)
        unit_type = target_unit.get('unit_type', '')
        
        # 根据动作类型解析
        try:
            action_type = ActionType(action_type_val)
        except ValueError:
            return Action(type=ActionType.NO_OP, target_id=None, params={})
        
        # 只实现前四个动作类型
        if action_type == ActionType.CHANGE_BUSINESS:
            if unit_type != 'shop':
                return Action(type=ActionType.NO_OP, target_id=None, params={})
            # 将type_id转换为业态类型字符串（使用BusinessCategory枚举值作为简化映射）
            from .geo.business_type import BusinessCategory
            category_list = list(BusinessCategory)
            if param < len(category_list):
                # 使用category的value作为new_type（简化处理）
                new_type = category_list[param].value
            else:
                new_type = 'undefined'
            return Action(type=action_type, target_id=target_id, params={'new_type': new_type})
        
        elif action_type == ActionType.SHOP_TO_PUBLIC_SPACE:
            if unit_type != 'shop':
                return Action(type=ActionType.NO_OP, target_id=None, params={})
            return Action(type=action_type, target_id=target_id, params={})
        
        elif action_type == ActionType.ATRIUM_TO_PUBLIC_SPACE:
            if unit_type != 'atrium':
                return Action(type=ActionType.NO_OP, target_id=None, params={})
            return Action(type=action_type, target_id=target_id, params={})
        
        elif action_type == ActionType.CLOSE_PUBLIC_SPACE_NODE:
            if unit_type != 'public_space':
                return Action(type=ActionType.NO_OP, target_id=None, params={})
            return Action(type=action_type, target_id=target_id, params={})
        
        else:
            # 其他动作类型暂时返回NO_OP
            return Action(type=ActionType.NO_OP, target_id=None, params={})

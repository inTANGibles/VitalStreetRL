"""终止条件：episode结束判断"""
from typing import Dict, Any, Tuple, List
from env.world_state import WorldState


class TerminationReason:
    """终止原因枚举"""
    BUDGET_EXCEEDED = "budget_exceeded"
    MAX_STEPS = "max_steps"
    NO_VALID_ACTIONS = "no_valid_actions"
    STAGNATION = "stagnation"
    SEVERE_VIOLATION = "severe_violation"
    MANUAL = "manual"


class TerminationChecker:
    """终止条件检查器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化终止配置"""
        self.max_steps = config.get('max_steps', 100)
        self.stagnation_threshold = config.get('stagnation_threshold', 10)
        self.violation_threshold = config.get('violation_threshold', 100.0)
    
    def check(self, state: WorldState, history: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        检查是否终止
        
        Returns:
            done: 是否终止
            reason_code: 终止原因
        """
        # 检查各种终止条件
        if state.budget < 0:
            return True, TerminationReason.BUDGET_EXCEEDED
        
        if state.step_idx >= self.max_steps:
            return True, TerminationReason.MAX_STEPS
        
        # ... 其他检查
        
        return False, ""

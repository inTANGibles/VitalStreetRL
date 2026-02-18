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
    
    def check(self, state: WorldState, history: List[float]) -> Tuple[bool, str]:
        """
        检查是否终止
        
        Args:
            state: 当前状态
            history: 最近N步的reward列表（用于停滞检查）
        
        Returns:
            done: 是否终止
            reason_code: 终止原因
        """
        # 检查各种终止条件
        if state.budget < 0:
            return True, TerminationReason.BUDGET_EXCEEDED
        
        if state.step_idx >= self.max_steps:
            return True, TerminationReason.MAX_STEPS
        
        # 检查停滞（reward变化很小）
        if len(history) >= self.stagnation_threshold:
            recent_rewards = history[-self.stagnation_threshold:]
            if len(set(recent_rewards)) <= 1:  # 所有reward都相同
                return True, TerminationReason.STAGNATION
        
        return False, ""

"""奖励计算：活力变化、成本、违规惩罚"""
from typing import Dict, Any, Tuple
import numpy as np
from env.world_state import WorldState
from env.action_space import Action


class RewardCalculator:
    """奖励计算器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 包含权重lambda, mu等
        """
        self.lambda_cost = config.get('lambda_cost', 0.1)
        self.mu_violation = config.get('mu_violation', 1.0)
    
    def compute(self, state: WorldState, action: Action, next_state: WorldState,
                V_prev: np.ndarray, V_next: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        计算奖励与分项
        
        Returns:
            reward: 标量奖励
            reward_terms: 分项字典（用于日志）
        """
        # R = ΔV - λ·Cost(a) - μ·Violation(s')
        delta_v = self._compute_vitality_change(V_prev, V_next)
        cost = self._compute_cost(action)
        violation = self._compute_violation(next_state)
        
        reward = delta_v - self.lambda_cost * cost - self.mu_violation * violation
        
        reward_terms = {
            'delta_vitality': delta_v,
            'cost': cost,
            'violation': violation,
            'total': reward
        }
        
        return reward, reward_terms
    
    def _compute_vitality_change(self, V_prev: np.ndarray, V_next: np.ndarray) -> float:
        """计算活力变化"""
        pass
    
    def _compute_cost(self, action: Action) -> float:
        """计算动作成本"""
        pass
    
    def _compute_violation(self, state: WorldState) -> float:
        """计算约束违背惩罚"""
        pass

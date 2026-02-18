"""奖励计算：活力变化、成本、违规惩罚"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
from env.world_state import WorldState
from env.action_space import Action
from objective.vitality_metrics import VitalityMetrics


class RewardCalculator:
    """奖励计算器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 包含权重lambda, mu等
        """
        self.mu_violation = config.get('mu_violation', 1.0)
        # 保存初始状态的活力值，用于计算变化
        self.initial_vitality: Optional[float] = None
        # 活力指标计算器
        vitality_config = config.get('vitality_metrics', {})
        self.vitality_metrics = VitalityMetrics(vitality_config)
        # 保存初始状态的三个活力指标值
        self.initial_mixedness: Optional[float] = None
        self.initial_vacancy_rate: Optional[float] = None
        self.initial_concentration: Optional[float] = None
    
    def reset(self, initial_state: WorldState):
        """
        重置计算器，保存初始状态的活力值和活力指标
        
        Args:
            initial_state: 初始世界状态
        """
        self.initial_vitality = self._compute_vitality(initial_state)
        
        # 计算并保存初始状态的三个活力指标
        # 使用空的F_hat数组（因为VitalityMetrics.compute需要F_hat参数，但实际计算不依赖它）
        initial_v_vec = self.vitality_metrics.compute(
            F_hat=np.array([]), 
            state=initial_state
        )
        self.initial_mixedness = float(initial_v_vec[0])
        self.initial_vacancy_rate = float(initial_v_vec[1])
        self.initial_concentration = float(initial_v_vec[2])
    
    def compute(self, state: WorldState, action: Action, next_state: WorldState,
                V_prev: np.ndarray = None, V_next: np.ndarray = None) -> Tuple[float, Dict[str, float]]:
        """
        计算奖励与分项
        
        Args:
            state: 当前状态
            action: 执行的动作
            next_state: 下一个状态
            V_prev: 之前的活力值（可选，如果不提供则从state计算）
            V_next: 之后的活力值（可选，如果不提供则从next_state计算）
        
        Returns:
            reward: 标量奖励
            reward_terms: 分项字典（用于日志）
        """
        step_idx = next_state.step_idx if hasattr(next_state, 'step_idx') else 0
        # 计算活力变化（与初始值对比）
        if V_prev is None:
            V_prev = self._compute_vitality(state)
        
        if V_next is None:
            V_next = self._compute_vitality(next_state)
        
        delta_v = self._compute_vitality_change(V_prev, V_next)
        violation = self._compute_violation(next_state)
        
        reward = delta_v - self.mu_violation * violation
        
        # 计算当前状态的活力指标（用于日志）
        current_v_vec = self.vitality_metrics.compute(
            F_hat=np.array([]),
            state=next_state
        )
        
        reward_terms = {
            'delta_vitality': delta_v,
            'vitality_current': V_next,
            'vitality_initial': self.initial_vitality if self.initial_vitality is not None else 0.0,
            'violation': violation,
            'mixedness_current': float(current_v_vec[0]),
            'mixedness_initial': self.initial_mixedness if self.initial_mixedness is not None else 0.0,
            'vacancy_rate_current': float(current_v_vec[1]),
            'vacancy_rate_initial': self.initial_vacancy_rate if self.initial_vacancy_rate is not None else 0.0,
            'concentration_current': float(current_v_vec[2]),
            'concentration_initial': self.initial_concentration if self.initial_concentration is not None else 0.0,
            'total': reward
        }
        
        return reward, reward_terms
    
    def _compute_vitality(self, state: WorldState) -> float:
        """
        计算当前状态的活力值
        活力 = 公共空间面积加权的人流量累加
        
        Args:
            state: 世界状态
            
        Returns:
            vitality: 活力值（面积加权的人流量总和）
        """
        step_idx = state.step_idx if hasattr(state, 'step_idx') else 0
        public_spaces = state.space_units.get_public_spaces()
        
        if len(public_spaces) == 0:
            return 0.0
        
        # 计算面积加权的人流量累加
        # vitality = sum(area_i * flow_prediction_i) for all public_space units
        areas = public_spaces['area'].values
        flows = public_spaces['flow_prediction'].values
        
        # 确保flow_prediction不为NaN
        flows = np.nan_to_num(flows, nan=0.0)
        
        vitality = np.sum(areas * flows)
        
        return float(vitality)
    
    def _compute_vitality_change(self, V_prev: float, V_next: float) -> float:
        """
        计算活力变化（与初始值对比，然后归一化）
        
        Args:
            V_prev: 之前的活力值
            V_next: 之后的活力值
            
        Returns:
            normalized_delta: 归一化后的活力变化值
        """
        if self.initial_vitality is None:
            # 如果没有初始值，使用V_prev作为初始值
            self.initial_vitality = V_prev
        
        # 计算相对于初始值的变化
        delta_v = V_next - self.initial_vitality
        
        # 归一化：使用初始值作为分母（避免除零）
        if abs(self.initial_vitality) > 1e-6:
            normalized_delta = delta_v / abs(self.initial_vitality)
        else:
            # 如果初始值为0，直接使用变化值（可能需要进一步缩放）
            normalized_delta = delta_v / 1000.0  # 假设一个合理的缩放因子
        
        return normalized_delta
    
    def _compute_violation(self, state: WorldState) -> float:
        """
        计算约束违背惩罚
        基于三个活力指标与初始值的差值：
        - 商业混合度应该提高（差值应该为正），如果降低则惩罚
        - 空置率应该降低（差值应该为负），如果提高则惩罚
        - 集中度应该降低（差值应该为负），如果提高则惩罚
        
        Args:
            state: 世界状态
            
        Returns:
            violation: 违规惩罚值（非负）
        """
        if (self.initial_mixedness is None or 
            self.initial_vacancy_rate is None or 
            self.initial_concentration is None):
            # 如果初始值未设置，返回0（不惩罚）
            return 0.0
        
        # 计算当前状态的三个活力指标
        current_v_vec = self.vitality_metrics.compute(
            F_hat=np.array([]),
            state=state
        )
        current_mixedness = float(current_v_vec[0])
        current_vacancy_rate = float(current_v_vec[1])
        current_concentration = float(current_v_vec[2])
        
        # 计算与初始值的差值
        delta_mixedness = current_mixedness - self.initial_mixedness
        delta_vacancy_rate = current_vacancy_rate - self.initial_vacancy_rate
        delta_concentration = current_concentration - self.initial_concentration
        
        # 计算惩罚：
        # 1. 商业混合度降低 → 惩罚（差值<0）
        mixedness_penalty = max(0.0, -delta_mixedness)
        
        # 2. 空置率提高 → 惩罚（差值>0）
        vacancy_penalty = max(0.0, delta_vacancy_rate)
        
        # 3. 集中度提高 → 惩罚（差值>0）
        concentration_penalty = max(0.0, delta_concentration)
        
        # 总惩罚 = 三个惩罚的加权和
        violation = mixedness_penalty + vacancy_penalty + concentration_penalty
        
        return violation

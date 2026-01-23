"""活力/脆弱性指标：输出向量"""
from typing import Dict, Any, Tuple
import numpy as np
from collections import Counter
from env.world_state import WorldState


class VitalityMetrics:
    """商业活力指标计算器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化指标配置"""
        self.config = config
    
    def compute(self, F_hat: np.ndarray, state: WorldState) -> np.ndarray:
        """
        计算活力指标向量（经济维度）
        
        Args:
            F_hat: 预测客流 (N, T) 或 (N,)
            state: 当前状态
        
        Returns:
            V_vec: 活力指标向量 [商业混合度, 空置率, 集中度, ...]
        """
        # 经济维度：商业混合度、空置率、集中度
        mixedness = self._compute_business_mixedness(state)
        vacancy_rate = self._compute_vacancy_rate(state)
        concentration = self._compute_concentration(state)
        
        # 其他维度（社会、物理环境）可后续添加
        V_vec = np.array([mixedness, vacancy_rate, concentration])
        return V_vec
    
    def _compute_business_mixedness(self, state: WorldState) -> float:
        """
        计算商业混合度
        
        混合度衡量业态类型的多样性，使用Shannon熵或Simpson指数
        Returns: 0-1之间的值，1表示完全混合
        """
        shops = state.space_units.get_space_units_by_business_type('shop')
        if len(shops) == 0:
            return 0.0
        
        # 统计各业态类型的数量
        business_types = shops['business_type'].values
        type_counts = Counter(business_types)
        
        # 排除'UNDEFINED'和'N/A'
        valid_types = {k: v for k, v in type_counts.items() 
                      if k not in ['UNDEFINED', 'N/A']}
        
        if len(valid_types) == 0:
            return 0.0
        
        # 使用Shannon熵计算混合度
        total = sum(valid_types.values())
        proportions = [count / total for count in valid_types.values()]
        
        # Shannon熵: H = -Σ(p_i * log(p_i))
        shannon_entropy = -sum(p * np.log(p) for p in proportions if p > 0)
        
        # 归一化到0-1：H_norm = H / log(N_types)
        max_entropy = np.log(len(valid_types))
        mixedness = shannon_entropy / max_entropy if max_entropy > 0 else 0.0
        
        return mixedness
    
    def _compute_vacancy_rate(self, state: WorldState) -> float:
        """
        计算空置率
        
        空置率 = 空置shop数量 / 总shop数量
        Returns: 0-1之间的值，0表示无空置
        """
        all_shops = state.space_units.get_space_units_by_business_type('shop')
        if len(all_shops) == 0:
            return 0.0
        
        # 空置shop：business_type为'UNDEFINED'或'enabled'为False
        vacant_shops = all_shops[
            (all_shops['business_type'].isin(['UNDEFINED', 'N/A'])) |
            (all_shops['enabled'] == False)
        ]
        
        vacancy_rate = len(vacant_shops) / len(all_shops)
        return vacancy_rate
    
    def _compute_concentration(self, state: WorldState) -> float:
        """
        计算集中度
        
        集中度衡量业态类型的集中程度，使用Herfindahl-Hirschman Index (HHI)
        HHI = Σ(p_i^2)，其中p_i是第i种业态的比例
        Returns: 0-1之间的值，1表示完全集中（单一业态），0表示完全分散
        """
        shops = state.space_units.get_space_units_by_business_type('shop')
        if len(shops) == 0:
            return 0.0
        
        business_types = shops['business_type'].values
        type_counts = Counter(business_types)
        
        # 排除'UNDEFINED'和'N/A'
        valid_types = {k: v for k, v in type_counts.items() 
                      if k not in ['UNDEFINED', 'N/A']}
        
        if len(valid_types) == 0:
            return 1.0  # 无有效业态，视为完全集中
        
        total = sum(valid_types.values())
        proportions = [count / total for count in valid_types.values()]
        
        # HHI = Σ(p_i^2)
        hhi = sum(p ** 2 for p in proportions)
        
        # HHI范围是[1/N, 1]，归一化到[0, 1]
        # concentration = (HHI - 1/N) / (1 - 1/N)
        n_types = len(valid_types)
        if n_types == 1:
            return 1.0
        
        min_hhi = 1.0 / n_types
        concentration = (hhi - min_hhi) / (1.0 - min_hhi)
        
        return concentration
    
    def aggregate(self, V_vec: np.ndarray) -> float:
        """
        聚合为标量（用于reward）
        
        经济维度：混合度越高越好，空置率越低越好，集中度越低越好
        """
        mixedness, vacancy_rate, concentration = V_vec[0], V_vec[1], V_vec[2]
        
        # 经济维度得分 = 混合度 - 空置率 - 集中度（归一化后）
        economic_score = mixedness - vacancy_rate - concentration
        
        # 标准化到0-100范围（根据需求调整）
        normalized_score = (economic_score + 2) / 4 * 100  # 假设范围是[-2, 2]
        return max(0, min(100, normalized_score))

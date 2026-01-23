"""特征提取：从state抽取STGNN输入特征"""
from typing import Dict, Any
import numpy as np
from env.world_state import WorldState


class FeatureExtractor:
    """STGNN特征提取器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化特征提取配置"""
        self.config = config
    
    def extract(self, state: WorldState) -> np.ndarray:
        """
        从状态提取STGNN输入特征
        
        Returns:
            X: 特征矩阵 (N, F)，N为节点数，F为特征维度
        """
        # 提取：空间配置、业态、界面、容量、租金代理等
        pass
    
    def _extract_spatial_features(self, state: WorldState) -> np.ndarray:
        """提取空间配置特征"""
        pass
    
    def _extract_business_features(self, state: WorldState) -> np.ndarray:
        """提取业态特征"""
        pass
    
    def _extract_interface_features(self, state: WorldState) -> np.ndarray:
        """提取界面特征"""
        pass

"""STGNN包装器：统一推理接口、缓存、批处理"""
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch


class STGNNWrapper:
    """STGNN模型包装器（作为surrogate simulator）"""
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            model_path: 预训练模型路径
            config: 模型配置（时间窗口、特征维度等）
        """
        self.config = config or {}
        self.model = None  # 加载的STGNN模型
        self.cache = {}  # 缓存预测结果
    
    def predict(self, X: np.ndarray, t_ctx: Optional[np.ndarray] = None) -> np.ndarray:
        """
        预测客流
        
        Args:
            X: 空间配置特征 (N, F) 或 (B, N, F)
            t_ctx: 时间上下文 (可选)
        
        Returns:
            F_hat: 预测客流 (N, T) 或 (B, N, T)
        """
        # 1. 检查缓存
        # 2. 模型推理
        # 3. 更新缓存
        pass
    
    def predict_batch(self, X_batch: np.ndarray, t_ctx_batch: Optional[np.ndarray] = None) -> np.ndarray:
        """批处理预测"""
        pass
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
    
    def load_model(self, path: str):
        """加载预训练模型"""
        pass

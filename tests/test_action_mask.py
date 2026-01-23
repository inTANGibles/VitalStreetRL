"""测试动作mask"""
import unittest
import numpy as np
from env.action_space import ActionSpace
from env.world_state import WorldState


class TestActionMask(unittest.TestCase):
    """动作mask测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = {}
        self.action_space = ActionSpace(self.config)
    
    def test_protected_mask(self):
        """测试保护区域mask"""
        # 创建包含保护区域的状态
        # 验证保护区域动作被mask
        pass
    
    def test_budget_mask(self):
        """测试预算约束mask"""
        pass
    
    def test_min_width_mask(self):
        """测试最小街宽约束mask"""
        pass

"""世界状态：图/几何/属性的权威状态"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import geopandas as gpd
import copy
from .geo.space_unit import SpaceUnitCollection
from .geo.business_type import BusinessTypeCollection


@dataclass
class WorldState:
    """环境状态表示"""
    # 空间对象集合（核心数据结构）
    space_units: SpaceUnitCollection      # 空间单元集合（核心操作对象）
    business_types: BusinessTypeCollection  # 业态类型集合
    
    # 图结构（从space_units构建）
    graph: Any  # NetworkX Graph 或自定义图结构
    
    # 约束与预算
    budget: float
    constraints: Dict[str, Any]  # 保护约束、最小街宽等
    
    # 元数据
    step_idx: int = 0
    episode_id: Optional[str] = None
    
    @classmethod
    def from_geojson(
        cls,
        geojson_path: str,
        budget: float = 1000000.0,
        constraints: Optional[Dict[str, Any]] = None,
        episode_id: Optional[str] = None
    ) -> 'WorldState':
        """
        从GeoJSON文件加载初始状态
        
        Args:
            geojson_path: GeoJSON文件路径
            budget: 初始预算
            constraints: 约束字典（如果为None，使用默认约束）
            episode_id: 回合ID（可选）
            
        Returns:
            state: WorldState对象
        """
        # 读取GeoJSON文件
        geojson_path = Path(geojson_path)
        if not geojson_path.exists():
            raise FileNotFoundError(f"GeoJSON文件不存在: {geojson_path}")
        
        gdf = gpd.read_file(geojson_path)
        
        # 转换为SpaceUnitCollection
        # 使用scripts中的函数
        import sys

        project_root = Path(__file__).resolve().parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from scripts.geojson_to_raster import geojson_to_spaceunit_collection
        
        space_units = geojson_to_spaceunit_collection(gdf)
        
        # 初始化BusinessTypeCollection（使用默认类型）
        business_types = BusinessTypeCollection()
        
        # 默认约束
        if constraints is None:
            constraints = {
                'protected_zones': [],
                'max_budget': budget
            }
        
        # 创建WorldState
        state = cls(
            space_units=space_units,
            business_types=business_types,
            graph=None,  # 图结构延迟构建
            budget=budget,
            constraints=constraints,
            step_idx=0,
            episode_id=episode_id
        )
        
        return state
    
    def copy(self) -> 'WorldState':
        """
        深拷贝状态
        
        Returns:
            copied_state: 深拷贝后的WorldState对象
        """
        # 深拷贝SpaceUnitCollection（通过重新创建GeoDataFrame）
        copied_space_units = SpaceUnitCollection()
        original_gdf = self.space_units.get_all_space_units()
        if not original_gdf.empty:
            # 创建GeoDataFrame的深拷贝
            copied_gdf = original_gdf.copy(deep=True)
            # 将拷贝的GeoDataFrame设置到新的collection中
            copied_space_units._SpaceUnitCollection__unit_gdf = copied_gdf
        
        # BusinessTypeCollection通常是只读的，可以直接引用或浅拷贝
        # 但为了安全，我们创建一个新的实例（它会重新初始化默认类型）
        copied_business_types = BusinessTypeCollection()
        
        # 深拷贝约束字典
        copied_constraints = copy.deepcopy(self.constraints)
        
        # 创建新的WorldState
        copied_state = WorldState(
            space_units=copied_space_units,
            business_types=copied_business_types,
            graph=None,  # 图结构需要重新构建
            budget=self.budget,
            constraints=copied_constraints,
            step_idx=self.step_idx,
            episode_id=self.episode_id
        )
        
        return copied_state
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'space_units': self.space_units.get_all_space_units().to_dict(),
            'budget': self.budget,
            'constraints': self.constraints,
            'step_idx': self.step_idx,
            'episode_id': self.episode_id
        }
    
    def get_graph(self) -> Any:
        """获取或构建图结构"""
        if self.graph is None:
            # 从space_units构建图
            self.graph = self._build_graph()
        return self.graph
    
    def _build_graph(self) -> Any:
        """构建图结构（用于STGNN和RL观测）"""
        # TODO: 实现图构建逻辑
        # 基于space_units的空间关系构建图
        pass

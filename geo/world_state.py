"""世界状态：图/几何/属性的权威状态"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import geopandas as gpd
import copy
from .tool.space_unit import SpaceUnitCollection
from .tool.business_type import BusinessTypeCollection


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
        space_units = cls._geojson_to_spaceunit_collection(gdf)
        
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
    
    @staticmethod
    def _geojson_to_spaceunit_collection(gdf: gpd.GeoDataFrame) -> SpaceUnitCollection:
        """
        将GeoDataFrame转换为SpaceUnitCollection
        
        Args:
            gdf: GeoDataFrame，包含空间单元数据
            
        Returns:
            collection: SpaceUnitCollection对象
        """
        collection = SpaceUnitCollection()
        
        # 遍历每个要素
        for idx, row in gdf.iterrows():
            try:
                geometry = row['geometry']
                
                # 提取坐标
                if hasattr(geometry, 'exterior'):
                    coords = np.array(geometry.exterior.coords[:-1])  # 移除重复的最后一个点
                elif hasattr(geometry, 'geoms'):  # MultiPolygon
                    # 选择最大的多边形
                    largest = max(geometry.geoms, key=lambda p: p.area)
                    coords = np.array(largest.exterior.coords[:-1])
                else:
                    print(f"跳过要素 {idx}: 不支持的几何类型")
                    continue
                
                # 提取属性
                unit_type = row.get('unit_type', 'shop')
                business_type = row.get('business_type', 'UNDEFINED')
                business_category = row.get('business_category', None)
                protected = row.get('protected', False)
                enabled = row.get('enabled', True)
                replaceable = row.get('replaceable', None)
                
                # 确保business_category是字符串格式
                if business_category is not None and not isinstance(business_category, str):
                    # 如果是枚举或其他类型，转换为字符串
                    if hasattr(business_category, 'value'):
                        business_category = business_category.value
                    else:
                        business_category = str(business_category)
                
                # 创建空间单元
                unit_gdf = SpaceUnitCollection._create_unit_by_coords(
                    coords=coords,
                    unit_type=unit_type,
                    business_type=business_type,
                    business_category=business_category,
                    protected=protected,
                    replaceable=replaceable,
                    enabled=enabled
                )
                
                # 只有public_space类型的单元才设置flow_prediction
                # shop单元不应该有flow_prediction属性（必须设为0）
                if unit_type == 'public_space' and 'flow_prediction' in gdf.columns:
                    # public_space单元：使用GeoJSON中的值，如果没有则设为0
                    flow_val = row.get('flow_prediction', 0.0)
                    unit_gdf.loc[unit_gdf.index[0], 'flow_prediction'] = float(flow_val) if flow_val is not None else 0.0
                else:
                    # 确保shop和其他非public_space单元没有flow_prediction（强制设为0）
                    unit_gdf.loc[unit_gdf.index[0], 'flow_prediction'] = 0.0
                
                # 双重检查：确保shop单元的flow_prediction为0
                if unit_type == 'shop':
                    if unit_gdf.loc[unit_gdf.index[0], 'flow_prediction'] != 0.0:
                        unit_gdf.loc[unit_gdf.index[0], 'flow_prediction'] = 0.0
                
                collection.add_space_unit(unit_gdf)
                
            except Exception as e:
                print(f"处理要素 {idx} 失败: {e}")
                continue
        
        # 最终验证：确保所有shop单元的flow_prediction都为0
        all_units = collection.get_all_space_units()
        shop_mask = all_units['unit_type'] == 'shop'
        if shop_mask.any():
            shop_flow_values = all_units.loc[shop_mask, 'flow_prediction']
            if not (shop_flow_values == 0.0).all():
                # 强制设置为0
                all_units.loc[shop_mask, 'flow_prediction'] = 0.0
                # 更新collection
                collection._SpaceUnitCollection__unit_gdf = all_units
        
        return collection

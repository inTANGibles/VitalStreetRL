"""空间单元集合：管理片段空间（parcel/unit），RL动作的直接操作对象"""
import uuid
from typing import Optional, List, Dict, Any, Tuple
import geopandas as gpd
import numpy as np
# 分别导入，避免一个导入失败影响另一个
try:
    from shapely.geometry import Polygon
except ImportError:
    Polygon = None  # 如果shapely未安装，Polygon将为None

# STRtree 在不同版本的shapely中位置不同，尝试多个位置
STRtree = None
try:
    from shapely.strtree import STRtree
except ImportError:
    try:
        # 某些版本可能在ops中
        from shapely.ops import STRtree
    except ImportError:
        # 如果都失败，STRtree保持为None（可选功能）
        STRtree = None

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None  # 如果sklearn未安装，PCA将为None

from .business_type import BusinessCategory


class SpaceUnitCollection:
    """
    空间单元集合
    
    管理商业小镇的片段空间单元，每个单元对应一个可操作的目标。
    """
    
    __space_unit_attrs = [
        # === 基础标识与几何 ===
        'uid',              # UUID标识
        'geometry',         # Shapely几何（Polygon）
        'coords',           # 坐标数组
        'area',             # 面积
        
        # === 单元类型与状态 ===
        'unit_type',        # 单元类型: 'shop', 'atrium', 'public_space', 'circulation'
        'enabled',          # 是否启用（protected=True时自动为False，不参与action）
        'protected',        # 是否受保护（protected=True时enabled自动为False）
        'replaceable',      # 是否可替换（SUPPORTING_FACILITIES和RESIDENTIAL不可替换）
        
        # === Shop相关属性 ===
        'business_type',    # 当前业态类型
        'business_category', # 业态大类（BusinessCategory枚举值的字符串形式）
        
        # === Public Space相关属性 ===
        'min_clear_width',  # 最小净宽
        'main_direction',   # 主方向
        'adjacent_units',   # 相邻单元列表
        'flow_prediction'   # 预测客流（STGNN输出，用于街巷空间）
    ]
    
    __unit_gdf = gpd.GeoDataFrame(columns=__space_unit_attrs)
    __unit_gdf.set_index('uid')
    
    __uid = uuid.uuid4()  # Collection级别的UID
    
    def uid(self):
        """返回Collection的UID"""
        return self.__uid
    
    # region DXF图层解析
    
    @staticmethod
    def parse_layer_name(layer_name: str) -> Tuple[str, Optional[BusinessCategory], bool]:
        """
        解析DXF图层名称，提取单元类型、业态类别和保护状态
        
        图层命名规则：
        - Shop图层: 'shop_BusinessCategory' 或 'shop_BusinessCategory_protected'
          BusinessCategory包括: dining, retail, cultural, service, leisure, residential, supporting, undefined
        - 其他图层: 'atrium', 'public_space', 'circulation'（直接使用）
        
        Args:
            layer_name: DXF图层名称
            
        Returns:
            tuple: (unit_type, business_category, protected)
                - unit_type: 'shop', 'atrium', 'public_space', 'circulation'
                - business_category: BusinessCategory枚举值（shop类型）或None（非shop类型）
                - protected: 是否受保护（bool）
        
        Examples:
            >>> parse_layer_name('shop_dining')
            ('shop', BusinessCategory.DINING, False)
            >>> parse_layer_name('shop_retail_protected')
            ('shop', BusinessCategory.RETAIL, True)
            >>> parse_layer_name('atrium')
            ('atrium', None, False)
            >>> parse_layer_name('public_space')
            ('public_space', None, False)
        """
        layer_name = layer_name.strip().lower()
        
        # 检查是否是shop图层
        if layer_name.startswith('shop_'):
            # 解析shop图层: shop_BusinessCategory 或 shop_BusinessCategory_protected
            parts = layer_name.split('_')
            
            if len(parts) < 2:
                # 格式错误，默认为shop_undefined
                return ('shop', BusinessCategory.UNDEFINED, False)
            
            # 检查是否有protected后缀
            protected = False
            if parts[-1] == 'protected':
                protected = True
                parts = parts[:-1]  # 移除protected后缀
            
            # 提取BusinessCategory
            if len(parts) >= 2:
                category_str = parts[1]  # shop_后面的部分
                
                # 映射到BusinessCategory枚举
                category_mapping = {
                    'dining': BusinessCategory.DINING,
                    'retail': BusinessCategory.RETAIL,
                    'cultural': BusinessCategory.CULTURAL_EXPERIENCE,
                    'service': BusinessCategory.SERVICE,
                    'leisure': BusinessCategory.LEISURE_ENTERTAINMENT,
                    'residential': BusinessCategory.RESIDENTIAL,
                    'supporting': BusinessCategory.SUPPORTING_FACILITIES,
                    'undefined': BusinessCategory.UNDEFINED,
                }
                
                business_category = category_mapping.get(category_str, BusinessCategory.UNDEFINED)
                return ('shop', business_category, protected)
            else:
                # 格式错误，默认为shop_undefined
                return ('shop', BusinessCategory.UNDEFINED, protected)
        
        # 检查其他单元类型
        elif layer_name in ['atrium', 'public_space', 'circulation']:
            return (layer_name, None, False)
        
        # 未知图层类型，默认为shop_undefined
        else:
            return ('shop', BusinessCategory.UNDEFINED, False)
    
    # endregion
    
    # region 增删操作
    
    @staticmethod
    def _create_unit_by_coords(coords: np.ndarray,
                               unit_type: str = "shop",  # 'shop', 'atrium', 'public_space', 'circulation'
                               business_type: str = "UNDEFINED",
                               business_category: str = None,  # BusinessCategory枚举值（用于判断replaceable）
                               protected: bool = False,
                               replaceable: bool = None,  # None表示自动推断
                               enabled: bool = None) -> gpd.GeoDataFrame:
        """
        创建单个空间单元
        
        Args:
            coords: 坐标数组
            unit_type: 单元类型
            business_type: 业态类型
            business_category: 业态大类（BusinessCategory枚举值），用于自动判断replaceable
            protected: 是否受保护
            replaceable: 是否可替换（None时自动推断）
            enabled: 是否启用（None时自动推断：protected=True则enabled=False）
        """
        if Polygon is None:
            raise ImportError("shapely库未安装，无法创建几何对象。请安装shapely: pip install shapely")
        
        geometry = Polygon(coords)
        uid = uuid.uuid4()
        area = geometry.area
        
        # protected自动定义为enabled=False（不参与action）
        if enabled is None:
            enabled = not protected
        
        # 自动推断replaceable
        if replaceable is None:
            replaceable = SpaceUnitCollection._infer_replaceable(
                unit_type=unit_type,
                business_category=business_category,
                protected=protected
            )
        
        new_row = {
            'uid': [uid],
            'geometry': [geometry],
            'coords': [coords],
            'area': [area],
            'unit_type': [unit_type],
            'enabled': [enabled],
            'protected': [protected],
            'replaceable': [replaceable],
            'business_type': [business_type] if unit_type == 'shop' else ['N/A'],
            'business_category': [business_category] if business_category else [None],
            'min_clear_width': [0.0] if unit_type == 'public_space' else [None],
            'main_direction': [None] if unit_type == 'public_space' else [None],
            'adjacent_units': [[]],
            'flow_prediction': [0.0]
        }
        return gpd.GeoDataFrame(new_row, index=new_row['uid'])
    
    @staticmethod
    def _infer_replaceable(unit_type: str, business_category: str = None, protected: bool = False) -> bool:
        """
        推断单元是否可替换
        
        规则:
        1. protected单元不可替换
        2. SUPPORTING_FACILITIES和RESIDENTIAL不可替换
        3. 其他shop类型默认可替换
        
        Args:
            unit_type: 单元类型
            business_category: 业态大类（BusinessCategory枚举值）
            protected: 是否受保护
        
        Returns:
            bool: 是否可替换
        """
        # protected单元不可替换
        if protected:
            return False
        
        # 非shop单元默认不可替换（除非是atrium/public_space等可转换类型）
        if unit_type != 'shop':
            return False
        
        # SUPPORTING_FACILITIES和RESIDENTIAL不可替换
        if business_category:
            non_replaceable_categories = ['supporting', 'residential', 'SUPPORTING_FACILITIES', 'RESIDENTIAL']
            if business_category.upper() in [c.upper() for c in non_replaceable_categories]:
                return False
        
        # 其他shop类型默认可替换
        return True
    
    def add_space_unit(self, unit, return_uid: bool = True) -> Optional[uuid.UUID]:
        """添加空间单元"""
        if not self.__unit_gdf.empty:
            # 排除全NA的列以避免FutureWarning
            # 根据pandas建议，在concat之前排除空或全NA的列
            def _drop_allna_columns(df):
                """移除全NA的列"""
                return df.loc[:, ~df.isna().all()]
            
            # 清理DataFrame，移除全NA的列
            gdf_clean = _drop_allna_columns(self.__unit_gdf)
            unit_clean = _drop_allna_columns(unit)
            
            # 执行concat（使用sort=False避免排序警告）
            self.__unit_gdf = gpd.pd.concat([gdf_clean, unit_clean], ignore_index=False, sort=False)
        else:
            self.__unit_gdf = unit
        self.__uid = uuid.uuid4()
        if return_uid:
            return unit['uid'].iloc[0] if len(unit) == 1 else unit['uid']
    
    def delete_space_unit(self, unit):
        """删除空间单元（对应DEMOLISH动作）"""
        uid = unit['uid'] if hasattr(unit, 'uid') else unit
        self.__unit_gdf.drop(uid, inplace=True)
        self.__uid = uuid.uuid4()
    
    def delete_space_unit_by_uid(self, uid: uuid.UUID):
        """通过UID删除"""
        unit = self.get_space_unit_by_uid(uid)
        self.delete_space_unit(unit)
    
    # endregion
    
    # region 查询操作
    
    def get_space_unit_by_uid(self, uid: uuid.UUID):
        """通过UID获取空间单元"""
        return self.__unit_gdf.loc[uid]
    
    def get_space_unit_by_index(self, idx: int):
        """通过索引获取"""
        return self.__unit_gdf.iloc[idx]
    
    def get_all_space_units(self) -> gpd.GeoDataFrame:
        """获取所有空间单元"""
        return self.__unit_gdf
    
    def get_space_units_by_business_type(self, business_type: str) -> gpd.GeoDataFrame:
        """按业态类型查询"""
        return self.__unit_gdf[self.__unit_gdf['business_type'] == business_type]
    
    def get_replaceable_shops(self) -> gpd.GeoDataFrame:
        """
        获取可替换的shop（用于业态置换动作mask）
        
        条件:
        - unit_type == 'shop'
        - replaceable == True（自动排除SUPPORTING_FACILITIES和RESIDENTIAL）
        - enabled == True（自动排除protected单元，因为protected自动设置enabled=False）
        """
        return self.__unit_gdf[
            (self.__unit_gdf['unit_type'] == 'shop') &
            (self.__unit_gdf['replaceable'] == True) &
            (self.__unit_gdf['enabled'] == True)
        ]
    
    def get_shops_for_circulation(self) -> gpd.GeoDataFrame:
        """
        获取可转换为circulation的shop（用于SHOP_TO_CIRCULATION动作）
        
        条件:
        - unit_type == 'shop'
        - enabled == True（自动排除protected单元）
        """
        return self.__unit_gdf[
            (self.__unit_gdf['unit_type'] == 'shop') &
            (self.__unit_gdf['enabled'] == True)
        ]
    
    def get_atriums_for_circulation(self) -> gpd.GeoDataFrame:
        """
        获取可转换为circulation的atrium（用于ATRIUM_TO_CIRCULATION动作）
        
        条件:
        - unit_type == 'atrium'
        - enabled == True（自动排除protected单元）
        """
        return self.__unit_gdf[
            (self.__unit_gdf['unit_type'] == 'atrium') &
            (self.__unit_gdf['enabled'] == True)
        ]
    
    def get_public_spaces(self) -> gpd.GeoDataFrame:
        """获取公共空间（用于公共空间改造动作）"""
        return self.__unit_gdf[self.__unit_gdf['unit_type'] == 'public_space']
    
    def get_circulation_nodes(self) -> gpd.GeoDataFrame:
        """获取circulation节点（用于CLOSE_CIRCULATION_NODE动作）"""
        return self.__unit_gdf[self.__unit_gdf['unit_type'] == 'circulation']
    
    def get_protected_units(self) -> gpd.GeoDataFrame:
        """获取保护单元（用于约束检查）"""
        return self.__unit_gdf[self.__unit_gdf['protected'] == True]
    
    # endregion
    
    # region 更新操作（对应RL动作）
    
    def update_business_type(self, uid: uuid.UUID, new_business_type: str):
        """更新业态类型（对应CHANGE_BUSINESS动作 - 策略1）"""
        self.__unit_gdf.loc[uid, 'business_type'] = new_business_type
        self.__uid = uuid.uuid4()
    
    def convert_to_circulation(self, uid: uuid.UUID):
        """转换为circulation节点（对应SHOP_TO_CIRCULATION/ATRIUM_TO_CIRCULATION动作 - 策略2）"""
        self.__unit_gdf.loc[uid, 'unit_type'] = 'circulation'
        self.__uid = uuid.uuid4()

    def convert_to_public_space(self, uid: uuid.UUID):
        """Shop/Atrium → Public Space（策略2）"""
        self.__unit_gdf.loc[uid, 'unit_type'] = 'public_space'
        self.__unit_gdf.loc[uid, 'flow_prediction'] = 0.0
        self.__uid = uuid.uuid4()

    def close_public_space_node(self, uid: uuid.UUID):
        """关闭公共空间节点（策略2）"""
        self.__unit_gdf.loc[uid, 'unit_type'] = 'closed'
        self.__unit_gdf.loc[uid, 'flow_prediction'] = 0.0
        self.__uid = uuid.uuid4()
    
    def close_circulation_node(self, uid: uuid.UUID):
        """关闭circulation节点（对应CLOSE_CIRCULATION_NODE动作 - 策略2）"""
        self.__unit_gdf.loc[uid, 'enabled'] = False
        self.__uid = uuid.uuid4()
    
    def widen_public_space(self, public_space_uid: uuid.UUID, adjacent_unit_uid: uuid.UUID):
        """拓宽公共空间（对应WIDEN_PUBLIC_SPACE动作 - 策略3）"""
        # 合并相邻单元到公共空间
        # 更新geometry和area
        self.__uid = uuid.uuid4()
    
    def narrow_public_space(self, public_space_uid: uuid.UUID, occupancy_zone_id: int, 
                           zone_type: str, width_level: int):
        """收窄公共空间（对应NARROW_PUBLIC_SPACE动作 - 策略3）"""
        # 插入占用区域（户外座位、绿化、设施带）
        self.__uid = uuid.uuid4()
    
    def generate_pocket_node(self, public_space_uid: uuid.UUID, adjacent_unit_uids: List[uuid.UUID]):
        """生成口袋节点（对应GENERATE_POCKET_NODE动作 - 策略3）"""
        # 从公共空间和相邻单元生成pocket节点
        self.__uid = uuid.uuid4()
    
    def dissolve_pocket_node(self, pocket_node_uid: uuid.UUID):
        """回收口袋节点（对应DISSOLVE_POCKET_NODE动作 - 策略3）"""
        # 回收pocket节点
        self.__uid = uuid.uuid4()
    
    def regularize_boundary(self, public_space_uid: uuid.UUID, rule_level: int):
        """边界规整化（对应REGULARIZE_BOUNDARY动作 - 策略3）"""
        # 执行角落切割或边界对齐
        self.__uid = uuid.uuid4()
    
    def update_flow_prediction(self, uid: uuid.UUID, flow: float):
        """更新预测客流（STGNN输出）"""
        self.__unit_gdf.loc[uid, 'flow_prediction'] = flow
    
    # endregion
    
    # region 派生属性计算
    
    def compute_adjacent_units(self, distance_threshold: float = 0.1, use_spatial_index: bool = True):
        """
        计算所有单元的相邻关系（adjacent_units）
        
        使用空间关系分析确定哪些单元是相邻的：
        - touches(): 边界接触
        - distance() < threshold: 距离小于阈值视为相邻
        
        Args:
            distance_threshold: 距离阈值（米），小于此距离视为相邻
            use_spatial_index: 是否使用空间索引加速查询（推荐）
        """
        all_units = self.__unit_gdf
        
        if all_units.empty:
            return
        
        if use_spatial_index and STRtree is not None:
            # 使用空间索引加速查询
            geometries = all_units['geometry'].tolist()
            tree = STRtree(geometries)
            
            # 创建索引映射
            index_to_row = {i: idx for i, idx in enumerate(all_units.index)}
            
            for i, (idx, unit) in enumerate(all_units.iterrows()):
                adjacent_uids = []
                geometry = unit['geometry']
                
                # 查询可能相邻的单元
                possible_neighbors = tree.query(geometry)
                
                for neighbor_i in possible_neighbors:
                    if neighbor_i == i:
                        continue
                    
                    neighbor_idx = index_to_row[neighbor_i]
                    neighbor_unit = all_units.loc[neighbor_idx]
                    neighbor_geometry = neighbor_unit['geometry']
                    
                    # 检查空间关系
                    if geometry.touches(neighbor_geometry) or \
                       geometry.distance(neighbor_geometry) < distance_threshold:
                        adjacent_uids.append(neighbor_unit['uid'])
                
                self.__unit_gdf.loc[idx, 'adjacent_units'] = adjacent_uids
        else:
            # 暴力搜索（不使用空间索引）
            for idx1, unit1 in all_units.iterrows():
                adjacent_uids = []
                geometry1 = unit1['geometry']
                
                for idx2, unit2 in all_units.iterrows():
                    if idx1 == idx2:
                        continue
                    
                    geometry2 = unit2['geometry']
                    
                    # 检查空间关系
                    if geometry1.touches(geometry2) or \
                       geometry1.distance(geometry2) < distance_threshold:
                        adjacent_uids.append(unit2['uid'])
                
                self.__unit_gdf.loc[idx1, 'adjacent_units'] = adjacent_uids
        
        self.__uid = uuid.uuid4()  # 更新Collection UID
    
    def compute_min_clear_width(self, uid: uuid.UUID, method: str = 'min_rect'):
        """
        计算最小净宽（min_clear_width）
        
        方法：
        - 'min_rect': 使用最小外接矩形（快速，适合大多数情况）
        - 'skeleton': 使用骨架分析（更准确但更复杂，需要skimage）
        
        Args:
            uid: 空间单元UID
            method: 计算方法（'min_rect' 或 'skeleton'）
        """
        unit = self.get_space_unit_by_uid(uid)
        geometry = unit['geometry']
        
        if method == 'min_rect':
            # 方法1: 最小外接矩形宽度
            min_rect = geometry.minimum_rotated_rectangle
            bounds = min_rect.bounds
            width = bounds[2] - bounds[0]  # max_x - min_x
            height = bounds[3] - bounds[1]  # max_y - min_y
            min_clear_width = min(width, height)
        
        elif method == 'skeleton':
            # 方法2: 骨架分析（需要skimage，这里简化处理）
            # 如果skimage未安装，回退到min_rect方法
            min_rect = geometry.minimum_rotated_rectangle
            bounds = min_rect.bounds
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            min_clear_width = min(width, height)
        else:
            raise ValueError(f'不支持的计算方法: {method}')
        
        self.__unit_gdf.loc[uid, 'min_clear_width'] = min_clear_width
        self.__uid = uuid.uuid4()
    
    def compute_main_direction(self, uid: uuid.UUID, method: str = 'pca'):
        """
        计算主方向（main_direction）
        
        方法：
        - 'pca': 使用PCA分析主方向（推荐）
        - 'min_rect': 使用最小外接矩形方向
        
        Args:
            uid: 空间单元UID
            method: 计算方法（'pca' 或 'min_rect'）
        """
        unit = self.get_space_unit_by_uid(uid)
        geometry = unit['geometry']
        
        if method == 'pca' and PCA is not None:
            # 方法1: PCA分析主方向
            coords_2d = np.array(geometry.exterior.coords[:-1])  # 移除重复的最后一个点
            
            if len(coords_2d) < 2:
                main_direction = None
            elif PCA is None:
                # 如果sklearn未安装，无法使用PCA方法
                raise ImportError("sklearn库未安装，无法使用PCA方法计算主方向。请安装sklearn: pip install scikit-learn")
            else:
                # 中心化坐标
                coords_centered = coords_2d - coords_2d.mean(axis=0)
                
                # PCA分析
                pca = PCA(n_components=2)
                pca.fit(coords_centered)
                
                # 主方向向量（第一主成分）
                main_direction = pca.components_[0].tolist()
        
        elif method == 'min_rect':
            # 方法2: 最小外接矩形方向
            min_rect = geometry.minimum_rotated_rectangle
            coords = np.array(min_rect.exterior.coords[:-1])
            
            if len(coords) < 2:
                main_direction = None
            else:
                # 计算长边的方向向量
                edge1 = coords[1] - coords[0]
                edge2 = coords[2] - coords[1]
                
                # 选择较长的边作为主方向
                if np.linalg.norm(edge1) > np.linalg.norm(edge2):
                    direction = edge1
                else:
                    direction = edge2
                
                # 归一化
                norm = np.linalg.norm(direction)
                if norm > 0:
                    main_direction = (direction / norm).tolist()
                else:
                    main_direction = None
        
        else:
            # PCA不可用时，使用min_rect方法
            min_rect = geometry.minimum_rotated_rectangle
            coords = np.array(min_rect.exterior.coords[:-1])
            
            if len(coords) < 2:
                main_direction = None
            else:
                edge1 = coords[1] - coords[0]
                edge2 = coords[2] - coords[1]
                
                if np.linalg.norm(edge1) > np.linalg.norm(edge2):
                    direction = edge1
                else:
                    direction = edge2
                
                norm = np.linalg.norm(direction)
                if norm > 0:
                    main_direction = (direction / norm).tolist()
                else:
                    main_direction = None
        
        self.__unit_gdf.loc[uid, 'main_direction'] = main_direction
        self.__uid = uuid.uuid4()
    
    def compute_public_space_attributes(self, uid: uuid.UUID, 
                                       min_clear_width_method: str = 'min_rect',
                                       main_direction_method: str = 'pca'):
        """
        计算Public Space的所有相关属性
        
        包括：
        - min_clear_width: 最小净宽
        - main_direction: 主方向
        
        Args:
            uid: 空间单元UID
            min_clear_width_method: 最小净宽计算方法
            main_direction_method: 主方向计算方法
        """
        unit = self.get_space_unit_by_uid(uid)
        
        if unit['unit_type'] != 'public_space':
            raise ValueError(f'单元 {uid} 不是 public_space 类型')
        
        self.compute_min_clear_width(uid, method=min_clear_width_method)
        self.compute_main_direction(uid, method=main_direction_method)
    
    def compute_all_derived_attributes(self, 
                                     distance_threshold: float = 0.1,
                                     use_spatial_index: bool = True,
                                     min_clear_width_method: str = 'min_rect',
                                     main_direction_method: str = 'pca'):
        """
        计算所有单元的派生属性
        
        包括：
        1. adjacent_units: 所有单元的相邻关系
        2. min_clear_width: Public Space的最小净宽
        3. main_direction: Public Space的主方向
        
        Args:
            distance_threshold: 相邻单元距离阈值（米）
            use_spatial_index: 是否使用空间索引加速查询
            min_clear_width_method: 最小净宽计算方法
            main_direction_method: 主方向计算方法
        """
        # 1. 计算所有单元的相邻关系
        self.compute_adjacent_units(distance_threshold=distance_threshold, 
                                   use_spatial_index=use_spatial_index)
        
        # 2. 计算Public Space的属性
        public_spaces = self.get_public_spaces()
        for idx, unit in public_spaces.iterrows():
            uid = unit['uid']
            self.compute_public_space_attributes(
                uid,
                min_clear_width_method=min_clear_width_method,
                main_direction_method=main_direction_method
            )
    
    # endregion
    
    def get_attrs(self):
        """返回属性列表"""
        return self.__space_unit_attrs

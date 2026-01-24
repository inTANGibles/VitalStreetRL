"""街道网络集合：管理街道网络，支持连通打通和阻抗降低动作"""
import uuid
from typing import Optional, List, Dict, Any
import geopandas as gpd
import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point


class StreetNetworkCollection:
    """
    街道网络集合
    
    管理商业小镇的街道网络，支持：
    - 连通打通动作（connect_streets）
    - 阻抗降低动作（reduce_impedance）
    - 最小街宽约束检查
    """
    
    __street_attrs = [
        'uid',              # UUID标识
        'geometry',         # LineString几何
        'coords',           # 坐标数组
        'enabled',          # 是否启用
        'width',            # 街道宽度
        'impedance',        # 阻抗值
        'level',            # 街道等级
        'connectable',      # 是否可打通
        'protected',        # 是否受保护
        'flow_prediction'   # 预测客流
    ]
    
    __node_attrs = [
        'uid',
        'geometry',         # Point
        'coord',
        'connectivity_state'  # 连通状态
    ]
    
    __street_gdf = gpd.GeoDataFrame(columns=__street_attrs)
    __street_gdf.set_index('uid')
    
    __node_gdf = gpd.GeoDataFrame(columns=__node_attrs)
    __node_gdf.set_index('uid')
    
    __cached_graph = None
    __uid = uuid.uuid4()
    
    def uid(self):
        """返回Collection的UID"""
        return self.__uid
    
    # region 增删操作
    
    def add_street(self, street, return_uid: bool = True) -> Optional[uuid.UUID]:
        """添加街道"""
        if not self.__street_gdf.empty:
            self.__street_gdf = gpd.pd.concat([self.__street_gdf, street], ignore_index=False)
        else:
            self.__street_gdf = street
        self.__uid = uuid.uuid4()
        self.__cached_graph = None  # 清除缓存
        if return_uid:
            return street['uid'].iloc[0] if len(street) == 1 else street['uid']
    
    def delete_street(self, street):
        """删除街道"""
        uid = street['uid'] if hasattr(street, 'uid') else street
        self.__street_gdf.drop(uid, inplace=True)
        self.__uid = uuid.uuid4()
        self.__cached_graph = None
    
    # endregion
    
    # region 查询操作
    
    def get_street_by_uid(self, uid: uuid.UUID):
        """通过UID获取街道"""
        return self.__street_gdf.loc[uid]
    
    def get_all_streets(self) -> gpd.GeoDataFrame:
        """获取所有街道"""
        return self.__street_gdf
    
    def get_connectable_streets(self) -> gpd.GeoDataFrame:
        """获取可打通的街道（用于动作mask）"""
        return self.__street_gdf[
            (self.__street_gdf['connectable'] == True) &
            (self.__street_gdf['protected'] == False) &
            (self.__street_gdf['enabled'] == True)
        ]
    
    def get_network_graph(self) -> nx.Graph:
        """获取NetworkX图（用于STGNN）"""
        if self.__cached_graph is None:
            self.__cached_graph = self._build_graph()
        return self.__cached_graph
    
    def _build_graph(self) -> nx.Graph:
        """构建NetworkX图"""
        G = nx.Graph()
        # 添加节点和边
        # ... 实现细节
        return G
    
    # endregion
    
    # region 更新操作（对应RL动作）
    
    def connect_streets(self, street_uids: List[uuid.UUID]):
        """执行连通打通动作（对应CONNECT动作）"""
        # 更新街道连通性
        # 更新节点连通状态
        # 重建图
        self.__cached_graph = None
        self.__uid = uuid.uuid4()
    
    def reduce_impedance(self, uid: uuid.UUID, new_impedance: float):
        """执行阻抗降低动作（对应REDUCE_IMPEDANCE动作）"""
        self.__street_gdf.loc[uid, 'impedance'] = new_impedance
        self.__uid = uuid.uuid4()
    
    def check_min_width(self, min_width: float) -> Dict[uuid.UUID, bool]:
        """检查最小街宽约束"""
        violations = {}
        for uid, street in self.__street_gdf.iterrows():
            violations[uid] = street['width'] < min_width
        return violations
    
    # endregion
    
    def get_attrs(self):
        """返回属性列表"""
        return self.__street_attrs

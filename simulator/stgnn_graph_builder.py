"""STGNN图构建：将状态转换为STGNN输入图结构

注意：此模块用于构建STGNN（时空图神经网络）的输入图，不是用于RL观测。
RL观测使用RasterObservation（栅格化观测），由CNN处理。
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
from env.world_state import WorldState


class STGNNGraphBuilder:
    """STGNN图构建器：将WorldState转换为STGNN输入图结构"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化图构建配置
        
        Args:
            config: 配置字典，包含：
                - node_feature_dim: 节点特征维度
                - edge_feature_dim: 边特征维度
                - use_spatial_edges: 是否使用空间相邻关系作为边
                - use_street_edges: 是否使用街道网络作为边
        """
        self.config = config
        self.node_feature_dim = config.get('node_feature_dim', 32)
        self.edge_feature_dim = config.get('edge_feature_dim', 16)
        self.use_spatial_edges = config.get('use_spatial_edges', True)
        self.use_street_edges = config.get('use_street_edges', True)
    
    def build_graph(self, state: WorldState) -> Dict[str, np.ndarray]:
        """
        构建STGNN输入图结构
        
        Returns:
            graph_data: {
                'node_features': (N, F_n),      # 节点特征矩阵
                'edge_index': (2, E),           # 边索引 [source, target]
                'edge_features': (E, F_e),       # 边特征矩阵（可选）
                'node_positions': (N, 2),        # 节点位置（可选）
            }
        """
        # 1. 提取节点特征
        node_features = self._extract_node_features(state)
        
        # 2. 构建边（空间相邻关系 + 街道网络）
        edge_index, edge_features = self._build_edges(state)
        
        graph_data = {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
        }
        
        return graph_data
    
    def _extract_node_features(self, state: WorldState) -> np.ndarray:
        """
        提取节点特征（用于STGNN）
        
        节点特征包括：
        - 空间配置特征（面积、形状、位置等）
        - 业态特征（业态类型、业态类别等）
        - 界面特征（临街长度、相邻单元数等）
        - 容量特征（面积、最大容量等）
        - 其他特征（保护状态、可替换性等）
        
        Returns:
            node_features: (N, F_n) 节点特征矩阵
        """
        space_units = state.space_units.get_all_space_units()
        n_nodes = len(space_units)
        
        # 初始化特征矩阵
        features = []
        
        for idx, unit in space_units.iterrows():
            node_feat = []
            
            # 空间配置特征
            node_feat.extend([
                unit['area'],
                unit['geometry'].centroid.x if hasattr(unit['geometry'], 'centroid') else 0.0,
                unit['geometry'].centroid.y if hasattr(unit['geometry'], 'centroid') else 0.0,
            ])
            
            # 单元类型特征（one-hot编码）
            unit_type_onehot = {
                'shop': [1, 0, 0, 0],
                'atrium': [0, 1, 0, 0],
                'public_space': [0, 0, 1, 0],
                'circulation': [0, 0, 0, 1],
            }
            node_feat.extend(unit_type_onehot.get(unit['unit_type'], [0, 0, 0, 0]))
            
            # 状态特征
            node_feat.extend([
                float(unit['enabled']),
                float(unit['protected']),
                float(unit['replaceable']),
            ])
            
            # 业态特征（如果是shop）
            if unit['unit_type'] == 'shop':
                # 业态类型编码（简化处理）
                business_type_encoded = hash(unit.get('business_type', 'UNDEFINED')) % 100
                node_feat.append(float(business_type_encoded))
            else:
                node_feat.append(0.0)
            
            # 公共空间特征
            if unit['unit_type'] == 'public_space':
                node_feat.extend([
                    unit.get('min_clear_width', 0.0),
                    # main_direction 可以编码为2个特征（方向向量的x, y）
                    unit.get('main_direction', [0.0, 0.0])[0] if unit.get('main_direction') else 0.0,
                    unit.get('main_direction', [0.0, 0.0])[1] if unit.get('main_direction') else 0.0,
                ])
            else:
                node_feat.extend([0.0, 0.0, 0.0])
            
            # 相邻单元数
            adjacent_count = len(unit.get('adjacent_units', []))
            node_feat.append(float(adjacent_count))
            
            # 预测客流（STGNN输出，用于下一轮预测）
            node_feat.append(unit.get('flow_prediction', 0.0))
            
            features.append(node_feat)
        
        # 转换为numpy数组
        node_features = np.array(features, dtype=np.float32)
        
        # 如果特征维度不匹配，进行填充或截断
        if node_features.shape[1] < self.node_feature_dim:
            # 填充零
            padding = np.zeros((n_nodes, self.node_feature_dim - node_features.shape[1]), dtype=np.float32)
            node_features = np.concatenate([node_features, padding], axis=1)
        elif node_features.shape[1] > self.node_feature_dim:
            # 截断
            node_features = node_features[:, :self.node_feature_dim]
        
        return node_features
    
    def _build_edges(self, state: WorldState) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        构建边（空间相邻关系 + 街道网络）
        
        Returns:
            edge_index: (2, E) 边索引矩阵
            edge_features: (E, F_e) 边特征矩阵（可选）
        """
        space_units = state.space_units.get_all_space_units()
        edges = []
        edge_features_list = []
        
        # 1. 空间相邻关系边
        if self.use_spatial_edges:
            for idx, unit in space_units.iterrows():
                adjacent_uids = unit.get('adjacent_units', [])
                unit_idx = space_units.index.get_loc(idx)
                
                for adj_uid in adjacent_uids:
                    try:
                        adj_idx = space_units.index.get_loc(adj_uid)
                        # 无向边，添加两个方向
                        edges.append([unit_idx, adj_idx])
                        edges.append([adj_idx, unit_idx])
                        
                        # 边特征：距离、共享边界长度等
                        if self.edge_feature_dim > 0:
                            # 计算距离
                            distance = unit['geometry'].distance(
                                space_units.loc[adj_uid]['geometry']
                            )
                            edge_feat = [distance]
                            # 填充到指定维度
                            while len(edge_feat) < self.edge_feature_dim:
                                edge_feat.append(0.0)
                            edge_feat = edge_feat[:self.edge_feature_dim]
                            edge_features_list.extend([edge_feat, edge_feat])
                    except KeyError:
                        # 相邻单元不在当前集合中，跳过
                        continue
        
        # 2. 街道网络边（如果启用）
        if self.use_street_edges and state.street_network:
            # TODO: 从street_network构建边
            # 这里需要根据street_network的实际结构来实现
            pass
        
        # 转换为numpy数组
        if len(edges) == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)
        else:
            edge_index = np.array(edges, dtype=np.int64).T
        
        if len(edge_features_list) > 0 and self.edge_feature_dim > 0:
            edge_features = np.array(edge_features_list, dtype=np.float32)
        else:
            edge_features = None
        
        return edge_index, edge_features

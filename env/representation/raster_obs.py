"""栅格化观测：固定维度多通道画布（CNN policy用）"""
from typing import Tuple, Dict, Any, Optional
import numpy as np
import geopandas as gpd
try:
    from shapely.geometry import Polygon, Point, box
    from shapely.ops import unary_union
except ImportError:
    Polygon = None
    Point = None
    box = None
    unary_union = None

from ..world_state import WorldState


class RasterObservation:
    """栅格化观测编码器：将WorldState转换为多通道栅格图像"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 包含resolution, channels定义等
                - resolution: [H, W] 栅格分辨率，或 'auto' 表示自动计算
                - channels: 通道名称列表
                - target_resolution: [H, W] 目标分辨率（用于填充到固定尺寸，默认: None表示不填充）
                - auto_resolution: 当resolution='auto'时使用的参数
                    - target_pixels: 目标像素数（用于长边，默认: 256）
                    - min_resolution: 最小分辨率（默认: 128）
                    - max_resolution: 最大分辨率（默认: 1024）
        """
        self.channels = config['channels']  # 通道定义列表
        self.n_channels = len(self.channels)
        
        # 处理分辨率配置
        resolution = config.get('resolution', [256, 256])
        if resolution == 'auto':
            # 延迟到encode时计算
            self.resolution = None
            self._auto_resolution_config = config.get('auto_resolution', {})
        else:
            self.resolution = resolution  # (H, W)
            self._auto_resolution_config = None
        
        # 目标分辨率（用于填充到固定尺寸）
        self.target_resolution = config.get('target_resolution', None)  # (H, W) 或 None
        
        # 缓存边界框和变换矩阵
        self._bounds = None
        self._transform = None
    
    @staticmethod
    def compute_resolution_from_bounds(
        bounds: Dict[str, float],
        target_pixels: int = 256,
        min_resolution: int = 128,
        max_resolution: int = 1024
    ) -> list:
        """
        根据边界框计算保持宽高比的分辨率（静态方法）
        
        Args:
            bounds: 边界框字典 {'x_min', 'x_max', 'y_min', 'y_max'}
            target_pixels: 目标像素数（用于长边）
            min_resolution: 最小分辨率
            max_resolution: 最大分辨率
            
        Returns:
            resolution: [H, W] 分辨率
        """
        x_range = bounds['x_max'] - bounds['x_min']
        y_range = bounds['y_max'] - bounds['y_min']
        
        # 计算宽高比
        if x_range > y_range:
            # 宽度更大
            aspect_ratio = y_range / x_range
            w = target_pixels
            h = max(min_resolution, min(max_resolution, int(target_pixels * aspect_ratio)))
        else:
            # 高度更大
            aspect_ratio = x_range / y_range
            h = target_pixels
            w = max(min_resolution, min(max_resolution, int(target_pixels * aspect_ratio)))
        
        return [h, w]
    
    @classmethod
    def create_with_auto_resolution(
        cls,
        state: 'WorldState',
        channels: list,
        target_pixels: int = 256,
        min_resolution: int = 128,
        max_resolution: int = 1024,
        padding: float = 0.05,
        target_resolution: Optional[list] = None
    ) -> 'RasterObservation':
        """
        创建栅格观测编码器，自动计算保持宽高比的分辨率（类方法）
        
        Args:
            state: 世界状态
            channels: 通道列表
            target_pixels: 目标像素数（用于长边）
            min_resolution: 最小分辨率
            max_resolution: 最大分辨率
            padding: 边界填充比例
            target_resolution: [H, W] 目标分辨率（用于填充到固定尺寸，默认: None）
            
        Returns:
            raster_obs: RasterObservation对象
        """
        # 计算边界框
        space_units = state.space_units.get_all_space_units()
        
        if space_units.empty:
            bounds = {
                'x_min': 0.0,
                'x_max': 1000.0,
                'y_min': 0.0,
                'y_max': 1000.0
            }
        else:
            total_bounds = space_units.total_bounds  # [x_min, y_min, x_max, y_max]
            x_range = total_bounds[2] - total_bounds[0]
            y_range = total_bounds[3] - total_bounds[1]
            
            bounds = {
                'x_min': total_bounds[0] - x_range * padding,
                'x_max': total_bounds[2] + x_range * padding,
                'y_min': total_bounds[1] - y_range * padding,
                'y_max': total_bounds[3] + y_range * padding
            }
        
        # 计算保持宽高比的分辨率
        resolution = cls.compute_resolution_from_bounds(
            bounds, target_pixels, min_resolution, max_resolution
        )
        
        # 创建配置
        raster_config = {
            'resolution': resolution,
            'channels': channels,
            'target_resolution': target_resolution
        }
        
        # 创建栅格观测编码器
        return cls(raster_config)
    
    def encode(self, state: WorldState) -> np.ndarray:
        """
        将状态编码为栅格观测
        
        Returns:
            obs: shape=(n_channels, H, W) 的numpy数组，dtype=float32
                如果设置了target_resolution，会自动填充到目标尺寸
        """
        step_idx = state.step_idx if hasattr(state, 'step_idx') else 0
        
        # 如果分辨率是auto，先计算分辨率
        if self.resolution is None:
            self._compute_auto_resolution(state)
        
        # 计算边界框（如果还没有缓存）
        if self._bounds is None:
            self._compute_bounds(state)
        
        # 为每个通道创建栅格
        obs_channels = []
        for i, channel_name in enumerate(self.channels):
            channel_data = self._render_channel(state, channel_name)
            obs_channels.append(channel_data)
        
        # 堆叠为多通道数组 (C, H, W)
        obs = np.stack(obs_channels, axis=0).astype(np.float32)
        
        # 如果设置了target_resolution，填充到目标尺寸
        if self.target_resolution is not None:
            obs = self._pad_to_target_resolution(obs)
        
        return obs
    
    def _pad_to_target_resolution(self, obs: np.ndarray) -> np.ndarray:
        """将观测填充到目标分辨率（居中填充）"""
        current_shape = obs.shape  # (C, H, W)
        target_shape = (self.n_channels, self.target_resolution[0], self.target_resolution[1])
        
        if current_shape == target_shape:
            return obs
        
        # 如果当前尺寸大于目标尺寸，裁剪（居中裁剪）
        c, h, w = current_shape
        target_h, target_w = self.target_resolution[0], self.target_resolution[1]
        
        if h > target_h or w > target_w:
            h_start = (h - target_h) // 2
            w_start = (w - target_w) // 2
            obs = obs[:, h_start:h_start+target_h, w_start:w_start+target_w]
            h, w = target_h, target_w
        
        # 创建目标尺寸的零数组并居中填充
        padded = np.zeros(target_shape, dtype=obs.dtype)
        h_start = (target_h - h) // 2
        w_start = (target_w - w) // 2
        
        padded[:, h_start:h_start+h, w_start:w_start+w] = obs
        
        return padded
    
    def _compute_auto_resolution(self, state: WorldState):
        """自动计算分辨率（当resolution='auto'时调用）"""
        # 先计算边界框
        space_units = state.space_units.get_all_space_units()
        
        if space_units.empty:
            bounds = {
                'x_min': 0.0,
                'x_max': 1000.0,
                'y_min': 0.0,
                'y_max': 1000.0
            }
        else:
            total_bounds = space_units.total_bounds
            x_range = total_bounds[2] - total_bounds[0]
            y_range = total_bounds[3] - total_bounds[1]
            padding = 0.05
            
            bounds = {
                'x_min': total_bounds[0] - x_range * padding,
                'x_max': total_bounds[2] + x_range * padding,
                'y_min': total_bounds[1] - y_range * padding,
                'y_max': total_bounds[3] + y_range * padding
            }
        
        # 计算分辨率
        auto_config = self._auto_resolution_config or {}
        target_pixels = auto_config.get('target_pixels', 256)
        min_resolution = auto_config.get('min_resolution', 128)
        max_resolution = auto_config.get('max_resolution', 1024)
        
        self.resolution = self.compute_resolution_from_bounds(
            bounds, target_pixels, min_resolution, max_resolution
        )
    
    def _compute_bounds(self, state: WorldState, padding: float = 0.05):
        """
        计算所有空间单元的边界框
        
        Args:
            state: 世界状态
            padding: 边界填充比例（相对于范围）
        """
        space_units = state.space_units.get_all_space_units()
        
        if space_units.empty:
            # 默认边界
            self._bounds = {
                'x_min': 0.0,
                'x_max': 1000.0,
                'y_min': 0.0,
                'y_max': 1000.0
            }
        else:
            # 计算所有几何的边界
            try:
                bounds = space_units.total_bounds  # [x_min, y_min, x_max, y_max]
                
                # 检查边界是否有效
                if not np.isfinite(bounds).all() or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                    print(f"[警告] 无效的边界框: {bounds}, 使用默认边界")
                    self._bounds = {
                        'x_min': 0.0,
                        'x_max': 1000.0,
                        'y_min': 0.0,
                        'y_max': 1000.0
                    }
                    return
                
                x_range = bounds[2] - bounds[0]
                y_range = bounds[3] - bounds[1]
                
                # 检查范围是否有效
                if x_range <= 0 or y_range <= 0 or not np.isfinite(x_range) or not np.isfinite(y_range):
                    print(f"[警告] 无效的边界范围: x_range={x_range}, y_range={y_range}, 使用默认边界")
                    self._bounds = {
                        'x_min': 0.0,
                        'x_max': 1000.0,
                        'y_min': 0.0,
                        'y_max': 1000.0
                    }
                    return
                
                self._bounds = {
                    'x_min': bounds[0] - x_range * padding,
                    'x_max': bounds[2] + x_range * padding,
                    'y_min': bounds[1] - y_range * padding,
                    'y_max': bounds[3] + y_range * padding
                }
            except Exception as e:
                print(f"[错误] 计算边界框失败: {e}, 使用默认边界")
                self._bounds = {
                    'x_min': 0.0,
                    'x_max': 1000.0,
                    'y_min': 0.0,
                    'y_max': 1000.0
                }
        
        # 计算变换矩阵（世界坐标 -> 栅格坐标）
        x_range = self._bounds['x_max'] - self._bounds['x_min']
        y_range = self._bounds['y_max'] - self._bounds['y_min']
        
        # 检查范围是否有效，避免除零错误
        if x_range <= 0 or y_range <= 0 or not np.isfinite(x_range) or not np.isfinite(y_range):
            print(f"[错误] 无效的边界范围用于计算变换: x_range={x_range}, y_range={y_range}")
            # 使用默认值
            x_range = max(x_range, 1.0)
            y_range = max(y_range, 1.0)
        
        self._transform = {
            'x_scale': self.resolution[1] / x_range,  # W / x_range
            'y_scale': self.resolution[0] / y_range,  # H / y_range
            'x_min': self._bounds['x_min'],
            'y_min': self._bounds['y_min']
        }
        
        # 检查变换是否有效
        if not np.isfinite(self._transform['x_scale']) or not np.isfinite(self._transform['y_scale']):
            print(f"[错误] 无效的变换参数: x_scale={self._transform['x_scale']}, y_scale={self._transform['y_scale']}")
            # 使用默认变换
            self._transform = {
                'x_scale': self.resolution[1] / 1000.0,
                'y_scale': self.resolution[0] / 1000.0,
                'x_min': 0.0,
                'y_min': 0.0
            }
    
    def _world_to_raster(self, x: float, y: float) -> Tuple[int, int]:
        """
        将世界坐标转换为栅格坐标
        
        Args:
            x, y: 世界坐标
            
        Returns:
            i, j: 栅格坐标 (行, 列)
        """
        if self._transform is None:
            raise ValueError("边界框未初始化，请先调用 _compute_bounds")
        
        j = int((x - self._transform['x_min']) * self._transform['x_scale'])
        i = int((y - self._transform['y_min']) * self._transform['y_scale'])
        
        # 限制在有效范围内
        i = max(0, min(self.resolution[0] - 1, i))
        j = max(0, min(self.resolution[1] - 1, j))
        
        return i, j
    
    def _rasterize_polygon(self, geometry: Polygon, value: float = 1.0) -> np.ndarray:
        """
        将多边形栅格化到栅格图像
        
        使用高效的栅格化方法：先计算边界框，然后批量检查栅格点
        
        Args:
            geometry: Shapely多边形
            value: 填充值
            
        Returns:
            raster: (H, W) 栅格图像
        """
        # 注意：这个方法涉及大量shapely几何操作，可能是崩溃点
        raster = np.zeros((self.resolution[0], self.resolution[1]), dtype=np.float32)
        
        if geometry is None or geometry.is_empty or Point is None:
            return raster
        
        # 计算多边形的边界框（栅格坐标）- shapely操作
        try:
            # 检查几何对象是否有效
            if not hasattr(geometry, 'bounds') or geometry.is_empty:
                return raster
            
            bounds = geometry.bounds  # (x_min, y_min, x_max, y_max) - 可能崩溃点
            
            # 检查边界是否有效
            if len(bounds) != 4 or not all(np.isfinite(bounds)) or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                return raster
        except Exception as e:
            print(f"[警告] 获取几何边界失败: {e}, 跳过此多边形")
            return raster
        
        i_min, j_min = self._world_to_raster(bounds[0], bounds[1])
        i_max, j_max = self._world_to_raster(bounds[2], bounds[3])
        
        # 限制在有效范围内
        i_min = max(0, i_min)
        i_max = min(self.resolution[0] - 1, i_max)
        j_min = max(0, j_min)
        j_max = min(self.resolution[1] - 1, j_max)
        
        if i_min > i_max or j_min > j_max:
            return raster
        
        # 使用prepared geometry加速（如果可用）- shapely操作
        try:
            from shapely.prepared import prep
            # 检查几何对象是否有效
            if geometry is None or not hasattr(geometry, 'is_valid') or not geometry.is_valid:
                # 如果几何对象无效，尝试修复
                try:
                    geometry = geometry.buffer(0)  # 尝试修复无效几何
                except:
                    print(f"[警告] 几何对象无效且无法修复，跳过")
                    return raster
            
            prepared_geom = prep(geometry)  # 可能崩溃点
        except ImportError:
            prepared_geom = geometry
        except Exception as e:
            print(f"[警告] 准备几何对象失败: {e}, 使用原始几何对象")
            prepared_geom = geometry
        
        # 批量生成栅格点坐标
        i_coords, j_coords = np.mgrid[i_min:i_max+1, j_min:j_max+1]
        
        # 将栅格坐标转换为世界坐标（栅格中心点）
        world_x = self._transform['x_min'] + (j_coords + 0.5) / self._transform['x_scale']
        world_y = self._transform['y_min'] + (i_coords + 0.5) / self._transform['y_scale']
        
        # 展平坐标数组
        world_x_flat = world_x.flatten()
        world_y_flat = world_y.flatten()
        
        # 批量检查点是否在多边形内（使用prepared geometry加速）- 大量shapely操作，可能崩溃点
        # 限制处理的点数，避免内存溢出
        max_points = 10000  # 限制最大点数
        if len(world_x_flat) > max_points:
            # 如果点数太多，只处理边界框内的部分区域
            print(f"[警告] 栅格化点数过多 ({len(world_x_flat)}), 跳过此多边形以避免内存错误")
            return raster
        
        try:
            # 使用更安全的方式：批量创建Point对象并检查
            contains_mask = np.zeros(len(world_x_flat), dtype=bool)
            
            # 分批处理，避免一次性创建太多对象
            batch_size = 1000
            for i in range(0, len(world_x_flat), batch_size):
                end_idx = min(i + batch_size, len(world_x_flat))
                batch_x = world_x_flat[i:end_idx]
                batch_y = world_y_flat[i:end_idx]
                
                # 创建Point对象并检查
                try:
                    if hasattr(prepared_geom, 'contains'):
                        # 使用prepared geometry
                        batch_points = [Point(x, y) for x, y in zip(batch_x, batch_y)]
                        batch_mask = [prepared_geom.contains(p) for p in batch_points]
                    else:
                        # 回退到普通geometry
                        batch_points = [Point(x, y) for x, y in zip(batch_x, batch_y)]
                        batch_mask = [geometry.contains(p) or geometry.touches(p) for p in batch_points]
                    
                    contains_mask[i:end_idx] = batch_mask
                except Exception as batch_e:
                    # 如果批次处理失败，跳过这个批次
                    print(f"[警告] 批次 {i//batch_size} 处理失败: {batch_e}, 跳过")
                    continue
        except Exception as e:
            print(f"[错误] 栅格化多边形失败: {e}, 返回空栅格")
            return raster
        
        # 重塑为原始形状并填充值
        try:
            contains_mask = contains_mask.reshape(i_coords.shape)
            # 再次检查索引范围（防止在计算过程中发生变化）
            if i_min >= 0 and i_max < self.resolution[0] and j_min >= 0 and j_max < self.resolution[1]:
                raster[i_min:i_max+1, j_min:j_max+1] = np.where(contains_mask, value, 0.0)
            else:
                # 如果索引超出范围，只填充有效部分
                i_min_safe = max(0, i_min)
                i_max_safe = min(self.resolution[0] - 1, i_max)
                j_min_safe = max(0, j_min)
                j_max_safe = min(self.resolution[1] - 1, j_max)
                
                if i_min_safe <= i_max_safe and j_min_safe <= j_max_safe:
                    # 计算需要提取的 mask 部分
                    mask_i_start = i_min_safe - i_min
                    mask_i_end = mask_i_start + (i_max_safe - i_min_safe + 1)
                    mask_j_start = j_min_safe - j_min
                    mask_j_end = mask_j_start + (j_max_safe - j_min_safe + 1)
                    
                    if (mask_i_end <= contains_mask.shape[0] and mask_j_end <= contains_mask.shape[1] and
                        mask_i_start >= 0 and mask_j_start >= 0):
                        raster[i_min_safe:i_max_safe+1, j_min_safe:j_max_safe+1] = np.where(
                            contains_mask[mask_i_start:mask_i_end, mask_j_start:mask_j_end], value, 0.0
                        )
        except Exception as e:
            print(f"[错误] 填充栅格值失败: {e}, 返回空栅格")
            return raster
        
        return raster
    
    def _render_channel(self, state: WorldState, channel_name: str) -> np.ndarray:
        """
        渲染单个通道
        
        Args:
            state: 世界状态
            channel_name: 通道名称
            
        Returns:
            channel: (H, W) 栅格图像
        """
        space_units = state.space_units.get_all_space_units()
        H, W = self.resolution
        
        # 初始化通道
        channel = np.zeros((H, W), dtype=np.float32)
        
        if space_units.empty:
            return channel
        
        # 根据通道名称渲染不同的内容
        if channel_name == 'walkable_mask':
            # 通道1: Street/Walkable Space Map (白=可走)
            # 包括 circulation 和 public_space
            walkable_units = space_units[
                (space_units['unit_type'] == 'circulation') | 
                (space_units['unit_type'] == 'public_space')
            ]
            
            for idx, unit in walkable_units.iterrows():
                geometry = unit['geometry']
                raster = self._rasterize_polygon(geometry, value=1.0)
                channel = np.maximum(channel, raster)
        
        elif channel_name == 'predicted_flow':
            # 通道2: Predicted Flow Map (Heatmap)
            # 只针对public_space类型的单元（walkable_place）
            # 使用百分位数归一化: flow_norm = clip(flow / P95(flow), 0, 1)
            
            # 只考虑public_space类型的单元
            public_space_units = space_units[space_units['unit_type'] == 'public_space']
            
            if len(public_space_units) == 0:
                return channel
            
            # 首先收集所有flow_prediction值
            # 双重检查：确保只处理public_space类型的单元
            flow_values = []
            unit_geometries = []
            for idx, unit in public_space_units.iterrows():
                # 显式检查unit_type，确保只处理public_space
                if unit.get('unit_type') != 'public_space':
                    continue  # 跳过非public_space单元
                flow = unit.get('flow_prediction', 0.0)
                if flow > 0:
                    flow_values.append(flow)
                    unit_geometries.append((idx, unit['geometry'], flow))
            
            if len(flow_values) == 0:
                return channel
            
            # 计算P95（95百分位数）
            flow_array = np.array(flow_values)
            p95 = np.percentile(flow_array, 95)
            
            # 如果P95为0，使用最大值作为归一化因子
            if p95 == 0:
                p95 = np.max(flow_array) if len(flow_array) > 0 else 1.0
            
            # 归一化并栅格化
            for idx, geometry, flow in unit_geometries:
                normalized_flow = np.clip(flow / p95, 0.0, 1.0)
                raster = self._rasterize_polygon(geometry, value=normalized_flow)
                channel = np.maximum(channel, raster)
        
        elif channel_name == 'landuse_id':
            # 通道3: Land Use Distribution Map (8 Grayscale Categories)
            # 将土地利用类型映射到8个离散灰度值 [0, 1]
            
            # 导入BusinessCategory用于映射
            try:
                from ..geo.business_type import BusinessCategory
            except ImportError:
                BusinessCategory = None
            
            # 8个BusinessCategory字符串值映射到灰度值 (0.0 到 1.0，均匀分布)
            # 每个类别映射到一个离散值
            category_str_to_gray = {
                'dining': 0.125,              # 1/8
                'retail': 0.25,                # 2/8
                'cultural': 0.375,            # 3/8
                'service': 0.5,                # 4/8
                'leisure': 0.625,             # 5/8
                'residential': 0.75,          # 6/8
                'supporting': 0.875,          # 7/8
                'undefined': 1.0,              # 8/8
            }
            
            # 非shop单元类型的映射
            unit_type_to_gray = {
                'atrium': 0.0,
                'public_space': 0.0,
                'circulation': 0.0,
            }
            
            for idx, unit in space_units.iterrows():
                geometry = unit['geometry']
                unit_type = unit['unit_type']
                
                # 确定灰度值
                if unit_type == 'shop':
                    # shop单元：根据business_category映射
                    business_category = unit.get('business_category')
                    if business_category is None:
                        gray_value = category_str_to_gray.get('undefined', 1.0)
                    elif isinstance(business_category, str):
                        # 字符串形式（如"dining", "retail"等）
                        gray_value = category_str_to_gray.get(business_category.lower(), 1.0)
                    elif BusinessCategory is not None:
                        # 如果是枚举，转换为字符串
                        try:
                            cat_str = business_category.value if hasattr(business_category, 'value') else str(business_category).lower()
                            gray_value = category_str_to_gray.get(cat_str, 1.0)
                        except (AttributeError, TypeError):
                            gray_value = category_str_to_gray.get('undefined', 1.0)
                    else:
                        gray_value = 1.0  # 默认值
                else:
                    # 非shop单元：使用unit_type映射
                    gray_value = unit_type_to_gray.get(unit_type, 0.0)
                
                raster = self._rasterize_polygon(geometry, value=gray_value)
                # 使用最大值（重叠区域取最大灰度值）
                channel = np.maximum(channel, raster)
        
        else:
            # 未知通道名称，返回零图像
            pass
        
        return channel
    
    def get_obs_shape(self) -> Tuple[int, int, int]:
        """返回观测形状 (C, H, W)"""
        return (self.n_channels, self.resolution[0], self.resolution[1])
    
    def get_bounds(self) -> Optional[Dict[str, float]]:
        """获取当前边界框"""
        return self._bounds.copy() if self._bounds else None
    
    def reset_bounds(self):
        """重置边界框缓存（当场景变化时调用）"""
        self._bounds = None
        self._transform = None

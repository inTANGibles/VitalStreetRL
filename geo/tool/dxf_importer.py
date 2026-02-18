"""DXF文件导入工具：将DXF文件转换为SpaceUnitCollection

本模块负责：
1. 加载和解析DXF文件
2. 提取几何数据（坐标点）
3. 验证和修复几何
4. 调用 SpaceUnitCollection.parse_layer_name() 解析图层名称
5. 创建 SpaceUnitCollection 对象

注意：图层解析功能由 SpaceUnitCollection.parse_layer_name() 提供，本模块不重复实现。
"""
import os
from typing import Optional, Dict, Any
import numpy as np
import ezdxf
try:
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.errors import ShapelyError
    SHAPELY_AVAILABLE = True
except ImportError:
    Polygon = None
    MultiPolygon = None
    ShapelyError = Exception
    SHAPELY_AVAILABLE = False

from .space_unit import SpaceUnitCollection
from .business_type import BusinessCategory
from .polyline_utils import get_closed_polyline


def _get_entity_points_auto(entity) -> list:
    """
    从DXF实体自动提取坐标点
    
    Args:
        entity: DXF实体对象
        
    Returns:
        points: 坐标点列表 [[x1, y1], [x2, y2], ...]
        
    Raises:
        ValueError: 不支持的实体类型
    """
    entity_type = entity.dxftype()
    
    if entity_type == 'LWPOLYLINE':
        # 轻量级多段线
        points = np.array(entity.get_points())
        points = points[:, :2].tolist()  # 只取x, y坐标
        return points
    
    elif entity_type == 'POLYLINE':
        # 多段线
        points = []
        for pt in entity.points():
            points.append([pt.xyz[0], pt.xyz[1]])
        return points
    
    elif entity_type == 'LINE':
        # 直线（需要闭合才能形成多边形）
        return [
            [entity.dxf.start.xyz[0], entity.dxf.start.xyz[1]],
            [entity.dxf.end.xyz[0], entity.dxf.end.xyz[1]]
        ]
    
    else:
        raise ValueError(f'不支持的实体类型: {entity_type}')


def _validate_and_repair_geometry(coords: np.ndarray, simplify_tolerance: Optional[float] = None) -> Optional[Polygon]:
    """
    验证和修复几何数据，支持复杂多边形的处理
    
    Args:
        coords: 坐标数组
        simplify_tolerance: 简化容差（None表示自动计算，或指定具体值如0.1）
        
    Returns:
        geometry: 修复后的Polygon对象，如果无效则返回None
    """
    if not SHAPELY_AVAILABLE or Polygon is None:
        raise ImportError("shapely库未安装，无法处理几何数据")
    
    try:
        # 确保至少3个点才能形成多边形
        if len(coords) < 3:
            return None
        
        # 移除重复的连续点
        coords_clean = []
        prev_point = None
        for point in coords:
            if prev_point is None or not np.allclose(point, prev_point, atol=1e-6):
                coords_clean.append(point)
                prev_point = point
        
        if len(coords_clean) < 3:
            return None
        
        coords = np.array(coords_clean)
        
        # 创建Polygon
        geometry = Polygon(coords)
        
        # 检查几何有效性
        if not geometry.is_valid:
            # 方法1: 尝试使用make_valid（Shapely 2.0+）
            try:
                if hasattr(geometry, 'make_valid'):
                    geometry = geometry.make_valid()
                elif hasattr(geometry, 'buffer'):
                    # 方法2: 使用buffer(0)修复
                    repaired = geometry.buffer(0)
                    
                    # buffer(0) 可能返回 MultiPolygon，需要提取最大的多边形
                    if isinstance(repaired, MultiPolygon):
                        if len(repaired.geoms) == 0:
                            return None
                        # 选择面积最大的多边形
                        geometry = max(repaired.geoms, key=lambda p: p.area)
                    else:
                        geometry = repaired
            except Exception as e:
                # 如果修复失败，尝试简化后再修复
                try:
                    # 计算合适的简化容差（基于坐标范围）
                    if simplify_tolerance is None:
                        coords_range = np.ptp(coords, axis=0)
                        simplify_tolerance = max(coords_range) * 0.001  # 0.1%的坐标范围
                    
                    # 先简化多边形（减少顶点数）
                    geometry_simplified = geometry.simplify(simplify_tolerance, preserve_topology=True)
                    
                    # 再次尝试修复
                    if not geometry_simplified.is_valid:
                        repaired = geometry_simplified.buffer(0)
                        if isinstance(repaired, MultiPolygon):
                            if len(repaired.geoms) == 0:
                                return None
                            geometry = max(repaired.geoms, key=lambda p: p.area)
                        else:
                            geometry = repaired
                    else:
                        geometry = geometry_simplified
                except Exception as e2:
                    print(f'几何修复失败（简化后）: {e2}')
                    return None
        
        # 再次检查有效性
        if not geometry.is_valid or geometry.is_empty:
            # 最后尝试：如果仍然无效，尝试更激进的简化
            try:
                coords_range = np.ptp(coords, axis=0)
                aggressive_tolerance = max(coords_range) * 0.01  # 1%的坐标范围
                geometry = geometry.simplify(aggressive_tolerance, preserve_topology=False)
                
                if not geometry.is_valid:
                    geometry = geometry.buffer(0)
                    if isinstance(geometry, MultiPolygon):
                        if len(geometry.geoms) == 0:
                            return None
                        geometry = max(geometry.geoms, key=lambda p: p.area)
            except Exception as e3:
                print(f'最后修复尝试失败: {e3}')
                return None
        
        # 确保返回的是Polygon而不是MultiPolygon
        if isinstance(geometry, MultiPolygon):
            if len(geometry.geoms) == 0:
                return None
            # 如果仍然是MultiPolygon，选择最大的
            geometry = max(geometry.geoms, key=lambda p: p.area)
        
        # 最终有效性检查
        if not isinstance(geometry, Polygon) or not geometry.is_valid or geometry.is_empty:
            return None
        
        # 检查面积是否合理（避免退化情况）
        if geometry.area < 1e-6:  # 面积太小，可能是错误数据
            return None
        
        return geometry
    
    except (ShapelyError, ValueError, TypeError, AttributeError) as e:
        print(f'几何验证失败: {e}')
        return None


def load_dxf(dxf_path: str):  # -> 'ezdxf.drawing.Drawing'
    """
    加载DXF文件
    
    Args:
        dxf_path: DXF文件路径
        
    Returns:
        doc: DXF文档对象
        
    Raises:
        FileNotFoundError: 文件不存在
        ezdxf.DXFStructureError: DXF文件格式错误
    """
    if not os.path.exists(dxf_path):
        raise FileNotFoundError(f'DXF文件不存在: {dxf_path}')
    
    print(f'正在读取DXF文件: {dxf_path}')
    doc = ezdxf.readfile(dxf_path)
    print(f'DXF文件读取成功')
    return doc


def get_dxf_layers(doc) -> set:  # doc: 'ezdxf.drawing.Drawing'
    """
    获取DXF文件中的所有图层名称
    
    Args:
        doc: DXF文档对象
        
    Returns:
        layers: 图层名称集合
    """
    layers = set()
    msp = doc.modelspace()
    for entity in msp:
        layers.add(entity.dxf.layer)
    return layers


def dxf_to_spaceunit_collection(
    dxf_path: str,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> SpaceUnitCollection:
    """
    将DXF文件转换为SpaceUnitCollection对象
    
    处理流程:
    1. 加载DXF文件
    2. 遍历所有实体，提取闭合多边形
    3. 解析图层名称，获取单元类型、业态类别和保护状态
    4. 验证和修复几何数据
    5. 创建SpaceUnitCollection
    
    Args:
        dxf_path: DXF文件路径
        config: 配置字典（预留，当前未使用）
        verbose: 是否打印详细信息
        
    Returns:
        collection: SpaceUnitCollection对象
        
    Examples:
        >>> collection = dxf_to_spaceunit_collection('path/to/file.dxf')
        >>> units = collection.get_all_space_units()
        >>> print(f'共导入 {len(units)} 个空间单元')
    """
    # 加载DXF文件
    doc = load_dxf(dxf_path)
    msp = doc.modelspace()
    
    # 创建SpaceUnitCollection
    collection = SpaceUnitCollection()
    
    # 统计信息
    stats = {
        'total_entities': 0,
        'processed': 0,
        'auto_closed': 0,  # 自动闭合的非闭合polyline数量
        'skipped_non_closed': 0,  # 无法闭合而跳过的数量
        'skipped_invalid_geometry': 0,
        'skipped_unknown_layer': 0,
        'by_unit_type': {},
        'by_business_category': {},
    }
    
    if verbose:
        print('开始解析DXF实体...')
    
    # 遍历所有实体
    for entity in msp:
        stats['total_entities'] += 1
        
        # 只处理多边形（LWPOLYLINE和POLYLINE）
        entity_type = entity.dxftype()
        if entity_type not in ['LWPOLYLINE', 'POLYLINE']:
            continue
        
        # 获取图层名称
        layer_name = entity.dxf.layer
        
        # 解析图层名称
        unit_type, business_category, protected = SpaceUnitCollection.parse_layer_name(layer_name)
        
        # 检查是否是已知的单元类型（排除未知图层）
        if unit_type == 'shop' and business_category == BusinessCategory.UNDEFINED:
            # 检查是否是真正的shop图层格式
            if not layer_name.lower().startswith('shop_'):
                stats['skipped_unknown_layer'] += 1
                if verbose:
                    print(f'跳过未知图层: {layer_name}')
                continue
        
        # 提取坐标点
        try:
            points = _get_entity_points_auto(entity)
            coords = np.array(points)
        except (ValueError, AttributeError) as e:
            if verbose:
                print(f'提取坐标失败 (图层: {layer_name}): {e}')
            stats['skipped_invalid_geometry'] += 1
            continue
        
        # 处理非闭合的polyline：尝试自动闭合
        is_originally_closed = entity.is_closed
        if not is_originally_closed:
            # 检查首尾点距离，如果距离较大，尝试闭合
            if len(coords) >= 3:
                try:
                    # 使用get_closed_polyline自动闭合
                    coords = get_closed_polyline(coords)
                    stats['auto_closed'] += 1
                    if verbose:
                        print(f'自动闭合非闭合polyline (图层: {layer_name})')
                except Exception as e:
                    # 如果闭合失败，跳过该实体
                    stats['skipped_non_closed'] += 1
                    if verbose:
                        print(f'无法闭合polyline (图层: {layer_name}): {e}')
                    continue
            else:
                # 点数太少，无法形成有效多边形
                stats['skipped_non_closed'] += 1
                if verbose:
                    print(f'点数不足，无法闭合 (图层: {layer_name}): 只有 {len(coords)} 个点')
                continue
        
        # 验证和修复几何
        geometry = _validate_and_repair_geometry(coords)
        if geometry is None:
            stats['skipped_invalid_geometry'] += 1
            if verbose:
                print(f'几何验证失败 (图层: {layer_name})')
            continue
        
        # 更新坐标（使用修复后的几何）
        # 确保geometry是Polygon（不是MultiPolygon）
        if isinstance(geometry, MultiPolygon):
            # 如果仍然是MultiPolygon，选择最大的
            geometry = max(geometry.geoms, key=lambda p: p.area)
        
        # 提取exterior坐标
        try:
            coords = np.array(geometry.exterior.coords[:-1])  # 移除重复的最后一个点
        except AttributeError as e:
            if verbose:
                print(f'提取坐标失败 (图层: {layer_name}): {e}, geometry类型: {type(geometry)}')
            stats['skipped_invalid_geometry'] += 1
            continue
        
        # 创建空间单元
        try:
            unit_gdf = SpaceUnitCollection._create_unit_by_coords(
                coords=coords,
                unit_type=unit_type,
                business_type='UNDEFINED',  # 具体业态类型默认为UNDEFINED
                business_category=business_category.value if business_category else None,
                protected=protected,
                replaceable=None,  # 自动推断
                enabled=None  # 自动推断（protected=True时自动为False）
            )
            
            collection.add_space_unit(unit_gdf)
            stats['processed'] += 1
            
            # 更新统计信息
            stats['by_unit_type'][unit_type] = stats['by_unit_type'].get(unit_type, 0) + 1
            if business_category:
                cat_name = business_category.value
                stats['by_business_category'][cat_name] = stats['by_business_category'].get(cat_name, 0) + 1
        
        except Exception as e:
            if verbose:
                print(f'创建空间单元失败 (图层: {layer_name}): {e}')
            stats['skipped_invalid_geometry'] += 1
            continue
    
    # 打印统计信息
    if verbose:
        print('\n=== DXF导入统计 ===')
        print(f'总实体数: {stats["total_entities"]}')
        print(f'成功处理: {stats["processed"]}')
        print(f'自动闭合: {stats["auto_closed"]}')  # 新增：自动闭合的数量
        print(f'跳过（非闭合）: {stats["skipped_non_closed"]}')
        print(f'跳过（无效几何）: {stats["skipped_invalid_geometry"]}')
        print(f'跳过（未知图层）: {stats["skipped_unknown_layer"]}')
        print(f'\n按单元类型统计:')
        for unit_type, count in stats['by_unit_type'].items():
            print(f'  {unit_type}: {count}')
        if stats['by_business_category']:
            print(f'\n按业态类别统计:')
            for category, count in stats['by_business_category'].items():
                print(f'  {category}: {count}')
        print('==================\n')
    
    return collection


def dxf_to_spaceunit_data(
    dxf_path: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    将DXF文件转换为中间数据格式（用于调试或进一步处理）
    
    Args:
        dxf_path: DXF文件路径
        config: 配置字典
        
    Returns:
        data: 中间数据字典
            {
                'version': str,
                'space_units': [
                    {
                        'points': np.ndarray,
                        'unit_type': str,
                        'layer': str,
                        'business_category': BusinessCategory,
                        'protected': bool,
                        ...
                    },
                    ...
                ]
            }
    """
    doc = load_dxf(dxf_path)
    msp = doc.modelspace()
    
    data = {
        'version': '1.0',
        'space_units': []
    }
    
    for entity in msp:
        entity_type = entity.dxftype()
        if entity_type not in ['LWPOLYLINE', 'POLYLINE']:
            continue
        
        layer_name = entity.dxf.layer
        unit_type, business_category, protected = SpaceUnitCollection.parse_layer_name(layer_name)
        
        # 提取坐标点
        try:
            points = _get_entity_points_auto(entity)
            coords = np.array(points)
        except (ValueError, AttributeError):
            continue
        
        # 处理非闭合的polyline：尝试自动闭合
        if not entity.is_closed:
            if len(coords) >= 3:
                try:
                    coords = get_closed_polyline(coords)
                except Exception:
                    continue  # 如果闭合失败，跳过
            else:
                continue  # 点数太少，跳过
        
        # 验证几何
        geometry = _validate_and_repair_geometry(coords)
        if geometry is None:
            continue
        
        coords = np.array(geometry.exterior.coords[:-1])
        
        space_unit_data = {
            'points': coords,
            'unit_type': unit_type,
            'layer': layer_name,
            'business_category': business_category,
            'protected': protected,
        }
        
        data['space_units'].append(space_unit_data)
    
    return data


if __name__ == '__main__':
    # 示例用法
    import sys
    
    if len(sys.argv) < 2:
        print('用法: python dxf_importer.py <dxf_file_path>')
        sys.exit(1)
    
    dxf_path = sys.argv[1]
    
    try:
        # 导入DXF文件
        collection = dxf_to_spaceunit_collection(dxf_path, verbose=True)
        
        # 获取所有空间单元
        all_units = collection.get_all_space_units()
        print(f'\n成功导入 {len(all_units)} 个空间单元')
        
        # 示例：查询不同类型的单元
        shops = collection.get_replaceable_shops()
        print(f'可替换的店铺数量: {len(shops)}')
        
        protected_units = collection.get_protected_units()
        print(f'受保护单元数量: {len(protected_units)}')
        
    except Exception as e:
        print(f'导入失败: {e}')
        import traceback
        traceback.print_exc()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于周边图块功能复杂度计算流量预测

功能：
1. 对每个空间单元，扩大buffer范围
2. 计算与buffer重叠的周边单元
3. 基于周边单元的功能复杂度计算flow_prediction
4. 更新SpaceUnitCollection中的flow_prediction值

复杂度计算公式：
- 功能多样性（Shannon熵）：衡量周边业态类型的多样性
- 功能权重：不同业态类型对吸引力的贡献权重（加权功能总和）
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import geopandas as gpd
from collections import Counter
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from env.geo.space_unit import SpaceUnitCollection
from env.geo.business_type import BusinessCategory


def compute_shannon_entropy(categories: List[str]) -> float:
    """
    计算Shannon熵（功能多样性指标）
    
    Args:
        categories: 业态类别列表
        
    Returns:
        entropy: Shannon熵值（0-1之间，归一化）
    """
    if len(categories) == 0:
        return 0.0
    
    # 统计各类别数量
    counter = Counter(categories)
    total = len(categories)
    
    # 计算Shannon熵
    entropy = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    
    # 归一化到[0, 1]：除以最大可能熵值（log2(类别数)）
    max_entropy = np.log2(len(counter)) if len(counter) > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    return normalized_entropy


def get_business_category_weight(category: Optional[str]) -> float:
    """
    获取业态类别的吸引力权重
    
    Args:
        category: 业态类别字符串（如'dining', 'retail'等）
        
    Returns:
        weight: 权重值（0-1之间）
    """
    if category is None:
        return 0.0
    
    # 定义不同业态类别的吸引力权重
    category_weights = {
        'dining': 1.0,              # 餐饮：最高吸引力
        'retail': 0.9,              # 零售：高吸引力
        'leisure': 0.8,              # 休闲娱乐：较高吸引力
        'cultural': 0.7,             # 文化体验：中等吸引力
        'service': 0.6,              # 服务：中等吸引力
        'residential': 0.4,          # 居住：较低吸引力
        'supporting': 0.3,           # 配套设施：低吸引力
        'undefined': 0.2,            # 未定义：最低吸引力
    }
    
    return category_weights.get(category.lower(), 0.2)


def compute_surrounding_complexity(
    target_unit: gpd.GeoSeries,
    all_units: gpd.GeoDataFrame,
    buffer_distance: float,
    exclude_self: bool = True
) -> Dict[str, float]:
    """
    计算目标单元周边图块的功能复杂度
    
    Args:
        target_unit: 目标空间单元（GeoSeries）
        all_units: 所有空间单元的GeoDataFrame
        buffer_distance: buffer距离（米）
        exclude_self: 是否排除自身
        
    Returns:
        complexity_metrics: 复杂度指标字典
            - diversity: 功能多样性（Shannon熵）
            - weighted_sum: 加权功能总和
            - total_area: 周边单元总面积
            - count: 周边单元数量
    """
    # 创建buffer
    target_geometry = target_unit['geometry']
    if target_geometry is None or target_geometry.is_empty:
        return {
            'diversity': 0.0,
            'weighted_sum': 0.0,
            'total_area': 0.0,
            'count': 0
        }
    
    buffer_geom = target_geometry.buffer(buffer_distance)
    
    # 找到与buffer重叠的单元
    overlapping_units = all_units[all_units.geometry.intersects(buffer_geom)].copy()
    
    # 排除自身
    if exclude_self and 'uid' in target_unit.index:
        target_uid = target_unit['uid']
        overlapping_units = overlapping_units[overlapping_units['uid'] != target_uid]
    
    if len(overlapping_units) == 0:
        return {
            'diversity': 0.0,
            'weighted_sum': 0.0,
            'total_area': 0.0,
            'count': 0
        }
    
    # 提取周边单元的业态类别
    categories = []
    areas = []
    weights = []
    
    for idx, unit in overlapping_units.iterrows():
        unit_type = unit.get('unit_type', '')
        business_category = unit.get('business_category', None)
        
        # 只考虑shop类型的单元
        if unit_type == 'shop':
            if business_category is None:
                cat_str = 'undefined'
            elif isinstance(business_category, str):
                cat_str = business_category.lower()
            elif hasattr(business_category, 'value'):
                cat_str = business_category.value.lower()
            else:
                cat_str = str(business_category).lower()
            
            categories.append(cat_str)
            area = unit.get('area', 0.0)
            areas.append(area)
            weights.append(get_business_category_weight(cat_str))
    
    if len(categories) == 0:
        return {
            'diversity': 0.0,
            'weighted_sum': 0.0,
            'total_area': 0.0,
            'count': 0
        }
    
    # 计算功能多样性（Shannon熵）
    diversity = compute_shannon_entropy(categories)
    
    # 计算加权功能总和（面积加权）
    total_area = sum(areas)
    weighted_sum = sum(w * a for w, a in zip(weights, areas)) / total_area if total_area > 0 else 0.0
    
    return {
        'diversity': diversity,
        'weighted_sum': weighted_sum,
        'total_area': total_area,
        'count': len(categories)
    }


def compute_flow_from_complexity(
    collection: SpaceUnitCollection,
    buffer_distance: float = 10.0,
    base_flow: float = 0.0,
    diversity_weight: float = 0.5,
    weighted_sum_weight: float = 0.5,
    normalize: bool = True
) -> SpaceUnitCollection:
    """
    基于周边图块功能复杂度计算flow_prediction
    只针对public_space类型的单元（walkable_place）
    
    Args:
        collection: SpaceUnitCollection对象
        buffer_distance: buffer距离（米），默认10米
        base_flow: 基础流量值，默认0.0
        diversity_weight: 多样性权重，默认0.5
        weighted_sum_weight: 加权总和权重，默认0.5
        normalize: 是否归一化到[0, 1]范围
        
    Returns:
        collection: 更新后的SpaceUnitCollection（原地修改）
    """
    # 直接访问内部 GeoDataFrame，确保修改生效
    all_units = collection._SpaceUnitCollection__unit_gdf
    
    if len(all_units) == 0:
        print("警告: 没有空间单元")
        return collection
    
    # 首先确保所有shop单元的flow_prediction都为0（无论之前的状态如何）
    shop_mask = all_units['unit_type'] == 'shop'
    if shop_mask.any():
        all_units.loc[shop_mask, 'flow_prediction'] = 0.0
    
    # 只处理public_space类型的单元
    public_space_units = all_units[all_units['unit_type'] == 'public_space']
    
    if len(public_space_units) == 0:
        print("警告: 没有public_space类型的单元")
        return collection
    
    # 计算每个public_space单元的复杂度
    complexity_scores = []
    public_space_indices = []
    
    for idx, unit in public_space_units.iterrows():
        complexity = compute_surrounding_complexity(
            target_unit=unit,
            all_units=all_units,
            buffer_distance=buffer_distance,
            exclude_self=True
        )
        
        # 计算综合复杂度分数（只使用diversity和weighted_sum）
        score = (
            diversity_weight * complexity['diversity'] +
            weighted_sum_weight * complexity['weighted_sum']
        )
        
        complexity_scores.append(score)
        public_space_indices.append(idx)
    
    # 归一化到[0, 1]范围（可选）
    if normalize and len(complexity_scores) > 0:
        scores_array = np.array(complexity_scores)
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        if max_score > min_score:
            complexity_scores = (scores_array - min_score) / (max_score - min_score)
        else:
            complexity_scores = np.zeros_like(scores_array)
    
    # 只更新public_space单元的flow_prediction
    for idx, score in zip(public_space_indices, complexity_scores):
        all_units.loc[idx, 'flow_prediction'] = base_flow + float(score)
    
    # 再次确保所有shop单元的flow_prediction都为0（双重保险）
    # 使用 mask 直接修改，确保修改生效
    shop_mask = all_units['unit_type'] == 'shop'
    if shop_mask.any():
        all_units.loc[shop_mask, 'flow_prediction'] = 0.0
    
    # 验证：确保所有shop单元的flow_prediction都为0
    shop_flow_values = all_units.loc[shop_mask, 'flow_prediction'] if shop_mask.any() else []
    if len(shop_flow_values) > 0 and not (shop_flow_values == 0.0).all():
        print(f"警告: 仍有 {len(shop_flow_values[shop_flow_values != 0.0])} 个shop单元的flow_prediction不为0")
        # 强制设置为0
        all_units.loc[shop_mask, 'flow_prediction'] = 0.0
    
    return collection


def main():
    parser = argparse.ArgumentParser(description='基于周边图块功能复杂度计算流量预测')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='输入的GeoJSON文件路径')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='输出的GeoJSON文件路径')
    parser.add_argument('--buffer', type=float, default=10.0,
                       help='Buffer距离（米），默认: 10.0')
    parser.add_argument('--base-flow', type=float, default=0.0,
                       help='基础流量值，默认: 0.0')
    parser.add_argument('--diversity-weight', type=float, default=0.5,
                       help='多样性权重，默认: 0.5')
    parser.add_argument('--weighted-sum-weight', type=float, default=0.5,
                       help='加权总和权重，默认: 0.5')
    parser.add_argument('--no-normalize', action='store_true',
                       help='不归一化复杂度分数')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("基于周边图块功能复杂度计算流量预测")
    print("=" * 60)
    
    # 1. 读取GeoJSON文件
    print(f"\n[1/3] 读取GeoJSON文件: {args.input}")
    try:
        gdf = gpd.read_file(args.input)
        print(f"  成功读取 {len(gdf)} 个要素")
    except Exception as e:
        print(f"  错误: {e}")
        return 1
    
    # 2. 转换为SpaceUnitCollection
    print(f"\n[2/3] 转换为SpaceUnitCollection并计算复杂度")
    try:
        from scripts.geojson_to_raster import geojson_to_spaceunit_collection
        
        collection = geojson_to_spaceunit_collection(gdf)
        all_units = collection.get_all_space_units()
        print(f"  成功转换 {len(all_units)} 个空间单元")
        
        # 计算复杂度并更新flow_prediction
        collection = compute_flow_from_complexity(
            collection=collection,
            buffer_distance=args.buffer,
            base_flow=args.base_flow,
            diversity_weight=args.diversity_weight,
            weighted_sum_weight=args.weighted_sum_weight,
            normalize=not args.no_normalize
        )
        
        # 显示统计信息
        updated_units = collection.get_all_space_units()
        flow_values = updated_units['flow_prediction'].values
        print(f"\n  流量预测统计:")
        print(f"    最小值: {flow_values.min():.4f}")
        print(f"    最大值: {flow_values.max():.4f}")
        print(f"    平均值: {flow_values.mean():.4f}")
        print(f"    非零数量: {(flow_values > 0).sum()} / {len(flow_values)}")
        
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 3. 保存结果
    print(f"\n[3/3] 保存结果到: {args.output}")
    try:
        updated_gdf = collection.get_all_space_units()
        updated_gdf.to_file(args.output, driver='GeoJSON')
        print(f"  成功保存 {len(updated_gdf)} 个要素")
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n完成！")
    return 0


if __name__ == '__main__':
    sys.exit(main())

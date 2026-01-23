#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GeoJSON转栅格通道脚本

功能：
1. 从GeoJSON文件导入空间单元数据
2. 转换为SpaceUnitCollection
3. 创建WorldState
4. 转换为栅格观测通道（保持原始尺寸比例）
5. 保存栅格图像和可视化
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import geopandas as gpd
import yaml

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from env.geo.space_unit import SpaceUnitCollection
from env.world_state import WorldState
from env.geo.street_network import StreetNetworkCollection
from env.geo.business_type import BusinessTypeCollection
from env.representation.raster_obs import RasterObservation
from env.representation.visualization import visualize_raster_channels


def geojson_to_spaceunit_collection(gdf: gpd.GeoDataFrame) -> SpaceUnitCollection:
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
                # 无论GeoJSON中是否有值，都强制设为0
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




def main():
    parser = argparse.ArgumentParser(description='GeoJSON转栅格通道')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='输入的GeoJSON文件路径')
    parser.add_argument('--config', '-c', type=str, default='configs/env.yaml',
                       help='配置文件路径（默认: configs/env.yaml）')
    parser.add_argument('--output-dir', '-o', type=str, default='output',
                       help='输出目录（默认: output）')
    parser.add_argument('--target-pixels', type=int, default=256,
                       help='目标像素数（用于长边，默认: 256）')
    parser.add_argument('--min-resolution', type=int, default=128,
                       help='最小分辨率（默认: 128）')
    parser.add_argument('--max-resolution', type=int, default=1024,
                       help='最大分辨率（默认: 1024）')
    parser.add_argument('--save-numpy', action='store_true',
                       help='保存numpy数组')
    parser.add_argument('--no-visualize', action='store_true',
                       help='不显示可视化图像')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("GeoJSON转栅格通道")
    print("=" * 60)
    
    # 1. 读取GeoJSON文件
    print(f"\n[1/5] 读取GeoJSON文件: {args.input}")
    try:
        gdf = gpd.read_file(args.input)
        print(f"  成功读取 {len(gdf)} 个要素")
        print(f"  列名: {list(gdf.columns)}")
    except Exception as e:
        print(f"  错误: {e}")
        return 1
    
    # 2. 转换为SpaceUnitCollection
    print(f"\n[2/5] 转换为SpaceUnitCollection")
    try:
        collection = geojson_to_spaceunit_collection(gdf)
        all_units = collection.get_all_space_units()
        print(f"  成功转换 {len(all_units)} 个空间单元")
        
        if len(all_units) > 0:
            print(f"  总面积: {all_units['area'].sum():.2f} 平方米")
            print(f"  按单元类型统计:")
            unit_type_counts = all_units['unit_type'].value_counts()
            for unit_type, count in unit_type_counts.items():
                print(f"    {unit_type}: {count} 个")
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    if len(all_units) == 0:
        print("  警告: 没有成功转换的空间单元")
        return 1
    
    # 3. 创建WorldState
    print(f"\n[3/5] 创建WorldState")
    try:
        # 读取配置文件
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建WorldState
        street_network = StreetNetworkCollection()
        business_types = BusinessTypeCollection()
        
        world_state = WorldState(
            space_units=collection,
            street_network=street_network,
            business_types=business_types,
            graph=None,
            budget=config['env']['constraints']['max_budget'],
            constraints=config['env']['constraints'],
            step_idx=0,
            episode_id=None
        )
        print(f"  WorldState创建成功")
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 4. 创建栅格观测编码器（自动计算分辨率）
    print(f"\n[4/5] 创建栅格观测编码器")
    try:
        channels = config['env']['representation']['raster']['channels']
        # 使用类方法自动计算分辨率
        raster_obs = RasterObservation.create_with_auto_resolution(
            state=world_state,
            channels=channels,
            target_pixels=args.target_pixels,
            min_resolution=args.min_resolution,
            max_resolution=args.max_resolution
        )
        
        print(f"  分辨率: {raster_obs.resolution} (保持宽高比)")
        print(f"  通道数: {len(channels)}")
        print(f"  通道列表: {channels}")
        print(f"  观测形状: {raster_obs.get_obs_shape()}")  # (C, H, W)
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 5. 生成栅格观测
    print(f"\n[5/5] 生成栅格观测")
    try:
        obs = raster_obs.encode(world_state)
        
        print(f"  栅格观测形状: {obs.shape}")  # (C, H, W)
        print(f"  数据类型: {obs.dtype}")
        print(f"  数值范围: [{obs.min():.3f}, {obs.max():.3f}]")
        
        # 显示各通道统计信息
        print(f"\n  各通道统计信息:")
        for i, channel_name in enumerate(channels):
            channel_data = obs[i]
            print(f"    {channel_name}:")
            print(f"      最小值: {channel_data.min():.4f}")
            print(f"      最大值: {channel_data.max():.4f}")
            print(f"      平均值: {channel_data.mean():.4f}")
            print(f"      非零像素数: {(channel_data > 0).sum()} / {channel_data.size}")
        
        # 保存numpy数组
        if args.save_numpy:
            np_path = output_dir / 'raster_observation.npy'
            np.save(np_path, obs)
            print(f"\n  NumPy数组已保存到: {np_path}")
        
        # 可视化
        if not args.no_visualize:
            # 可视化各通道
            vis_path = output_dir / 'raster_channels.png'
            visualize_raster_channels(obs, channels, str(vis_path))
        
        print(f"\n完成！输出目录: {output_dir}")
        
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

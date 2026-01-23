# 脚本说明

## geojson_to_raster.py

GeoJSON转栅格通道脚本，将GeoJSON文件转换为栅格观测通道。

### 功能特点

- ✅ 从GeoJSON文件导入空间单元数据
- ✅ 自动保持原始尺寸比例（不强制256x256）
- ✅ 转换为SpaceUnitCollection和WorldState
- ✅ 生成3个栅格通道（walkable_mask, predicted_flow, landuse_id）
- ✅ 保存可视化图像和numpy数组

### 使用方法

```bash
# 基本用法
python scripts/geojson_to_raster.py -i data/0123.geojson

# 指定输出目录
python scripts/geojson_to_raster.py -i data/0123.geojson -o output/raster

# 自定义分辨率参数
python scripts/geojson_to_raster.py -i data/0123.geojson --target-pixels 512 --min-resolution 256 --max-resolution 2048

# 保存numpy数组
python scripts/geojson_to_raster.py -i data/0123.geojson --save-numpy

# 不显示可视化（仅保存文件）
python scripts/geojson_to_raster.py -i data/0123.geojson --no-visualize
```

### 参数说明

- `--input, -i`: 输入的GeoJSON文件路径（必需）
- `--config, -c`: 配置文件路径（默认: configs/env.yaml）
- `--output-dir, -o`: 输出目录（默认: output）
- `--target-pixels`: 目标像素数，用于长边（默认: 256）
- `--min-resolution`: 最小分辨率（默认: 128）
- `--max-resolution`: 最大分辨率（默认: 1024）
- `--save-numpy`: 保存numpy数组到文件
- `--no-visualize`: 不显示可视化图像（仅保存文件）

### 分辨率计算

脚本会自动根据数据的实际边界框计算保持宽高比的分辨率：

- 长边使用 `target_pixels` 像素
- 短边按比例缩放
- 分辨率限制在 `min_resolution` 和 `max_resolution` 之间

例如：
- 如果数据是 1000m x 500m（宽高比 2:1）
- `target_pixels=256` 时，分辨率将是 `[128, 256]`（H x W）

### GeoJSON文件要求

GeoJSON文件应包含以下字段：

**必需字段：**
- `geometry`: 几何对象（Polygon或MultiPolygon）

**可选字段（会使用默认值）：**
- `unit_type`: 单元类型（'shop', 'atrium', 'public_space', 'circulation'），默认: 'shop'
- `business_category`: 业态类别（字符串，如 'dining', 'retail'等），默认: None
- `protected`: 是否受保护（布尔值），默认: False
- `enabled`: 是否启用（布尔值），默认: True
- `replaceable`: 是否可替换（布尔值），默认: None（自动推断）
- `flow_prediction`: 预测流量（浮点数），默认: 0.0

### 输出文件

脚本会在输出目录生成以下文件：

- `raster_channels.png`: 各通道的可视化图像
- `raster_observation.npy`: NumPy数组（如果使用 `--save-numpy`）

### 示例

```bash
# 处理GeoJSON文件并保存结果
python scripts/geojson_to_raster.py \
    --input data/0123.geojson \
    --output-dir output/0123 \
    --target-pixels 512 \
    --save-numpy
```

输出：
```
============================================================
GeoJSON转栅格通道
============================================================

[1/5] 读取GeoJSON文件: data/0123.geojson
  成功读取 50 个要素

[2/5] 转换为SpaceUnitCollection
  成功转换 50 个空间单元
  总面积: 12500.00 平方米

[3/5] 创建WorldState
  WorldState创建成功

[4/5] 创建栅格观测编码器
  分辨率: [256, 512] (保持宽高比)
  通道数: 3

[5/5] 生成栅格观测
  栅格观测形状: (3, 256, 512)
  完成！输出目录: output/0123
```

---

## compute_flow_from_complexity.py

基于周边图块功能复杂度计算流量预测脚本。

### 功能特点

- ✅ 对每个空间单元扩大buffer范围
- ✅ 计算与buffer重叠的周边单元
- ✅ 基于周边单元的功能复杂度计算flow_prediction
- ✅ 支持自定义权重和参数

### 复杂度计算公式

流量预测 = 基础流量 + 综合复杂度分数

综合复杂度分数 = 
- `diversity_weight` × 功能多样性（Shannon熵）
- + `density_weight` × 功能密度（周边单元数量/面积）
- + `weighted_sum_weight` × 加权功能总和（面积加权）

### 使用方法

```bash
# 基本用法
python scripts/compute_flow_from_complexity.py \
    -i data/0123.geojson \
    -o data/0123_with_flow.geojson

# 自定义buffer距离和权重
python scripts/compute_flow_from_complexity.py \
    -i data/0123.geojson \
    -o data/0123_with_flow.geojson \
    --buffer 100.0 \
    --diversity-weight 0.5 \
    --density-weight 0.3 \
    --weighted-sum-weight 0.2

# 不归一化复杂度分数
python scripts/compute_flow_from_complexity.py \
    -i data/0123.geojson \
    -o data/0123_with_flow.geojson \
    --no-normalize
```

### 参数说明

- `--input, -i`: 输入的GeoJSON文件路径（必需）
- `--output, -o`: 输出的GeoJSON文件路径（必需）
- `--buffer`: Buffer距离（米），默认: 50.0
- `--base-flow`: 基础流量值，默认: 0.0
- `--diversity-weight`: 多样性权重，默认: 0.4
- `--density-weight`: 密度权重，默认: 0.3
- `--weighted-sum-weight`: 加权总和权重，默认: 0.3
- `--no-normalize`: 不归一化复杂度分数到[0, 1]范围

### 复杂度指标说明

1. **功能多样性（Diversity）**
   - 使用Shannon熵衡量周边业态类型的多样性
   - 值范围: [0, 1]，1表示完全混合（所有业态类型均匀分布）

2. **功能密度（Density）**
   - 周边单元数量 / buffer面积
   - 衡量单位面积内的功能单元数量

3. **加权功能总和（Weighted Sum）**
   - 基于不同业态类别的吸引力权重
   - 权重定义：
     - `dining`: 1.0（最高）
     - `retail`: 0.9
     - `leisure`: 0.8
     - `cultural`: 0.7
     - `service`: 0.6
     - `residential`: 0.4
     - `supporting`: 0.3
     - `undefined`: 0.2（最低）

### 示例输出

```bash
============================================================
基于周边图块功能复杂度计算流量预测
============================================================

[1/3] 读取GeoJSON文件: data/0123.geojson
  成功读取 84 个要素

[2/3] 转换为SpaceUnitCollection并计算复杂度
  成功转换 84 个空间单元
  
  流量预测统计:
    最小值: 0.0000
    最大值: 1.0000
    平均值: 0.4523
    非零数量: 55 / 84

[3/3] 保存结果到: data/0123_with_flow.geojson
  成功保存 84 个要素

完成！
```

### 注意事项

1. **Buffer距离选择**
   - 较小的buffer（20-50米）：关注紧邻区域
   - 较大的buffer（50-100米）：考虑更广泛的周边影响

2. **权重调整**
   - 如果希望更强调多样性，增加 `diversity_weight`
   - 如果希望更强调密度，增加 `density_weight`
   - 权重总和建议为1.0（脚本不强制）

3. **归一化**
   - 默认归一化到[0, 1]范围，便于后续处理
   - 使用 `--no-normalize` 可以保留原始分数值

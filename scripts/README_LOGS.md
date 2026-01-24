# 训练日志和Observation查看工具使用说明

本文档说明如何查看训练日志和渲染每个Episode的Observation。

## 1. 查看训练日志

### 使用 `view_training_logs.py`

这个工具可以查看stable-baselines3生成的训练日志（monitor.csv）。

#### 基本用法

```bash
# 自动查找并显示日志
python scripts/view_training_logs.py

# 指定日志目录
python scripts/view_training_logs.py --log-dir ./logs

# 直接指定monitor.csv文件路径
python scripts/view_training_logs.py --monitor-file ./logs/monitor.csv

# 只显示最近10个episode
python scripts/view_training_logs.py --n-episodes 10

# 绘制训练曲线
python scripts/view_training_logs.py --plot

# 绘制训练曲线并保存到文件
python scripts/view_training_logs.py --plot --plot-output ./logs/training_curve.png
```

#### 输出说明

工具会显示：
- **Episode摘要统计**: 每个episode的累计奖励、长度、时间
- **总体统计**: 平均奖励、标准差、最大/最小奖励、平均episode长度等
- **最近N个Episode统计**: 最近episode的表现（用于判断训练是否收敛）

#### 日志文件位置

训练时，monitor.csv会保存在配置文件中指定的`log_dir`目录（默认为`./logs`）。

## 2. 渲染Episode的Observation

### 使用 `render_episode_observations.py`

这个工具可以加载训练好的模型，运行episode，并将每个step的observation渲染为图像。

#### 基本用法

```bash
# 渲染5个episode，每步都保存observation
python scripts/render_episode_observations.py \
    --checkpoint ./checkpoints/ppo_final.zip \
    --config configs/train.yaml \
    --output-dir ./logs/episode_observations

# 只渲染3个episode，每5步保存一次
python scripts/render_episode_observations.py \
    --checkpoint ./checkpoints/ppo_final.zip \
    --n-episodes 3 \
    --save-every-n-steps 5

# 只渲染RGB合成图，不渲染各通道
python scripts/render_episode_observations.py \
    --checkpoint ./checkpoints/ppo_final.zip \
    --no-channels

# 只渲染各通道，不渲染RGB合成图
python scripts/render_episode_observations.py \
    --checkpoint ./checkpoints/ppo_final.zip \
    --no-rgb
```

#### 参数说明

- `--checkpoint`: **必需**，模型检查点路径（.zip文件）
- `--config`: 配置文件路径（默认: `configs/train.yaml`）
- `--output-dir`: 输出目录（默认: `./logs/episode_observations`）
- `--n-episodes`: 要渲染的episode数量（默认: 5）
- `--save-every-n-steps`: 每N步保存一次observation（默认: 1，即每步都保存）
- `--no-channels`: 不渲染各通道图像
- `--no-rgb`: 不渲染RGB合成图

#### 输出文件结构

```
logs/episode_observations/
├── episode_0001/
│   ├── step_000_initial_channels.png    # 初始状态各通道
│   ├── step_000_initial_rgb.png        # 初始状态RGB合成
│   ├── step_001_channels.png           # 第1步各通道
│   ├── step_001_rgb.png                # 第1步RGB合成
│   ├── step_002_channels.png
│   ├── step_002_rgb.png
│   └── ...
├── episode_0002/
│   └── ...
└── ...
```

#### Observation通道说明

根据配置，observation包含以下通道：

1. **walkable_mask** (可走区域): 白色表示可走区域（circulation和public_space）
2. **predicted_flow** (预测流量): 热力图，显示预测的人流量（归一化到0-1）
3. **landuse_id** (土地利用): 8个灰度类别，表示不同的业态类型

## 3. 使用main.py的test模式

你也可以使用main.py的test模式来快速测试模型：

```bash
# 运行单个episode并输出详细日志
python main.py --mode test --checkpoint ./checkpoints/ppo_final.zip

# 评估模式：运行多个episode并统计性能
python main.py --mode eval --checkpoint ./checkpoints/ppo_final.zip
```

## 4. 完整工作流程示例

### 训练模型

```bash
python main.py --mode train --config configs/train.yaml
```

训练完成后，日志会保存在`./logs/monitor.csv`。

### 查看训练日志

```bash
# 查看所有episode统计
python scripts/view_training_logs.py

# 绘制训练曲线
python scripts/view_training_logs.py --plot
```

### 渲染Observation

```bash
# 渲染最近训练的模型的observation
python scripts/render_episode_observations.py \
    --checkpoint ./checkpoints/ppo_final.zip \
    --n-episodes 5 \
    --output-dir ./logs/episode_observations
```

## 5. 常见问题

### Q: 找不到monitor.csv文件

A: 检查以下位置：
- `./logs/monitor.csv`
- 训练时指定的`log_dir`目录
- 使用`--monitor-file`参数直接指定文件路径

### Q: 渲染的observation图像是黑色的

A: 可能的原因：
- 初始状态为空（没有空间单元）
- 检查GeoJSON文件路径是否正确（`configs/env.yaml`中的`initial_state.geojson_path`）

### Q: 模型加载失败

A: 确保：
- 检查点文件路径正确
- 检查点文件完整（没有被截断）
- 使用与训练时相同的配置文件

## 6. 高级用法

### 批量渲染多个检查点

```bash
# 渲染多个检查点
for checkpoint in ./checkpoints/ppo_model_*.zip; do
    python scripts/render_episode_observations.py \
        --checkpoint "$checkpoint" \
        --output-dir "./logs/obs_$(basename $checkpoint .zip)" \
        --n-episodes 3
done
```

### 只查看特定episode范围的日志

```bash
# 使用pandas直接处理CSV
python -c "
import pandas as pd
df = pd.read_csv('./logs/monitor.csv', skiprows=1)
print(df[df.index >= 50][['r', 'l']].describe())  # 查看第50个episode之后的数据
"
```

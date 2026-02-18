# 时间片整图节点回归实验

同一张静态图（101 节点，340 无向边）在不同时间片 t 有不同的节点标签 y_t（log1p(flow)）。每个时间片作为一个 PyG Data 样本，数据集大小 = 42。

## 运行

```bash
cd timegraph_regression_test
python train_timegraph.py --epochs 2000 --lr 1e-3 --kfold_by day
```

可选参数：
- `--epochs`：最大训练轮数（默认 2000）
- `--lr`：学习率（默认 1e-3）
- `--weight_decay`：权重衰减（默认 1e-3）
- `--patience`：早停耐心值（默认 50）
- `--batch_size`：批大小（默认 2）
- `--graph_pt`：静态图缓存路径（默认 `graph_cache_static.pt`）
- `--y_pt`：时间片标签路径（默认 `y_time.pt`）
- `--mask_pt`：时间片 mask 路径（默认 `mask_time.pt`）

## 输入 pt 格式

将以下文件放在 `timegraph_regression_test/` 目录下（或通过 `--graph_pt` 等指定路径）：

### graph_cache_static.pt

```python
{
    "x_cont": torch.Tensor,      # [101, 11] float，标准化后连续特征
    "func_type": torch.Tensor,   # [101] long，功能类型，取值 0 或 1
    "edge_index": torch.Tensor,  # [2, 680] long，无向边（340 条 × 2 方向）
}
```

### y_time.pt

- 形状：`[42, 101]` float
- 含义：每个时间片的节点标签，log1p(flow)

### mask_time.pt

- 形状：`[42, 101]` bool
- 含义：每个时间片的监督 mask；`mask_time[t, n]=True` 表示节点 n 在时间片 t 有观测，loss 仅在这些节点上计算

## 数据缺失时

若上述 pt 文件不存在，脚本会自动生成 mock 数据并保存，便于直接运行测试。

## 评估

- 7-fold by day：按天划分，留一天（6 个时间片）作 test，其余 6 天（36 个时间片）作 train
- 指标：test MAE(log)，即 log1p(flow) 空间的 MAE
- 报告：每折 test MAE，以及 GCN / MLP 的 mean±std

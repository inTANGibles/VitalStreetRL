# 子图数据集节点回归实验

以**每个节点**为中心抽取 k-hop ego-subgraph，形成 101 个子图样本。仅 20 个中心节点有客流观测，采用 **masked supervision**：仅对有标签样本计算回归损失，其余样本仅用于前向推断。

## 运行方式

```bash
python train_subgraph.py --k 2 --epochs 2000
```

可选参数：

- `--use_huber 1`：使用 HuberLoss（默认）；`0` 使用 L1Loss
- `--lr 1e-3`：学习率
- `--weight_decay 1e-3`：权重衰减
- `--patience 50`：早停耐心
- `--batch_size 8`：训练批大小
- `--pt_path graph_cache.pt`：图缓存路径

## 数据约定

`graph_cache.pt` 包含：

- `x_cont` [101, 11] float：连续特征（已标准化）
- `func_type` [101] long：类别特征（num_func_types=2）
- `y` [101] float：log1p(flow)
- `mask` [101] bool：监督 mask（20 个 True）
- `edge_index` [2, E*2] long：无向边（双向存）
- `labeled_node_ids` [20] long：有标签节点索引

子图：101 个（每节点为中心），k-hop 默认 2。

若 `graph_cache.pt` 不存在，将自动生成 mock 数据并保存。

### 从 CSV 生成 graph_cache.pt

```bash
python data_io.py --nodes ../data/FlowData/Jul/nodes.csv --edges ../data/FlowData/Jul/edges.csv --flow ../data/FlowData/Jul/flow.csv --output graph_cache.pt
```

或训练时直接指定 CSV 路径（会自动生成并保存 pt）：

```bash
python train_subgraph.py --nodes ../data/FlowData/Jul/nodes.csv --edges ../data/FlowData/Jul/edges.csv --flow ../data/FlowData/Jul/flow.csv --pt_path graph_cache.pt
```

## Optuna 超参数搜索

**方式一：Notebook**（`train_visualization.ipynb` 第 10 节）  
在 notebook 中设置 `OPTUNA_MODEL`、`N_TRIALS` 后运行即可。

**方式二：命令行**

```bash
pip install optuna
python train_optuna.py --model gcn --n_trials 30 --epochs 500
```

可选参数：

- `--model gcn|mlp`：搜索 GCN 或 MLP
- `--n_trials 30`：trial 数量
- `--epochs 500`：每折最大训练轮数
- `--storage optuna.db`：SQLite 持久化，支持断点续跑

搜索空间：`lr`、`weight_decay`、`hidden`、`emb_dim`、`patience`、`batch_size`。

## 输出

- GCN 5-fold CV：MAE(log) mean±std
- MLP baseline 5-fold CV：MAE(log) mean±std

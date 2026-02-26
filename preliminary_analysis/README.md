# 前期数据分析

本文件夹用于项目的前期数据探索与统计分析。

## 内容

### `business_flow_correlation.ipynb` — 业态与客流相关性分析

**目标**：探究不同业态（node_type）与客流（flow）之间是否存在显著相关关系。

**数据来源**：
- `streetflow_hetero_time_gnn/data_demo/nodes.csv`：节点与业态
- `streetflow_hetero_time_gnn/data_demo/flows.csv`：客流记录
- `streetflow_hetero_time_gnn/data_demo/edges.csv`：邻接关系

**分析方法**：
1. 客流构造：公共空间使用直接观测客流；商铺使用邻接公共空间客流聚合作为代理
2. Kruskal-Wallis 非参数多组差异检验
3. η² 效应量
4. Mann-Whitney U 事后两两比较
5. 箱线图、小提琴图、条形图

**运行方式**：在 Jupyter 或 VS Code 中打开该 notebook，依次运行各单元格即可。

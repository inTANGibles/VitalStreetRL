# Flow Mapping

本目录用于客流映射（flow mapping）相关分析与可视化。

## 内容

- **flow_mapping.ipynb**：Public Space 可视化与分析、客流与空间单元映射等（由根目录 `notebooks/` 移入）。
- **flow_visualize.py**：精简脚本。**输入** flow 文件路径，**输出**原始客流与还原客流两张 choropleth 图（可保存 PNG）。在 notebook 中先 `load_context(geojson, devices_axis)` 一次，再对多天 flow 循环调用 `compute_flow_and_plot(ctx, flow_path, output_dir=...)` 即可做一周可视化。
- **visualize_week_flow.ipynb**：示例：用 `flow_visualize` 对一周的 flow 文件批量生成原始/还原客流图。

## 与兄弟目录关系

与以下两个文件夹并列，同属 VitalStreetRL 子模块：

- `streetflow_hetero_time_gnn` — 时空图与客流预测
- `vitalstreet_ga_optimization` — 遗传算法多目标优化

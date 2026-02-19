# VitalStreet GA 多目标优化

基于 **NSGA-II** 的街道活力多目标优化，不依赖原 RL 模块。动作仅两类：`CHANGE_BUSINESS`、`SHOP_TO_PUBLIC_SPACE`。

## 目标（最小化）

- **f1** = -final_vitality（最大化活力）
- **f2** = violation_total（最小化约束恶化）
- **f3** = cost_proxy = alpha×n_shop_to_public + beta×n_change_business

## 目录结构

```
vitalstreet_ga_optimization/
├── configs/
│   ├── ga.yaml      # GA 配置（含 cost_alpha / cost_beta）
│   └── reward.yaml
├── src/vitalga/
│   ├── state.py           # WorldState, SpaceUnitCollection
│   ├── action_space.py    # ActionType(0,1), 编码/解码/合法动作
│   ├── transition.py      # Transition.step（含 flow 更新）
│   ├── flow_from_complexity.py
│   ├── objective/
│   │   ├── reward.py      # _compute_vitality, reward
│   │   └── vitality_metrics.py
│   ├── evaluator.py      # 执行动作序列、输出指标
│   ├── encoding.py       # 个体编码与合法初始化/交叉/变异
│   └── ga_nsga2.py       # pymoo NSGA-II
├── scripts/
│   └── 01_run_nsga2.py   # 命令行运行
├── notebooks/
│   └── 01_pareto_visualize.ipynb
├── tests/
│   └── test_smoke.py
├── outputs/              # pareto.json, plots/
├── requirements.txt
└── README.md
```

## 运行

```bash
cd vitalstreet_ga_optimization
pip install -r requirements.txt
python scripts/01_run_nsga2.py --toy
```

输出：`outputs/pareto.json`、`outputs/plots/pareto_2d.png`。

使用真实数据时提供 GeoJSON：

```bash
python scripts/01_run_nsga2.py --geojson /path/to/initial.geojson --pop 50 --gen 30
```

## 配置

`configs/ga.yaml` 必须包含：

- `cost_alpha`、`cost_beta`：cost_proxy 权重
- `max_steps`：单条序列最大步数
- `reward.mu_violation`、`transition.flow`、`action_space` 等

## 验收

- 安装依赖后 `python scripts/01_run_nsga2.py --toy` 可跑通，生成 `outputs/pareto.json` 与 `outputs/plots/*.png`（需安装 pymoo）
- `notebooks/01_pareto_visualize.ipynb` 可运行并出图（需先跑脚本生成 pareto.json）
- 冒烟测试：`python tests/test_smoke.py` 通过（无需 pymoo）；或 `pytest tests/test_smoke.py`

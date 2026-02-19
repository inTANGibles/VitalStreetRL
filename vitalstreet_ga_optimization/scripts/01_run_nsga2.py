#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运行 NSGA-II：--toy 可跑通，输出 outputs/pareto.json 与 plots/*.png"""
import sys
import json
import argparse
from pathlib import Path
import numpy as np

# 工程根目录
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    import yaml
except ImportError:
    yaml = None

def load_config(path: str) -> dict:
    if yaml is None:
        return _default_config()
    with open(path, "r", encoding="utf-8") as f:
        out = yaml.safe_load(f)
    return out if out else _default_config()

def _default_config() -> dict:
    return {
        "max_steps": 20,
        "cost_alpha": 1.0,
        "cost_beta": 0.5,
        "reward": {"mu_violation": 1.0, "vitality_metrics": {}},
        "transition": {"flow": {"buffer_distance": 10.0, "diversity_weight": 0.5, "weighted_sum_weight": 0.5}},
        "action_space": {"max_shops": 500, "max_business_types": 20, "max_total_units": 500},
    }

from vitalga import WorldState, run_nsga2
from vitalga.evaluator import Evaluator
from vitalga.action_space import ActionSpace


def make_toy_state():
    """创建最小 toy 状态：若干 shop + 一个 public_space，用于 --toy 跑通。"""
    from vitalga.state import SpaceUnitCollection, WorldState
    from vitalga.business_type import BusinessTypeCollection
    collection = SpaceUnitCollection()
    # 简单四边形
    coords = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    for i in range(3):
        c = coords + np.array([i * 12.0, 0])
        gdf = SpaceUnitCollection._create_unit_by_coords(
            c, unit_type="shop", business_type="retail",
            business_category="retail", protected=False, enabled=True,
        )
        collection.add_space_unit(gdf)
    c_pub = coords + np.array([0, 12.0])
    gdf_pub = SpaceUnitCollection._create_unit_by_coords(
        c_pub, unit_type="public_space", business_type="N/A",
        business_category=None, protected=False, enabled=True,
    )
    gdf_pub.loc[gdf_pub.index[0], "flow_prediction"] = 0.5
    collection.add_space_unit(gdf_pub)
    business_types = BusinessTypeCollection()
    state = WorldState(
        space_units=collection,
        business_types=business_types,
        graph=None,
        budget=1e6,
        constraints={},
        step_idx=0,
        episode_id="toy",
    )
    return state


def main():
    parser = argparse.ArgumentParser(description="Run NSGA-II for VitalStreet GA")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "ga.yaml"))
    parser.add_argument("--geojson", type=str, default=None, help="初始 GeoJSON（不用则 --toy）")
    parser.add_argument("--toy", action="store_true", help="使用内存 toy 状态跑通")
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--gen", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        config = _default_config()
    config.setdefault("max_steps", 20)
    if config.get("cost_alpha") is None:
        config["cost_alpha"] = 1.0
    if config.get("cost_beta") is None:
        config["cost_beta"] = 0.5

    if args.toy:
        initial_state = make_toy_state()
    else:
        geojson = args.geojson or (ROOT / "data" / "initial.geojson")
        if not Path(geojson).exists():
            print("未提供 --geojson 且 data/initial.geojson 不存在，请使用 --toy 或提供 GeoJSON")
            return 1
        initial_state = WorldState.from_geojson(geojson, budget=config.get("budget", 1e6))

    out_dir = Path(args.out_dir or ROOT / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    X, F, extra = run_nsga2(
        initial_state, config,
        pop_size=args.pop, n_gen=args.gen, seed=args.seed,
    )
    # F: (n, 3) = (-vitality, violation, cost_proxy)
    # 保存时把 vitality 存为正值
    n = len(F)
    vitality = -F[:, 0]
    violation = F[:, 1]
    cost = F[:, 2]
    evaluator = Evaluator(config)
    action_space = ActionSpace(config.get("action_space", {}))
    max_steps = config.get("max_steps", 20)
    details = []
    for i in range(n):
        seq = X[i].reshape(max_steps, 3)
        res = evaluator.evaluate(initial_state.copy(), seq)
        details.append({
            "final_vitality": res["final_vitality"],
            "violation_total": res["violation_total"],
            "cost_proxy": res["cost_proxy"],
            "n_change_business": res["n_change_business"],
            "n_shop_to_public": res["n_shop_to_public"],
            "violation_breakdown": res["violation_breakdown"],
            "delta_vitality": res["delta_vitality"],
        })
    pareto_data = {
        "objectives": {"vitality": vitality.tolist(), "violation": violation.tolist(), "cost": cost.tolist()},
        "n_solutions": n,
        "details": details,
        "config": {k: v for k, v in config.items() if k not in ("reward", "transition")},
    }
    pareto_path = out_dir / "pareto.json"
    with open(pareto_path, "w", encoding="utf-8") as f:
        json.dump(pareto_data, f, ensure_ascii=False, indent=2)
    print(f"已保存: {pareto_path}")

    # 简单 2D 图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].scatter(vitality, violation, alpha=0.7)
        ax[0].set_xlabel("vitality")
        ax[0].set_ylabel("violation")
        ax[0].set_title("Vitality vs Violation")
        ax[1].scatter(vitality, cost, alpha=0.7)
        ax[1].set_xlabel("vitality")
        ax[1].set_ylabel("cost_proxy")
        ax[1].set_title("Vitality vs Cost")
        plt.tight_layout()
        plt.savefig(plots_dir / "pareto_2d.png", dpi=100)
        plt.close()
        print(f"已保存: {plots_dir / 'pareto_2d.png'}")
    except Exception as e:
        print(f"绘图跳过: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

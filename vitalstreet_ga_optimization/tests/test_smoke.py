"""冒烟测试：状态、动作、转移、奖励、评估、编码可运行"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vitalga import WorldState, ActionSpace, ActionType, Transition, RewardCalculator, Evaluator
from vitalga.state import SpaceUnitCollection
from vitalga.business_type import BusinessTypeCollection


def _toy_state():
    collection = SpaceUnitCollection()
    coords = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    for i in range(2):
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
    return WorldState(
        space_units=collection,
        business_types=BusinessTypeCollection(),
        graph=None,
        budget=1e6,
        constraints={},
        step_idx=0,
        episode_id=None,
    )


def test_state_copy():
    state = _toy_state()
    c = state.copy()
    assert c.step_idx == state.step_idx
    assert len(c.space_units.get_all_space_units()) == len(state.space_units.get_all_space_units())


def test_action_space_decode():
    state = _toy_state()
    config = {"max_shops": 100, "max_business_types": 20, "max_total_units": 100}
    asp = ActionSpace(config)
    # NO_OP
    a = asp.decode(np.array([9, 0, 0]), state)
    assert a.type == ActionType.NO_OP
    # CHANGE_BUSINESS
    a = asp.decode(np.array([0, 0, 1]), state)
    assert a.type == ActionType.CHANGE_BUSINESS or a.type == ActionType.NO_OP
    dims = asp.get_action_dim()
    assert len(dims) == 3


def test_transition_step():
    state = _toy_state()
    trans = Transition({"flow": {"buffer_distance": 10.0, "diversity_weight": 0.5, "weighted_sum_weight": 0.5}})
    from vitalga.action_space import Action
    # NO_OP
    next_s, info = trans.step(state, Action(type=ActionType.NO_OP, target_id=None, params={}))
    assert next_s.step_idx == state.step_idx + 1


def test_reward_and_evaluator():
    state = _toy_state()
    config = {
        "mu_violation": 1.0,
        "vitality_metrics": {},
        "max_steps": 5,
        "cost_alpha": 1.0,
        "cost_beta": 0.5,
        "reward": {"mu_violation": 1.0, "vitality_metrics": {}},
        "transition": {"flow": {}},
        "action_space": {"max_total_units": 100, "max_business_types": 20},
    }
    calc = RewardCalculator(config["reward"])
    calc.reset(state)
    v = calc._compute_vitality(state)
    assert v >= 0
    ev = Evaluator(config)
    seq = np.zeros((5, 3), dtype=np.int32)  # 全 NO_OP 或无效
    res = ev.evaluate(state, seq)
    assert "final_vitality" in res and "violation_total" in res and "cost_proxy" in res


def test_encoding_legal_actions():
    state = _toy_state()
    config = {"max_shops": 100, "max_business_types": 20, "max_total_units": 100}
    asp = ActionSpace(config)
    from vitalga.encoding import create_individual, random_population
    rng = np.random.default_rng(42)
    ind = create_individual(state, asp, length=5, rng=rng)
    assert ind.shape == (5, 3)
    pop = random_population(state, asp, pop_size=3, length=5, rng=rng)
    assert pop.shape == (3, 5, 3)


def test_ga_nodewise_smoke():
    """方案A：2 代 GA nodewise，不报错且 F 形状正确。"""
    try:
        from vitalga import run_nsga2
    except ImportError:
        return
    state = _toy_state()
    config = {
        "encoding_type": "nodewise_action",
        "B": 2,
        "pm": 0.2,
        "seed": 42,
        "cost_alpha": 1.0,
        "cost_beta": 0.5,
        "reward": {"mu_violation": 1.0, "vitality_metrics": {}},
        "transition": {"flow": {"buffer_distance": 10.0, "diversity_weight": 0.5, "weighted_sum_weight": 0.5}},
        "action_space": {"max_shops": 100, "max_business_types": 20, "max_total_units": 100},
    }
    X, F, extra = run_nsga2(state, config, pop_size=4, n_gen=2, seed=42)
    assert F.ndim == 2 and F.shape[1] == 3
    assert len(F) > 0
    assert "editable_nodes" in extra


if __name__ == "__main__":
    test_state_copy()
    test_action_space_decode()
    test_transition_step()
    test_reward_and_evaluator()
    test_encoding_legal_actions()
    test_ga_nodewise_smoke()
    print("All tests passed.")

"""使用 pymoo 实现 NSGA-II，三目标：min -vitality, min violation, min cost_proxy；方案A nodewise_action 无时序 Genome"""
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from .state import WorldState
from .action_space import ActionSpace
from .evaluator import Evaluator, evaluate_genome
from .encoding import (
    random_population,
    get_editable_nodes,
    build_action_specs,
    build_action_mask,
    random_genome,
    repair_budget,
    decode_to_actions,
    crossover_nodewise,
    mutate_nodewise,
)
from .transition import Transition, apply_actions_set

try:
    from pymoo.core.problem import Problem
    from pymoo.core.sampling import Sampling
    from pymoo.core.crossover import Crossover
    from pymoo.core.mutation import Mutation
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.pntx import SinglePointCrossover
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.repair.rounding import RoundingRepair
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    Problem = object
    Sampling = object
    Crossover = object
    Mutation = object


class VitalStreetProblem(Problem):
    """三目标最小化：f1=-final_vitality, f2=violation_total, f3=cost_proxy。个体展平为 (L*3,) 整数。"""

    def __init__(
        self,
        initial_state: WorldState,
        evaluator: Evaluator,
        action_space: ActionSpace,
        config: Dict[str, Any],
        seed: Optional[int] = None,
    ):
        self._initial_state = initial_state
        self._evaluator = evaluator
        self._action_space = action_space
        self._config = config
        self._rng = np.random.default_rng(seed)
        self._max_steps = config.get('max_steps', 50)
        dims = action_space.get_action_dim()
        n_units = min(dims[1], max(1, len(initial_state.space_units.get_all_space_units())))
        n_var = self._max_steps * 3
        xl = np.zeros(n_var, dtype=int)
        xu = np.zeros(n_var, dtype=int)
        for i in range(self._max_steps):
            xl[i * 3] = 0
            xu[i * 3] = max(0, dims[0] - 1)
            xl[i * 3 + 1] = 0
            xu[i * 3 + 1] = max(0, n_units - 1)
            xl[i * 3 + 2] = 0
            xu[i * 3 + 2] = max(0, dims[2] - 1)
        super().__init__(n_var=n_var, n_obj=3, n_ieq_constr=0, xl=xl, xu=xu, vtype=int)

    def _decode(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).flatten()
        L = self._max_steps
        if len(x) >= L * 3:
            return np.clip(x[: L * 3].reshape(L, 3), self.xl.reshape(L, 3), self.xu.reshape(L, 3)).astype(np.int32)
        pad = np.array(self.xl.copy())
        pad[: min(len(x), L * 3)] = x[: min(len(x), L * 3)]
        return pad.reshape(L, 3).astype(np.int32)

    def _evaluate(self, x, out, *args, **kwargs):
        X = x if len(x.shape) > 1 else x.reshape(1, -1)
        F = np.zeros((len(X), 3))
        for i, row in enumerate(X):
            seq = self._decode(row)
            res = self._evaluator.evaluate(self._initial_state.copy(), seq)
            F[i, 0] = -res['final_vitality']
            F[i, 1] = res['violation_total']
            F[i, 2] = res['cost_proxy']
        out["F"] = F


class LegalActionSampling(Sampling if PYMOO_AVAILABLE else object):
    """从合法动作生成初始种群（尊重 mask）。"""
    def __init__(self, initial_state: WorldState, action_space: ActionSpace, max_steps: int):
        if PYMOO_AVAILABLE:
            super().__init__()
        self._state = initial_state
        self._action_space = action_space
        self._max_steps = max_steps

    def _do(self, problem, n_samples, **kwargs):
        rng = np.random.default_rng(kwargs.get("seed"))
        pop = random_population(
            self._state, self._action_space, n_samples, self._max_steps, rng
        )
        return pop.reshape(n_samples, -1)


# ---------- 方案A：无时序 Genome ----------
def _repair_population_nodewise(
    X: np.ndarray,
    B: int,
    mask: List[List[int]],
    rng: np.random.Generator,
    policy: str = "random_drop",
) -> np.ndarray:
    """对种群每行执行 repair_budget。"""
    out = np.asarray(X, dtype=np.int32).copy()
    for i in range(out.shape[0]):
        out[i] = repair_budget(out[i], B, policy=policy, mask=mask, rng=rng)
    return out


class VitalStreetProblemNodewise(Problem if PYMOO_AVAILABLE else object):
    """方案A：n_var=M，基因值 0..N，三目标。评估前先 repair_budget。"""
    def __init__(
        self,
        initial_state: WorldState,
        editable_nodes: List[Any],
        mask: List[List[int]],
        action_specs: List[Tuple[int, int]],
        B: int,
        decode_to_actions_fn,
        apply_fn,
        objective_fn,
        config: Dict[str, Any],
        seed: Optional[int] = None,
    ):
        self._initial_state = initial_state
        self._editable_nodes = editable_nodes
        self._mask = mask
        self._action_specs = action_specs
        self._B = B
        self._decode_to_actions_fn = decode_to_actions_fn
        self._apply_fn = apply_fn
        self._objective_fn = objective_fn
        self._config = config
        self._rng = np.random.default_rng(seed)
        M = len(editable_nodes)
        N = len(action_specs)
        xl = np.zeros(M, dtype=int)
        xu = np.full(M, N, dtype=int)
        super().__init__(n_var=M, n_obj=3, n_ieq_constr=0, xl=xl, xu=xu, vtype=int)

    def _evaluate(self, x, out, *args, **kwargs):
        X = x if len(x.shape) > 1 else x.reshape(1, -1)
        X = _repair_population_nodewise(X, self._B, self._mask, self._rng)
        F = np.zeros((len(X), 3))
        for i, row in enumerate(X):
            res = evaluate_genome(
                row,
                self._initial_state.copy(),
                self._editable_nodes,
                self._decode_to_actions_fn,
                self._apply_fn,
                self._objective_fn,
            )
            F[i, 0] = -res["final_vitality"]
            F[i, 1] = res["violation_total"]
            F[i, 2] = res["cost_proxy"]
        out["F"] = F


class LegalGenomeSampling(Sampling if PYMOO_AVAILABLE else object):
    """方案A：从 mask 与预算 B 生成合法 genome。"""
    def __init__(self, M: int, N: int, mask: List[List[int]], B: int):
        if PYMOO_AVAILABLE:
            super().__init__()
        self._M = M
        self._N = N
        self._mask = mask
        self._B = B

    def _do(self, problem, n_samples, **kwargs):
        rng = np.random.default_rng(kwargs.get("seed"))
        pop = np.zeros((n_samples, self._M), dtype=np.int32)
        for i in range(n_samples):
            pop[i] = random_genome(self._M, self._N, self._mask, self._B, rng)
        return pop


class NodewiseCrossover(Crossover if PYMOO_AVAILABLE else object):
    """均匀交叉，交叉后由 Problem 评估前 repair。"""
    def __init__(self):
        if PYMOO_AVAILABLE:
            super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        rng = np.random.default_rng(kwargs.get("seed"))
        Y = np.zeros((2, n_matings, n_var), dtype=np.int32)
        for j in range(n_matings):
            c1, c2 = crossover_nodewise(X[0, j], X[1, j], rng)
            Y[0, j], Y[1, j] = c1, c2
        return Y


class NodewiseMutation(Mutation if PYMOO_AVAILABLE else object):
    """按 mask 重采样变异。"""
    def __init__(self, mask: List[List[int]], pm: float):
        if PYMOO_AVAILABLE:
            super().__init__()
        self._mask = mask
        self._pm = pm

    def _do(self, problem, X, **kwargs):
        rng = np.random.default_rng(kwargs.get("seed"))
        Y = mutate_nodewise(X, self._mask, self._pm, rng)
        return Y


def _make_objective_fn_nodewise(evaluator: Evaluator, config: Dict[str, Any]):
    """返回 (initial_state, state_after, actions) -> dict 供 evaluate_genome 使用。"""
    from .action_space import ActionType
    alpha = config.get("cost_alpha", 1.0)
    beta = config.get("cost_beta", 0.5)
    def objective_fn(initial_state, state_after, actions):
        evaluator.reward_calc.reset(initial_state)
        n_change = sum(1 for a in actions if a.type == ActionType.CHANGE_BUSINESS)
        n_public = sum(1 for a in actions if a.type == ActionType.SHOP_TO_PUBLIC_SPACE)
        final_vitality = evaluator.reward_calc._compute_vitality(state_after)
        violation_total = evaluator.reward_calc._compute_violation(state_after)
        cost_proxy = alpha * n_public + beta * n_change
        return {
            "final_vitality": final_vitality,
            "violation_total": violation_total,
            "cost_proxy": cost_proxy,
        }
    return objective_fn


def run_nsga2(
    initial_state: WorldState,
    config: Dict[str, Any],
    pop_size: int = 50,
    n_gen: int = 30,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """运行 NSGA-II。encoding_type==nodewise_action 时走方案A 无时序 Genome，否则走原动作序列。"""
    if not PYMOO_AVAILABLE:
        raise ImportError("请安装 pymoo: pip install pymoo")
    if config.get('cost_alpha') is None or config.get('cost_beta') is None:
        raise ValueError("config 必须包含 cost_alpha 与 cost_beta")
    encoding_type = config.get("encoding_type", "sequence")
    seed = seed if seed is not None else config.get("seed", 42)
    rng = np.random.default_rng(seed)

    if encoding_type == "nodewise_action":
        action_space = ActionSpace(config.get("action_space", {}))
        editable_nodes = get_editable_nodes(initial_state, action_space)
        M = len(editable_nodes)
        if M == 0:
            raise ValueError("nodewise_action 需要至少 1 个可改造节点，当前为 0")
        action_specs = build_action_specs(action_space)
        N = len(action_specs)
        mask = build_action_mask(initial_state, action_space, editable_nodes, action_specs)
        B = int(config.get("B", min(10, M)))
        evaluator = Evaluator(config)
        transition = Transition(config.get("transition", {}))
        def decode_fn(g, nodes, specs=action_specs):
            return decode_to_actions(g, nodes, specs)
        def apply_fn(s, actions):
            return apply_actions_set(s, actions, transition)
        objective_fn = _make_objective_fn_nodewise(evaluator, config)
        problem = VitalStreetProblemNodewise(
            initial_state=initial_state,
            editable_nodes=editable_nodes,
            mask=mask,
            action_specs=action_specs,
            B=B,
            decode_to_actions_fn=decode_fn,
            apply_fn=apply_fn,
            objective_fn=objective_fn,
            config=config,
            seed=seed,
        )
        sampling = LegalGenomeSampling(M, N, mask, B)
        pm = float(config.get("pm", 0.2))
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=NodewiseCrossover(),
            mutation=NodewiseMutation(mask, pm),
            eliminate_duplicates=True,
        )
        termination = get_termination("n_gen", n_gen)
        res = minimize(problem, algorithm, termination, seed=seed, verbose=True)
        return res.X, res.F, {"res": res, "editable_nodes": editable_nodes}
    else:
        evaluator = Evaluator(config)
        action_space = ActionSpace(config.get("action_space", {}))
        problem = VitalStreetProblem(
            initial_state=initial_state,
            evaluator=evaluator,
            action_space=action_space,
            config=config,
            seed=seed,
        )
        sampling = LegalActionSampling(initial_state, action_space, problem._max_steps)
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=SinglePointCrossover(prob=0.9),
            mutation=PM(prob=0.2, eta=20, vtype=float, repair=RoundingRepair()),
            eliminate_duplicates=True,
        )
        termination = get_termination("n_gen", n_gen)
        res = minimize(problem, algorithm, termination, seed=seed, verbose=True)
        return res.X, res.F, {"res": res}

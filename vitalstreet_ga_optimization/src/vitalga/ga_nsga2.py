"""使用 pymoo 实现 NSGA-II，三目标：min -vitality, min violation, min cost_proxy"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
from .state import WorldState
from .action_space import ActionSpace
from .evaluator import Evaluator
from .encoding import random_population

try:
    from pymoo.core.problem import Problem
    from pymoo.core.sampling import Sampling
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


def run_nsga2(
    initial_state: WorldState,
    config: Dict[str, Any],
    pop_size: int = 50,
    n_gen: int = 30,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """运行 NSGA-II。返回 X (决策变量), F (目标值), extra."""
    if not PYMOO_AVAILABLE:
        raise ImportError("请安装 pymoo: pip install pymoo")
    if config.get('cost_alpha') is None or config.get('cost_beta') is None:
        raise ValueError("config 必须包含 cost_alpha 与 cost_beta")
    evaluator = Evaluator(config)
    action_space = ActionSpace(config.get('action_space', {}))
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
    return res.X, res.F, {'res': res}

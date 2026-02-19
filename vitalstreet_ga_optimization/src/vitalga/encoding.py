"""个体编码：长度 L=max_steps 的动作序列；初始化/交叉/变异需尊重 mask"""
from typing import List, Tuple, Optional
import numpy as np
from .state import WorldState
from .action_space import ActionSpace


def action_dim(config: dict) -> Tuple[int, int, int]:
    """(n_action_types, n_units, n_business_types)"""
    asp = ActionSpace(config.get('action_space', {}))
    return asp.get_action_dim()


def create_individual(
    state: WorldState,
    action_space: ActionSpace,
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """生成一条合法个体：length 行，每行 [action_type, unit_index, type_id]。仅从合法动作中采样。"""
    legal = action_space.get_legal_actions(state)
    if len(legal) == 0:
        # 无合法动作时返回 NO_OP 占位
        dims = action_space.get_action_dim()
        return np.zeros((length, 3), dtype=np.int32)
    ind = np.zeros((length, 3), dtype=np.int32)
    for i in range(length):
        idx = rng.integers(0, len(legal))
        ind[i] = legal[idx]
    return ind


def crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """单点交叉，保持长度一致。"""
    L = len(parent_a)
    assert len(parent_b) == L
    k = rng.integers(1, L) if L > 1 else 0
    c1 = np.concatenate([parent_a[:k], parent_b[k:]], axis=0)
    c2 = np.concatenate([parent_b[:k], parent_a[k:]], axis=0)
    return c1, c2


def mutate(
    individual: np.ndarray,
    state: WorldState,
    action_space: ActionSpace,
    mutation_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """以 mutation_rate 概率将某一步替换为从合法动作中随机采样的动作。"""
    out = individual.copy()
    legal = action_space.get_legal_actions(state)
    if len(legal) == 0:
        return out
    for i in range(len(out)):
        if rng.random() < mutation_rate:
            idx = rng.integers(0, len(legal))
            out[i] = legal[idx]
    return out


def random_population(
    state: WorldState,
    action_space: ActionSpace,
    pop_size: int,
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """形状 (pop_size, length, 3)。"""
    pop = np.zeros((pop_size, length, 3), dtype=np.int32)
    for i in range(pop_size):
        pop[i] = create_individual(state, action_space, length, rng)
    return pop

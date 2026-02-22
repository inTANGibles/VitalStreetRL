"""个体编码：长度 L=max_steps 的动作序列（旧）；方案A 无时序 Genome 长度 M（nodewise_action）。"""
from typing import List, Tuple, Optional, Any
import numpy as np
from .state import WorldState
from .action_space import ActionSpace, ActionType, Action
from .business_type import BusinessCategory


# ---------- 旧接口（动作序列）保留 ----------
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


# ---------- 方案A：无时序 Genome (nodewise_action) ----------
# Genome: 长度 M 的向量，G[i] in {0..N}，0=NO_OP，k=对第 i 个可改造节点选第 k 类动作。
# editable_nodes[i] -> node_id (uid)；mask[i] 为节点 i 允许的基因值集合（含 0）。


def get_editable_nodes(state: WorldState, action_space: ActionSpace) -> List[Any]:
    """可改造节点列表（node_id/uid），按 id 升序，与 mask 下标一一对应。"""
    all_units = state.space_units.get_all_space_units()
    replaceable = state.space_units.get_replaceable_shops()
    shops_for_public = state.space_units.get_shops_for_circulation()
    uids = set()
    for i in range(len(all_units)):
        if i >= action_space.max_total_units:
            break
        row = all_units.iloc[i]
        uid = row.name
        if row.get("protected", True) or not row.get("enabled", True):
            continue
        if row.get("unit_type", "") != "shop":
            continue
        if uid in replaceable.index or uid in shops_for_public.index:
            uids.add(uid)
    return sorted(uids, key=str)


def build_action_specs(action_space: ActionSpace) -> List[Tuple[int, int]]:
    """(action_type_value, type_id) 列表，下标 0..N-1 对应基因值 1..N。0=NO_OP 不在此列表。"""
    category_list = [c.value for c in BusinessCategory]
    n_types = min(len(category_list), action_space.max_business_types)
    specs: List[Tuple[int, int]] = []
    for type_id in range(n_types):
        specs.append((ActionType.CHANGE_BUSINESS.value, type_id))
    specs.append((ActionType.SHOP_TO_PUBLIC_SPACE.value, 0))
    return specs


def build_action_mask(
    state: WorldState,
    action_space: ActionSpace,
    editable_nodes: List[Any],
    action_specs: List[Tuple[int, int]],
) -> List[List[int]]:
    """每个节点允许的基因值（含 0）。mask[i] 为 list，0 一定在内。"""
    category_list = [c.value for c in BusinessCategory]
    mask: List[List[int]] = []
    for uid in editable_nodes:
        allowed = [0]
        for k, (action_type_val, type_id) in enumerate(action_specs):
            if action_type_val == ActionType.CHANGE_BUSINESS.value:
                new_type = category_list[type_id] if type_id < len(category_list) else "undefined"
                action = Action(
                    type=ActionType.CHANGE_BUSINESS,
                    target_id=uid,
                    params={"new_type": new_type},
                )
            else:
                action = Action(
                    type=ActionType.SHOP_TO_PUBLIC_SPACE,
                    target_id=uid,
                    params={},
                )
            if action_space.is_valid(state, action):
                allowed.append(k + 1)
        mask.append(allowed)
    return mask


def random_genome(
    M: int,
    N: int,
    mask: List[List[int]],
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """生成满足 mask 与预算 B 的随机 genome。M=节点数，N=动作选项数（基因值 1..N）。"""
    genome = np.zeros(M, dtype=np.int32)
    for i in range(M):
        genome[i] = int(rng.choice(mask[i]))
    genome = repair_budget(genome, B, policy="random_drop", mask=mask, rng=rng)
    return genome


def repair_budget(
    genome: np.ndarray,
    B: int,
    policy: str = "random_drop",
    mask: Optional[List[List[int]]] = None,
    priority_order: Optional[List[int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """若非零数量 > B，将超额位置置 0。priority_drop 时按 priority_order 先删低优先级。"""
    out = np.asarray(genome, dtype=np.int32).copy()
    nz = np.where(out != 0)[0]
    if len(nz) <= B:
        return out
    excess = len(nz) - B
    if policy == "priority_drop":
        if priority_order is None:
            priority_order = list(range(out.shape[0]))
        if rng is None:
            rng = np.random.default_rng()
        # 按优先级升序，低优先级的先删
        priorities = [priority_order[i] for i in nz]
        to_drop = [nz[i] for i in np.argsort(priorities)[:excess]]
        for idx in to_drop:
            if mask is not None and idx < len(mask):
                out[idx] = 0
            else:
                out[idx] = 0
    else:
        if rng is None:
            rng = np.random.default_rng()
        to_drop = rng.choice(nz, size=excess, replace=False)
        for idx in to_drop:
            out[idx] = 0
    return out


def decode_to_actions(
    genome: np.ndarray,
    editable_nodes: List[Any],
    action_specs: List[Tuple[int, int]],
) -> List[Action]:
    """Genome -> 动作列表，按 node_id 升序。返回 List[Action] 供 apply_actions_set 使用。"""
    category_list = [c.value for c in BusinessCategory]
    actions: List[Tuple[Any, Action]] = []
    for i in range(len(genome)):
        if genome[i] == 0:
            continue
        k = int(genome[i]) - 1
        if k < 0 or k >= len(action_specs):
            continue
        uid = editable_nodes[i]
        action_type_val, type_id = action_specs[k]
        if action_type_val == ActionType.CHANGE_BUSINESS.value:
            new_type = category_list[type_id] if type_id < len(category_list) else "undefined"
            actions.append((uid, Action(
                type=ActionType.CHANGE_BUSINESS,
                target_id=uid,
                params={"new_type": new_type},
            )))
        else:
            actions.append((uid, Action(
                type=ActionType.SHOP_TO_PUBLIC_SPACE,
                target_id=uid,
                params={},
            )))
    actions.sort(key=lambda x: (str(x[0]),))
    return [a for _, a in actions]


def crossover_nodewise(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """均匀交叉：每位以 0.5 概率来自 parent_a 或 parent_b。"""
    M = len(parent_a)
    assert len(parent_b) == M
    u = rng.random(M) < 0.5
    c1 = np.where(u, parent_a, parent_b).astype(np.int32)
    c2 = np.where(u, parent_b, parent_a).astype(np.int32)
    return c1, c2


def mutate_nodewise(
    genome: np.ndarray,
    mask: List[List[int]],
    pm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """以概率 pm 从 mask[i] 重采样每位。"""
    out = np.asarray(genome, dtype=np.int32).copy()
    for i in range(len(out)):
        if rng.random() < pm:
            out[i] = int(rng.choice(mask[i]))
    return out

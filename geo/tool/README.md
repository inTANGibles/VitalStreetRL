# 空间对象集合设计说明

## 概述

基于你的道路优化项目（`playground/geo`）中的Collection设计模式，为商业小镇更新RL系统设计了三个核心对象集合。

## 三类对象集合

### 1. SpaceUnitCollection（空间单元集合）⭐ 最核心

**文件**: `env/geo/space_unit.py`

**作用**: 管理"片段空间"（parcel/unit），这是RL动作的直接操作对象

**关键属性**:
- `uid`: UUID标识
- `geometry`: Shapely几何（Polygon）
- `demolishable`: 是否可拆除（用于DEMOLISH动作mask）
- `protected`: 是否受保护（用于约束检查）
- `business_type`: 当前业态类型（用于CHANGE_BUSINESS动作）
- `impedance`: 阻抗值（用于REDUCE_IMPEDANCE动作）
- `connectivity`: 连通性状态（用于CONNECT动作）
- `flow_prediction`: 预测客流（STGNN输出，用于观测）

**核心方法**:
- `get_demolishable_units()`: 获取可拆除单元（用于动作mask）
- `update_business_type()`: 更新业态（对应"业态替换"动作）
- `update_impedance()`: 更新阻抗（对应"阻抗降低"动作）
- `update_connectivity()`: 更新连通性（对应"连通打通"动作）

**与RL系统的关联**:
- 每个space unit对应一个action的`target_id`
- 属性直接用于`ActionSpace.get_action_mask()`
- 状态更新在`Transition.step()`中调用

---

### 2. StreetNetworkCollection（街道网络集合）⭐ 核心

**文件**: `env/geo/street_network.py`

**作用**: 管理街道网络，支持"连通打通"和"阻抗降低"动作

**关键属性**:
- `uid`: UUID标识
- `geometry`: LineString几何
- `width`: 街道宽度（用于最小街宽约束）
- `impedance`: 阻抗值（可降低）
- `connectable`: 是否可打通（用于动作mask）
- `flow_prediction`: 预测客流

**核心方法**:
- `get_connectable_streets()`: 获取可打通的街道（用于动作mask）
- `connect_streets()`: 执行连通打通动作
- `reduce_impedance()`: 执行阻抗降低动作
- `check_min_width()`: 检查最小街宽约束
- `get_network_graph()`: 获取NetworkX图（用于STGNN）

**与RL系统的关联**:
- 支持"连通打通"和"阻抗降低"两类动作
- 提供空间连通性信息给STGNN
- 阻抗值影响客流预测

---

### 3. BusinessTypeCollection（业态类型集合）⭐ 辅助

**文件**: `env/geo/business_type.py`

**作用**: 管理业态类型定义和属性，支持"业态替换"动作

**关键属性**（BusinessType数据类）:
- `type_id`: 业态ID
- `name`: 业态名称
- `category`: 业态类别（零售/餐饮/服务等）
- `attractiveness`: 吸引力系数（用于活力计算）
- `cost_to_change`: 替换成本（用于reward计算）
- `compatibility`: 兼容性矩阵

**核心方法**:
- `get_all_types()`: 获取所有业态类型
- `get_change_cost()`: 获取替换成本（用于RewardCalculator）
- `get_compatibility()`: 获取兼容性
- `get_attractiveness()`: 获取吸引力系数（用于VitalityMetrics）

**与RL系统的关联**:
- 定义"业态替换"动作的参数空间
- 提供成本信息给`RewardCalculator`
- 提供吸引力系数给`VitalityMetrics`

---

## 与现有系统的集成

### WorldState集成

```python
@dataclass
class WorldState:
    # 空间对象集合
    space_units: SpaceUnitCollection      # 核心
    street_network: StreetNetworkCollection  # 核心
    business_types: BusinessTypeCollection    # 辅助
    
    # 图结构（从space_units和street_network构建）
    graph: Any
    
    # 约束与预算
    budget: float
    constraints: Dict[str, Any]
```

### 动作类型映射

| 动作类型 | 操作对象 | 相关Collection方法 |
|---------|---------|-------------------|
| DEMOLISH | SpaceUnit | `SpaceUnitCollection.delete_space_unit()` |
| CONNECT | StreetNetwork | `StreetNetworkCollection.connect_streets()` |
| REDUCE_IMPEDANCE | StreetNetwork/SpaceUnit | `StreetNetworkCollection.reduce_impedance()` 或 `SpaceUnitCollection.update_impedance()` |
| CHANGE_BUSINESS | SpaceUnit | `SpaceUnitCollection.update_business_type()` |

### 动作Mask生成

在`ActionSpace.get_action_mask()`中：
```python
def get_action_mask(self, state: WorldState) -> np.ndarray:
    # 基于SpaceUnitCollection
    demolishable = state.space_units.get_demolishable_units()
    # 基于StreetNetworkCollection
    connectable = state.street_network.get_connectable_streets()
    # 基于预算和约束
    # ...
```

### 状态转移

在`Transition.step()`中：
```python
def step(self, state: WorldState, action: Action) -> Tuple[WorldState, Dict]:
    if action.type == ActionType.CHANGE_BUSINESS:
        state.space_units.update_business_type(action.target_id, action.params['new_type'])
    elif action.type == ActionType.CONNECT:
        state.street_network.connect_streets([action.target_id])
    # ...
```

---

## 设计模式对比

### 与playground/geo的对应关系

| playground/geo | VitalStreetRL | 说明 |
|---------------|--------------|------|
| RoadCollection | StreetNetworkCollection | 管理道路/街道网络 |
| BuildingCollection | SpaceUnitCollection | 管理空间单元（类似建筑物） |
| RegionCollection | - | 区域概念可融入SpaceUnit或StreetNetwork |

### 设计模式一致性

所有Collection类都遵循相同的设计模式：
1. **GeoDataFrame存储**: 使用`gpd.GeoDataFrame`存储几何和属性
2. **UID索引**: 使用UUID作为唯一标识和索引
3. **Collection级UID**: 每次修改后更新Collection的UID（用于缓存失效）
4. **增删改查**: 统一的`add_*`, `delete_*`, `get_*`, `update_*`方法
5. **属性查询**: `get_*_by_attr_and_value()`方法

---

## 实现优先级

1. **SpaceUnitCollection** (P0): 最核心，支持大部分动作
2. **StreetNetworkCollection** (P0): 支持连通和阻抗动作
3. **BusinessTypeCollection** (P1): 辅助，可先用简单字典替代

---

## 下一步工作

1. 完善各Collection的具体实现（几何操作、图构建等）
2. 实现WorldState的`_build_graph()`方法
3. 在`Transition.step()`中集成Collection的更新方法
4. 在`ActionSpace.get_action_mask()`中使用Collection的查询方法
5. 在`FeatureExtractor`中从Collection提取STGNN特征

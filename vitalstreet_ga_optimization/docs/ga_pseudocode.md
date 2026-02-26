# VitalStreet GA（NSGA-II）伪代码

GNN 仅用于：在**每个候选方案的末尾**，根据最终状态为每个 public_space 构建子图并预测客流量，从而计算目标 F1（活力）。交叉、变异、选择等操作不依赖 GNN。

---

## 符号

- **Genome（个体）** \(x\)：一条方案编码（如 nodewise 下为长度 M 的整数向量，每维表示该节点选的动作编号）。
- **Decode**：\(x \mapsto\) 动作列表 \(\mathcal{A}\)。
- **State**：\(s\) = 当前空间单元类型、业态、预算等；\(s_0\) = 初始状态。
- **Transition**：\(s' = \text{Apply}(s, \mathcal{A})\)，即按动作集合更新状态（改业态、商铺→公服等），**不**在此处算 flow。
- **F1（活力）**：在**最终状态**\(s'\) 上，用 GNN 为每个 public_space 预测客流量 → 写回 flow_prediction → F1 = \(\sum_{\text{public}} \text{area} \times \text{flow\_prediction}\)。
- **F2（违反）**：在 \(s'\) 上算混合度/空置率/集中度相对初始的恶化惩罚。

---

## NSGA-II 主流程（伪代码）

```
输入: 初始状态 s0, 种群大小 N, 最大代数 G, 编码/解码/变异/交叉参数
输出: Pareto 前沿 (X, F) 即 (决策变量, 目标值)

1.  初始化种群
    P ← 随机生成 N 个合法 Genome（或按 mask 采样，满足预算 B）
    t ← 0

2.  repeat until t = G:

    3.  评估种群（此处用 GNN 算 F1）
        for each 个体 x in P:
            A ← Decode(x)
            s' ← Apply(s0, A)           // 仅状态转移，不算 flow
            // -------- 方案末尾：用 GNN 算 F1 --------
            for each public_space 节点 in s':
                构建以该节点为中心的子图(1-hop/2-hop)
                flow_prediction[节点] ← GNN.predict(子图)
            F1(x) ← Σ (area × flow_prediction) over public_space
            F2(x) ← violation(s', s0)
            F(x) ← (-F1, F2)            // 最小化 -活力、最小化违反

    4.  非支配排序 + 拥挤度
        fronts ← FastNonDominatedSort(P, F)
        rank(x) ← 所在前沿层级
        distance(x) ← CrowdingDistance(P, F, front)

    5.  选择
        Q ← Select(P, rank, distance)   // 锦标赛等，选出 N 个

    6.  交叉与变异
        offspring ← ∅
        for 配对的父代 (p1, p2) from Q:
            c1, c2 ← Crossover(p1, p2)
            c1' ← Mutate(c1),  c2' ← Mutate(c2)
            offspring ← offspring ∪ {c1', c2'}
        (可选) Repair(offspring)       // 如预算 B、mask 约束

    7.  环境选择（精英保留）
        R ← P ∪ offspring
        P_next ← SelectBestN(R, rank, distance)   // 取前 N 个
        P ← P_next
        t ← t + 1

8.  return ParetoFront(P)   // 即 (X, F) 中 rank=1 的解
```

---

## 单次评估的细化（强调 GNN 只用于 F1）

```
Evaluate(x, s0):
    A ← Decode(x)
    s' ← Apply(s0, A)                    // Transition：只改 unit_type / business_type 等

    // --- 仅在此处用 GNN，为 F1 提供客流量 ---
    for each 单元 u where u.unit_type == public_space in s':
        subgraph ← BuildSubgraph(s', u, num_hops=2)
        u.flow_prediction ← GNN(subgraph)
    F1 ← Σ u.area × u.flow_prediction
    F2 ← violation(s', s0)

    return (F1, F2)
```

---

## 小结

| 环节           | 是否用 GNN | 说明 |
|----------------|------------|------|
| 初始化 / 交叉 / 变异 / 选择 | 否 | 只操作 Genome（整数编码） |
| Transition Apply(s, A) | 否 | 只改状态（类型、业态），不写 flow_prediction |
| **评估时算 F1** | **是** | 在最终状态 s' 上为每个 public_space 建子图 → GNN 预测 → 得到 F1 |

因此：**GNN 仅作为“方案末尾”的 F1 计算器**，GA 通过交叉与变异搜索方案空间，每次评估时在方案末尾调用 GNN 得到活力目标。

# 训练结果与调试指南

## 当前现象简要总结

| 现象 | 含义 |
|------|------|
| **Train loss** 平滑下降并稳定 ~1.65 | 模型在训练集上能学习 |
| **Val MAE** 剧烈波动、多次尖峰 | 过拟合或训练不稳定，泛化差 |
| **Pred vs True** 拟合线斜率 < 1（如 0.73×true+1.66） | 系统性偏差：低流量高估、高流量低估（向均值收缩） |
| **Error 分布** 主峰在 0、次峰在 4–5 | 部分样本被明显高估 |

## 已做的代码改动（可直接用）

1. **SAGE 增加 dropout**（`models/sage.py`）  
   - 默认 0，在 `run_pipeline` 中从 `optuna_best_params` 读，缺省为 **0.2**。  
   - 缓解过拟合、平滑 Val MAE 曲线。

2. **训练时梯度裁剪**（`scripts/run_pipeline.py`）  
   - `max_grad_norm` 默认 **1.0**，减轻 Val MAE 尖峰。

3. **学习率调度**  
   - `ReduceLROnPlateau`：Val MAE 连续 15 轮不降则 lr×0.5，最低 1e-5。  
   - 有助于后期稳定、减轻震荡。

4. **Optuna 可调 dropout**（`scripts/optuna_tune.py`）  
   - 新增超参 `dropout` ∈ [0, 0.4]，步长 0.1。  
   - 重新跑 Optuna 后，`optuna_best_params.json` 会包含 `dropout`，pipeline 会自动使用。

## 建议的调试顺序

1. **不重跑 Optuna 时**  
   - 直接再训练：当前逻辑已用默认 `dropout=0.2`、`max_grad_norm=1.0` 和 LR 调度。  
   - 观察 Val MAE 是否更平滑、Best val_mae 是否改善。

2. **重跑 Optuna**  
   - `python scripts/optuna_tune.py --n_trials 15`，让 Optuna 同时搜 lr、weight_decay、hidden_channels、**dropout** 等。  
   - 再用新生成的 `optuna_best_params.json` 跑 `run_pipeline.py`。

3. **若 Val 仍波动大**  
   - 适当**降低初始 lr**（如 5e-4）或增大 `weight_decay`。  
   - 在 `optuna_best_params.json` 中可手写 `"dropout": 0.3`、`"max_grad_norm": 0.5` 做快速试验。

4. **针对「高估次峰」与「斜率<1」**  
   - **数据**：检查误差在 4–5 的样本（高估）是否集中在某类子图/时段，或存在标注异常。  
   - **损失**：当前为 L1（MAE）；若希望更惩罚大误差，可尝试 Huber 或对误差>阈值的样本加权。  
   - **特征/目标**：检查高流量子图特征是否充分、标签分布是否均衡，必要时做分层采样或目标变换。

5. **早停**  
   - 已启用：`patience=50`，取 val_mae 最优的 checkpoint。  
   - 若收敛很慢，可适当增大 `n_epochs` 或 `patience`。

## 快速手动改参（不改 Optuna）

在 `checkpoints/optuna_best_params.json` 中增加或修改（若文件不存在，pipeline 会用默认 dropout/max_grad_norm）：

```json
{
  "lr": 0.0005,
  "weight_decay": 0.001,
  "hidden_channels": 64,
  "n_epochs": 400,
  "dropout": 0.25,
  "max_grad_norm": 1.0
}
```

然后运行：

```bash
python scripts/run_pipeline.py --no_optuna
```

即可用上述参数训练并观察曲线与 Pred vs True、Error 分布是否改善。

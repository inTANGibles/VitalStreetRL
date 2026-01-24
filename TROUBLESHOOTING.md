# 故障排查指南

## 访问冲突错误 (0xC0000005)

### 错误含义

`exit code -1073741819` 或 `0xC0000005` 表示**访问冲突 (Access Violation)**，这是Windows系统中最常见的严重错误之一。

**含义**：程序尝试访问不被允许的内存地址，导致操作系统强制终止进程。

### 可能的原因

1. **数组越界**：访问数组/列表时索引超出范围
2. **空指针解引用**：访问 `None` 或未初始化的对象
3. **内存对齐问题**：NumPy/PyTorch数组的内存对齐错误
4. **C扩展库bug**：底层C库（如geopandas、shapely）的内存管理问题
5. **多线程竞争**：多个线程同时访问共享资源
6. **内存泄漏**：长时间运行导致内存耗尽

### 已实施的修复措施

#### 1. 异常处理
- 在 `reset()` 和 `step()` 方法中添加了完整的异常捕获
- 每个关键操作都有独立的try-except块
- 失败时返回安全的默认值，而不是崩溃

#### 2. 内存安全检查
- 观测编码后验证形状和类型
- 动作解码失败时使用NO_OP作为fallback
- 状态转移失败时保持当前状态

#### 3. 调试信息
- 详细的错误日志，包括堆栈跟踪
- 每个步骤的状态信息打印
- 异常发生时的上下文信息

### 进一步调试建议

#### 1. 启用详细日志
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 2. 使用内存分析工具
```bash
# 安装内存分析器
pip install memory_profiler

# 运行并分析内存使用
python -m memory_profiler main.py
```

#### 3. 检查特定模块
如果错误频繁发生，可以逐个测试：
- `RasterObservation.encode()` - 栅格编码
- `Transition.step()` - 状态转移
- `RewardCalculator.compute()` - 奖励计算
- `VitalityMetrics.compute()` - 活力指标

#### 4. 减少并发
如果使用多进程/多线程，尝试：
- 设置 `n_envs=1`（单环境）
- 禁用GPU（使用CPU）
- 减少batch_size

#### 5. 检查数据文件
- 验证GeoJSON文件格式正确
- 检查文件路径是否存在
- 确认数据文件没有损坏

### 常见问题

**Q: 为什么大部分episode都是NO_OP动作？**
A: 这是正常的，在训练初期策略尚未学习到有效动作。随着训练进行，策略会逐渐改进。

**Q: 为什么episode总是因为stagnation终止？**
A: 如果环境在连续多步没有显著变化，会触发停滞检测。可以调整 `stagnation_threshold` 参数。

**Q: 如何避免内存错误？**
A: 
1. 确保所有数组操作都有边界检查
2. 及时释放不需要的对象
3. 使用较小的batch_size和n_steps
4. 定期重启训练进程

### 如果问题仍然存在

1. **收集错误信息**：
   - 完整的错误堆栈
   - 发生错误时的环境状态
   - 最近的操作序列

2. **简化环境**：
   - 使用最小的测试配置
   - 禁用STGNN预测（如果可能）
   - 使用mock数据

3. **检查依赖版本**：
   ```bash
   pip list | grep -E "(numpy|torch|geopandas|shapely)"
   ```
   确保所有依赖版本兼容

4. **联系支持**：
   提供完整的错误日志和复现步骤

# MultiAgentStockSelector 使用说明

## 📍 定位

`MultiAgentStockSelector` 是一个**独立的选股系统**，不依赖 Qlib 框架。

## ✅ 适用场景

### 1. 非 Qlib 工作流
如果你的策略不使用 Qlib 回测系统，MultiAgentStockSelector 提供了开箱即用的选股能力。

```python
from strategies.multi_agent_selector import MultiAgentStockSelector

# 独立使用
selector = MultiAgentStockSelector(
    chanlun_weight=0.35,
    technical_weight=0.25,
    volume_weight=0.15,
    fundamental_weight=0.15,
    sentiment_weight=0.10
)

# 批量评分
results = selector.batch_score(stock_data, top_n=10)
```

### 2. 快速原型验证
需要快速验证多因子选股逻辑，不想配置复杂的 Qlib 工作流。

### 3. 明确的因子逻辑
需要清晰的技术指标/成交量/情绪评分逻辑，而不是依赖机器学习模型的黑盒预测。

## ⚠️ 局限性

1. **技术指标重复**: MACD/RSI/MA/BBands 的实现与 Qlib Alpha191 因子功能重叠
2. **不支持 Qlib 生态**: 无法使用 Qlib 的回测、绩效分析、MLflow 集成等功能
3. **独立维护**: 需要单独维护技术指标计算逻辑

## 🔄 迁移建议

### 迁移到 ChanLunEnhancedStrategy

**适用于**:
- 使用 Qlib 工作流
- 希望复用麒麟系统的 Qlib 基础设施
- 需要完整的回测和绩效分析

**迁移步骤**:

1. **确认 Handler 包含必要特征**
   
   检查你的 Qlib Handler 是否包含技术指标特征。如果没有，需要先添加：
   
   ```python
   # 在 Handler 中添加技术指标
   class EnhancedHandler(DataHandlerLP):
       def __init__(self, **kwargs):
           # 添加 TA-Lib 技术指标
           infer_processors = [
               {"class": "TaLibProcessor", "kwargs": {}},  # MACD/RSI/MA
               {"class": "VolumeProcessor", "kwargs": {}},  # 成交量特征
           ]
   ```

2. **使用新策略**
   
   ```python
   from strategies.chanlun_qlib_strategy import ChanLunEnhancedStrategy
   
   strategy = ChanLunEnhancedStrategy(
       model=model,  # LightGBM 会自动学习技术指标特征
       dataset=dataset,
       chanlun_weight=0.35,
       topk=30
   )
   ```

3. **配置文件方式**
   
   ```bash
   qrun run --config_path configs/chanlun/enhanced_strategy.yaml
   ```

## 🎯 核心区别

| 特性 | MultiAgentStockSelector | ChanLunEnhancedStrategy |
|------|------------------------|------------------------|
| Qlib 集成 | ❌ 独立系统 | ✅ 深度集成 |
| 技术指标 | ✅ 明确实现 (150行) | ⚠️ 依赖模型学习 |
| 成交量分析 | ✅ 明确实现 (80行) | ⚠️ 依赖模型学习 |
| 情绪分析 | ✅ 明确实现 (60行) | ⚠️ 依赖模型学习 |
| 回测框架 | ❌ 需额外配置 | ✅ Qlib 原生 |
| 易用性 | ✅ 开箱即用 | ⚠️ 需配置 Qlib |
| 可解释性 | ✅ 明确规则 | ⚠️ 模型黑盒 |

## 📝 推荐使用流程

### 对于新项目
1. ✅ 优先使用 `ChanLunEnhancedStrategy` (Qlib 生态)
2. ✅ 确保 Handler 包含技术指标/成交量特征
3. ✅ 使用 LightGBM 学习特征组合

### 对于已有项目
1. ✅ 继续使用 `MultiAgentStockSelector` (稳定可用)
2. 🔄 逐步迁移到 Qlib 工作流 (如果需要)

## 🔗 相关文档

- [ChanLunEnhancedStrategy 文档](../docs/PHASE2_REFACTOR_SUMMARY.md)
- [Qlib Handler 配置](../qlib_enhanced/chanlun/)
- [融合优化分析](../docs/CHANLUN_INTEGRATION_OPTIMIZATION.md)

---

**维护状态**: ✅ 活跃维护  
**推荐迁移**: 视项目需求而定  
**删除计划**: 无计划删除

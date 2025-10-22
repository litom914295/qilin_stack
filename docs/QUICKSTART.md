# 🚀 快速开始指南

## 5分钟上手 Qilin Stack

### 前置要求

- Python 3.9+
- 8GB+ RAM
- Windows/Linux/Mac OS

### 1. 安装依赖

```bash
# 克隆项目（如果需要）
git clone <repository-url>
cd qilin_stack_with_ta

# 安装Python依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# Windows PowerShell
$env:LLM_API_KEY="your-api-key-here"
$env:LLM_API_BASE="https://api.tu-zi.com"

# Linux/Mac
export LLM_API_KEY="your-api-key-here"
export LLM_API_BASE="https://api.tu-zi.com"
```

### 3. 运行第一个决策

创建文件 `examples/quick_start.py`:

```python
import asyncio
from decision_engine.core import get_decision_engine

async def main():
    # 初始化引擎
    engine = get_decision_engine()
    
    # 生成决策
    symbols = ['000001.SZ', '600000.SH']
    decisions = await engine.make_decisions(symbols, '2024-06-30')
    
    # 打印结果
    for decision in decisions:
        print(f"\n股票: {decision.symbol}")
        print(f"信号: {decision.final_signal.value}")
        print(f"置信度: {decision.confidence:.2%}")
        print(f"推理: {decision.reasoning}")

if __name__ == '__main__':
    asyncio.run(main())
```

运行：
```bash
python examples/quick_start.py
```

### 4. 查看监控指标

```python
from monitoring.metrics import get_monitor

monitor = get_monitor()
summary = monitor.get_summary()

print(f"运行时间: {summary['uptime']:.2f}秒")
print(f"总决策数: {summary['total_decisions']}")
print(f"总信号数: {summary['total_signals']}")
```

---

## 核心概念

### 信号类型

| 信号 | 说明 | 使用场景 |
|------|------|----------|
| STRONG_BUY | 强烈买入 | 高置信度、多信号一致 |
| BUY | 买入 | 中等置信度、正面信号 |
| HOLD | 持有 | 观望、信号不明确 |
| SELL | 卖出 | 负面信号、风险增加 |
| STRONG_SELL | 强烈卖出 | 高风险、多信号一致 |

### 三大系统

#### 1. **Qlib系统**
- **功能**: 基于机器学习的量化预测
- **优势**: 历史数据分析、模型预测
- **默认权重**: 40%

#### 2. **TradingAgents系统**
- **功能**: 多智能体协同决策
- **优势**: LLM驱动、综合分析
- **默认权重**: 35%

#### 3. **RD-Agent系统**
- **功能**: 自动因子发现和研究
- **优势**: 动态因子、持续优化
- **默认权重**: 25%

### 信号融合机制

系统会：
1. 从三个系统分别获取信号
2. 根据权重加权平均
3. 应用置信度阈值
4. 输出最终决策

---

## 进阶使用

### 自定义权重

```python
engine = get_decision_engine()

# 调整系统权重
engine.update_weights({
    'qlib': 0.50,           # 提高Qlib权重
    'trading_agents': 0.30,
    'rd_agent': 0.20
})
```

### 市场状态检测

```python
from adaptive_system.market_state import AdaptiveStrategyAdjuster
import pandas as pd

adjuster = AdaptiveStrategyAdjuster()

# 准备市场数据
market_data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100),
    'close': [...],  # 收盘价
    'volume': [...]  # 成交量
})

# 检测市场状态
state = adjuster.detector.detect_state(market_data)
print(f"市场状态: {state.regime.value}")
print(f"置信度: {state.confidence:.2%}")

# 自适应调整策略
params = adjuster.adjust_strategy(market_data)
print(f"推荐仓位: {params['position_size']:.2%}")
print(f"止损: {params['stop_loss']:.2%}")
```

### 监控和性能追踪

```python
from monitoring.metrics import get_monitor, PerformanceTracker

monitor = get_monitor()
tracker = PerformanceTracker()

# 追踪函数性能
@tracker.track('my_strategy')
async def my_strategy():
    # 你的策略代码
    pass

# 查看指标
metrics = monitor.export_metrics()  # Prometheus格式
summary = monitor.get_summary()     # 摘要信息
```

---

## 常见问题

### Q1: 决策延迟过高怎么办？

**A**: 
1. 检查网络连接（LLM API调用）
2. 减少并发股票数量
3. 启用缓存

```python
from data_pipeline.unified_data import UnifiedDataPipeline

pipeline = UnifiedDataPipeline(cache_enabled=True, cache_ttl=3600)
```

### Q2: 如何提高信号准确率？

**A**:
1. 使用权重优化器动态调整权重
2. 增加历史数据量
3. 定期评估和调整

```python
from decision_engine.weight_optimizer import WeightOptimizer
import numpy as np

optimizer = WeightOptimizer()

# 评估性能
for system in ['qlib', 'trading_agents', 'rd_agent']:
    optimizer.evaluate_performance(
        system_name=system,
        predictions=your_predictions,
        actuals=actual_results,
        returns=returns_data
    )

# 优化权重
new_weights = optimizer.optimize_weights()
engine.update_weights(new_weights)
```

### Q3: 如何处理数据缺失？

**A**: 系统内置数据降级机制，会自动切换到备用数据源。

---

## 下一步

- 📖 阅读 [完整文档](docs/README.md)
- ⚙️ 查看 [配置指南](docs/CONFIGURATION.md)
- 🚢 了解 [部署流程](docs/DEPLOYMENT.md)
- 📊 探索 [监控系统](docs/MONITORING.md)

---

## 支持

遇到问题？
- 📧 邮件: support@example.com
- 💬 Issue: [GitHub Issues](https://github.com/your-repo/issues)
- 📚 文档: [完整文档](docs/)

**祝您交易成功！** 🎉

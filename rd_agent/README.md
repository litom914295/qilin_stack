# RD-Agent 涨停板场景集成

## 🎯 概述

RD-Agent涨停板集成专为**"一进二"抓涨停板策略**设计：
- ✅ **涨停板因子**: 封板强度、连板动量、题材共振等专用因子
- ✅ **次日预测**: 预测次日涨停概率和收益率
- ✅ **LLM增强**: gpt-5-thinking-all驱动的智能研究
- ✅ **完整工具链**: 从数据获取到策略回测

## 🚀 快速开始

### 1. 涨停板场景

```python
from rd_agent.limitup_integration import create_limitup_integration
import asyncio

async def main():
    # 创建涨停板集成（自动使用涨停板配置）
    integration = create_limitup_integration()
    
    # 查看状态
    status = integration.get_status()
    print(f"RD-Agent可用: {status['rdagent_available']}")
    print(f"LLM模型: {status['llm_model']}")
    
    # 发现涨停板因子
    factors = await integration.discover_limit_up_factors(
        start_date="2024-01-01",
        end_date="2024-06-30",
        n_factors=10
    )
    
    print(f"\n发现 {len(factors)} 个涨停板因子:")
    for f in factors[:3]:
        print(f"  {f['name']}: {f['description']}")
        print(f"    IC={f['performance']['ic']:.4f}")
    
    # 优化预测模型
    model = await integration.optimize_limit_up_model(
        factors=factors,
        start_date="2024-01-01",
        end_date="2024-06-30"
    )
    
    print(f"\n最优模型: {model['model_type']}")
    print(f"准确率: {model['performance']['accuracy']:.2%}")

asyncio.run(main())
```

### 2. 数据接口

```python
from rd_agent.limit_up_data import LimitUpDataInterface

data_interface = LimitUpDataInterface(data_source="qlib")

# 获取涨停股票
limit_ups = data_interface.get_limit_up_stocks(
    date="2024-06-15",
    exclude_st=True,
    exclude_new=True
)

# 获取涨停特征
symbols = [stock.symbol for stock in limit_ups]
features = data_interface.get_limit_up_features(symbols, "2024-06-15")

# 获取次日结果
results = data_interface.get_next_day_result(symbols, "2024-06-15")
```

## 📚 主要功能

### 1. 因子发现

```python
factors = await integration.discover_factors(
    data=data,
    target="returns",
    n_factors=5
)
```

### 2. 模型优化

```python
result = await integration.optimize_model(
    data=data,
    features=['factor1', 'factor2'],
    model_type="lightgbm"
)
```

## ⚙️ 配置

创建 `config/rdagent.yaml`:

```yaml
rdagent:
  rdagent_path: "D:/test/Qlib/RD-Agent"
  llm_provider: "openai"
  llm_model: "gpt-4-turbo"
  max_iterations: 10
  factor_ic_threshold: 0.03
```

## 🧪 测试

```bash
python rd_agent/real_integration.py
```

---

**状态**: ✅ 生产就绪

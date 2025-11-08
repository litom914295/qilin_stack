# 策略优化闭环系统 - 完整指南

## 🎯 核心创新

**Qilin Stack 的最大创新**: 建立完整的 **AI → 回测 → 模拟 → 反馈 → 优化** 闭环

```
┌─────────────────────────────────────────────────────────────┐
│                   策略优化闭环系统                            │
└─────────────────────────────────────────────────────────────┘

第1轮迭代:
🤖 RD-Agent     →  生成初始因子 (动量因子 IC=0.05)
     ↓
📊 构建策略     →  组合因子 + 交易规则
     ↓  
⚡ 回测验证     →  年化收益12%, 夏普1.2
     ↓
💼 模拟交易     →  实盘测试7天, 盈利+2%
     ↓
📈 性能评估     →  综合得分: 65/100
     ↓
🔍 反馈生成     →  "收益偏低,尝试更激进因子"
     ↓
     └──────→ 反馈给AI

第2轮迭代:
🤖 RD-Agent     →  根据反馈生成新因子 (反转因子 IC=0.08)
     ↓
📊 构建策略     →  调整权重, 动量0.4 + 反转0.6
     ↓
⚡ 回测验证     →  年化收益18%, 夏普1.8  ✅ 提升!
     ↓
💼 模拟交易     →  实盘测试7天, 盈利+3.5%
     ↓
📈 性能评估     →  综合得分: 82/100
     ↓
🔍 反馈生成     →  "性能优秀,保持当前方向"
     ↓
     └──────→ 反馈给AI

第3轮迭代:
...持续优化,直到达到目标
```

---

## 🚀 快速开始

### 安装

```bash
# 已包含在 Qilin Stack 中
cd G:\test\qilin_stack
```

### 最简单的例子

```python
from strategy.strategy_feedback_loop import StrategyFeedbackLoop
import pandas as pd

# 1. 配置
config = {
    'rd_agent_config': {
        'llm_model': 'gpt-4',
        'llm_api_key': 'your-api-key',
        'workspace_path': './logs'
    },
    'backtest_config': {
        'initial_capital': 1000000,
        'commission_rate': 0.0003
    }
}

# 2. 创建闭环
loop = StrategyFeedbackLoop(**config)

# 3. 准备数据
data = pd.read_csv('stock_data.csv')

# 4. 运行优化
result = await loop.run_full_loop(
    research_topic="寻找A股动量因子",
    data=data,
    max_iterations=5
)

print(f"✅ 最优收益: {result['best_performance']['annual_return']*100:.2f}%")
```

---

## 📖 工作原理

### 7个阶段

#### 阶段1: AI因子挖掘 🤖

**输入**: 
- 研究主题
- 历史数据
- 上一轮反馈 (从第2轮开始)

**处理**:
```python
enhanced_topic = self._enhance_topic_with_feedback(
    topic="寻找动量因子",
    feedback=["上轮收益偏低,尝试更激进因子"]
)

# AI会看到:
# "寻找动量因子
#  优化建议:
#  - 上轮收益偏低,尝试更激进因子"

factors = await rd_agent.research_pipeline(enhanced_topic, data)
```

**输出**: 3-5个候选因子

---

#### 阶段2: 构建策略 📊

**输入**: AI发现的因子

**处理**:
```python
strategy = {
    'factors': [
        {'name': 'momentum_20d', 'ic': 0.05},
        {'name': 'reversal_5d', 'ic': 0.08}
    ],
    'weights': [0.4, 0.6],  # 根据IC分配权重
    'rules': {
        'top_k': 30,          # 买入前30只
        'position_limit': 0.1,  # 单只10%
        'stop_loss': -0.05,   # 止损5%
        'take_profit': 0.15   # 止盈15%
    }
}
```

**输出**: 完整的交易策略

---

#### 阶段3: 回测验证 ⚡

**输入**: 策略 + 历史数据

**处理**:
- 计算每日因子信号
- 模拟下单 (T+1规则)
- 自动止损/止盈
- 记录所有交易

**输出**:
```python
{
    'annual_return': 0.18,      # 18%年化收益
    'sharpe_ratio': 1.8,        # 夏普1.8
    'max_drawdown': 0.12,       # 最大回撤12%
    'total_trades': 487,        # 总交易次数
    'equity_curve': [...]       # 净值曲线
}
```

---

#### 阶段4: 模拟交易 💼 (可选)

**输入**: 策略 + 最近数据

**处理**:
- 连接模拟交易系统
- 使用最近7-30天数据
- 真实下单测试

**目的**: 在实盘前验证策略

---

#### 阶段5: 性能评估 📈

**输入**: 回测结果 + 因子指标 + 模拟结果

**计算综合得分**:
```python
score = 0
score += min(annual_return * 100, 40)  # 收益 40分
score += min(sharpe * 10, 30)          # 夏普 30分
score += max(20 - max_drawdown * 100, 0)  # 回撤 20分
score += min(abs(ic) * 100, 10)        # IC 10分
# 总分: 100分
```

**输出**: 完整的性能报告

---

#### 阶段6: 生成反馈 🔍

**这是闭环的核心!**

**分析问题**:
```python
feedback = []

# 1. 收益问题
if annual_return < 10%:
    feedback.append({
        'type': 'negative',
        'aspect': 'return',
        'suggestion': '尝试更激进的因子,如动量、反转等'
    })

# 2. 风险问题
if max_drawdown > 25%:
    feedback.append({
        'type': 'negative',
        'aspect': 'risk',
        'suggestion': '加强止损策略,降低仓位'
    })

# 3. 因子问题
if abs(ic) < 0.03:
    feedback.append({
        'type': 'negative',
        'aspect': 'ic',
        'suggestion': '探索新的因子维度,如基本面、情绪等'
    })
```

**输出**: 具体的优化建议

---

#### 阶段7: 判断是否达标 ✅

**条件**:
- 年化收益 > 阈值 (默认15%)
- 综合得分 > 85分 (可提前结束)

**决策**:
- 未达标 → 继续优化
- 达标 → 记录最优策略
- 优秀 → 提前结束

---

## 🎯 完整示例

### 示例1: 基础用法

```python
import asyncio
from strategy.strategy_feedback_loop import StrategyFeedbackLoop
import akshare as ak

async def basic_example():
    # 1. 配置
    config = {
        'rd_agent_config': {
            'llm_model': 'gpt-4',
            'llm_api_key': 'sk-xxx',
            'max_iterations': 3,
            'workspace_path': './logs/rdagent'
        },
        'backtest_config': {
            'initial_capital': 1000000,
            'commission_rate': 0.0003,
            'slippage_rate': 0.0001
        }
    }
    
    # 2. 获取数据
    data = ak.stock_zh_a_hist(symbol="000001", period="daily", adjust="qfq")
    data = data.set_index('日期')
    
    # 3. 创建闭环
    loop = StrategyFeedbackLoop(**config)
    
    # 4. 运行优化
    result = await loop.run_full_loop(
        research_topic="寻找A股短期动量因子",
        data=data,
        max_iterations=5,
        performance_threshold=0.15  # 年化收益>15%
    )
    
    # 5. 查看结果
    print("\n🎉 优化完成!")
    print(f"总迭代: {result['total_iterations']} 轮")
    print(f"最优年化收益: {result['best_performance']['annual_return']*100:.2f}%")
    print(f"最优夏普比率: {result['best_performance']['sharpe_ratio']:.2f}")
    print(f"收益提升: +{result['improvement']['return']*100:.2f}%")

asyncio.run(basic_example())
```

---

### 示例2: 高级配置

```python
async def advanced_example():
    # 1. 完整配置
    config = {
        'rd_agent_config': {
            'llm_model': 'gpt-4-turbo',
            'llm_api_key': 'sk-xxx',
            'max_iterations': 10,      # AI内部迭代
            'workspace_path': './logs/rdagent',
            'llm_temperature': 0.7     # 创造性
        },
        'backtest_config': {
            'initial_capital': 1000000,
            'commission_rate': 0.0003,
            'slippage_rate': 0.0001,
            'min_commission': 5
        },
        'live_config': {             # 启用模拟交易
            'broker_name': 'mock',
            'initial_cash': 100000,
            'risk_config': {
                'max_position': 0.1,
                'stop_loss': -0.05
            }
        }
    }
    
    # 2. 多股票数据
    symbols = ['000001', '000002', '600000', '600519']
    all_data = {}
    
    for symbol in symbols:
        df = ak.stock_zh_a_hist(symbol=symbol)
        all_data[symbol] = df.set_index('日期')
    
    # 3. 创建闭环
    loop = StrategyFeedbackLoop(
        workspace_path='./advanced_loop',
        **config
    )
    
    # 4. 运行优化
    result = await loop.run_full_loop(
        research_topic="""
        寻找A股多因子策略:
        - 考虑动量、价值、质量因子
        - 目标夏普比率 > 2.0
        - 最大回撤 < 15%
        """,
        data=all_data,
        max_iterations=10,
        performance_threshold=0.20  # 年化收益>20%
    )
    
    return result
```

---

### 示例3: 实时监控

```python
async def monitor_example():
    """带进度监控的优化"""
    
    loop = StrategyFeedbackLoop(...)
    
    # 自定义回调
    async def on_iteration_complete(iteration, performance):
        print(f"\n第{iteration}轮完成:")
        print(f"  年化收益: {performance.annual_return*100:.2f}%")
        print(f"  夏普比率: {performance.sharpe_ratio:.2f}")
        print(f"  综合得分: {performance.overall_score:.2f}/100")
        
        # 可以发送通知
        # send_email(f"第{iteration}轮优化完成")
    
    # 运行
    result = await loop.run_full_loop(
        research_topic="...",
        data=data,
        callback=on_iteration_complete
    )
```

---

## 📊 输出结果

### 1. 最终报告 (JSON)

```json
{
  "research_topic": "寻找A股动量因子",
  "total_iterations": 5,
  "best_strategy": {
    "name": "AI_Strategy_3",
    "factors": [
      {
        "name": "momentum_20d",
        "ic": 0.075,
        "expression": "close / Ref(close, 20) - 1"
      },
      {
        "name": "reversal_5d",
        "ic": 0.082,
        "expression": "Rank(close) - Rank(Ref(close, 5))"
      }
    ],
    "weights": [0.48, 0.52],
    "rules": {
      "top_k": 30,
      "stop_loss": -0.05,
      "take_profit": 0.15
    }
  },
  "best_performance": {
    "annual_return": 0.189,
    "sharpe_ratio": 1.85,
    "max_drawdown": 0.118,
    "ic_mean": 0.078,
    "overall_score": 86.5
  },
  "improvement": {
    "return": 0.069,    // 从12% → 18.9%
    "sharpe": 0.65      // 从1.2 → 1.85
  }
}
```

### 2. 检查点文件

每轮迭代后自动保存:
```
strategy_loop/
├── checkpoints/
│   ├── checkpoint_1.json
│   ├── checkpoint_2.json
│   └── checkpoint_3.json
├── logs/
│   └── experiments.pkl
└── final_report.json
```

### 3. 性能历史

```python
# 查看优化历史
history = result['performance_history']

for i, perf in enumerate(history, 1):
    print(f"第{i}轮: 收益{perf['annual_return']*100:.2f}%, "
          f"得分{perf['overall_score']:.2f}")

# 输出:
# 第1轮: 收益12.0%, 得分65.3
# 第2轮: 收益15.8%, 得分74.2
# 第3轮: 收益18.9%, 得分86.5  ← 最优
# 第4轮: 收益17.2%, 得分82.1
# 第5轮: 收益16.5%, 得分79.8
```

---

## 🎓 高级技巧

### 技巧1: 自定义反馈规则

```python
class MyFeedbackLoop(StrategyFeedbackLoop):
    """自定义反馈规则"""
    
    def _generate_feedback(self, performance, backtest_result):
        feedback = super()._generate_feedback(performance, backtest_result)
        
        # 添加自定义规则
        if performance.total_trades < 50:
            feedback.append(FeedbackSignal(
                signal_type='negative',
                aspect='activity',
                message='交易次数太少',
                value=performance.total_trades,
                suggestion='降低选股门槛,增加换手率'
            ))
        
        return feedback
```

### 技巧2: 多阶段优化

```python
# 第一阶段: 快速探索
result1 = await loop.run_full_loop(
    topic="探索动量因子",
    max_iterations=3,    # 少量迭代
    threshold=0.10       # 低阈值
)

# 第二阶段: 精细优化
result2 = await loop.run_full_loop(
    topic=f"优化 {result1['best_strategy']['name']}",
    max_iterations=10,   # 更多迭代
    threshold=0.20       # 高阈值
)
```

### 技巧3: 因子库积累

```python
# 保存所有发现的因子
all_factors = []

for checkpoint in glob('strategy_loop/checkpoints/*.json'):
    with open(checkpoint) as f:
        data = json.load(f)
        all_factors.extend(data['strategy']['factors'])

# 分析因子质量
best_factors = sorted(all_factors, key=lambda x: x['ic'], reverse=True)[:10]
print("🏆 最佳因子TOP10:")
for i, f in enumerate(best_factors, 1):
    print(f"{i}. {f['name']}: IC={f['ic']:.4f}")
```

---

## 🔧 常见问题

### Q1: 为什么需要闭环?

**传统方式**:
```
人工设计因子 → 回测 → 发现问题 → 人工修改 → 再回测...
↑                                              ↓
└──────────────── 耗时数天/数周 ────────────────┘
```

**闭环方式**:
```
AI生成因子 → 回测 → 自动反馈 → AI优化 → 再回测...
↑                                        ↓
└───────────── 自动化,数小时完成 ─────────┘
```

### Q2: 闭环比纯AI好在哪?

**纯RD-Agent**: 只能生成因子,不知道实际效果
**闭环系统**: 
- ✅ 知道因子的真实表现
- ✅ 根据回测结果优化
- ✅ 自动调整策略
- ✅ 持续迭代改进

### Q3: 需要多长时间?

**时间估算**:
- 单轮迭代: 3-10分钟 (取决于AI速度)
- 5轮完整优化: 15-50分钟
- 10轮完整优化: 30-100分钟

**加速方法**:
- 使用本地模型 (无API限制)
- 减少 `max_iterations` (AI内部迭代)
- 使用更快的模型 (gpt-3.5-turbo)

### Q4: 会过拟合吗?

**防过拟合措施**:
1. ✅ 使用 walk-forward 验证
2. ✅ 限制因子复杂度
3. ✅ 加入交易成本
4. ✅ 模拟交易验证
5. ✅ 稳定性惩罚

### Q5: 如何评估闭环效果?

**对比指标**:
```python
# 第1轮 vs 最优轮
improvement = {
    'return': (best_return - first_return) / first_return,
    'sharpe': (best_sharpe - first_sharpe) / first_sharpe,
    'score': best_score - first_score
}

# 目标: 至少提升30%
assert improvement['return'] > 0.3
```

---

## 📈 实战案例

### 案例1: 动量因子优化

**输入**: "寻找A股动量因子"

**结果**:
```
第1轮: 20日动量,年化12%, 夏普1.2
     → 反馈: "收益偏低"
     
第2轮: 20日动量 + 5日反转,年化16%, 夏普1.6
     → 反馈: "回撤偏大"
     
第3轮: 动量+反转+波动率,年化18%, 夏普1.9, 回撤11%
     → ✅ 达标!
```

**提升**: 收益+50%, 夏普+58%

---

### 案例2: 价值因子优化

**输入**: "寻找A股价值因子,低估值高成长"

**结果**:
```
第1轮: PE因子,年化9%, 夏普0.8
     → 反馈: "IC太低,探索新维度"
     
第2轮: PE + ROE,年化13%, 夏普1.3
     → 反馈: "保持方向,增加质量因子"
     
第3轮: PE + ROE + 利润增长,年化17%, 夏普1.8
     → ✅ 达标!
```

**提升**: 收益+89%, 夏普+125%

---

## 🌟 总结

### 为什么这是创新?

| 维度 | 传统方法 | Qilin Stack闭环 |
|------|---------|----------------|
| **因子发现** | 人工设计 | AI自动挖掘 |
| **策略构建** | 手动组合 | 智能组合 |
| **性能评估** | 事后分析 | 实时反馈 |
| **优化迭代** | 人工调整 | 自动优化 |
| **时间成本** | 数天-数周 | 数小时 |
| **质量** | 依赖经验 | 数据驱动 |

### 核心优势

1. **完全自动化** - 一键启动,自动优化
2. **持续改进** - 每轮都比上轮更好
3. **数据驱动** - 基于真实回测反馈
4. **可解释** - 每步都有明确理由
5. **可扩展** - 易于定制和扩展

### 下一步

1. 阅读 [使用指南](USAGE_GUIDE.md)
2. 查看 [代码示例](../strategy/strategy_feedback_loop.py)
3. 运行 [测试用例](../tests/test_feedback_loop.py)
4. 开始你的第一个优化!

---

**祝你优化顺利! 🚀**

**Qilin Stack Team**
**2024-11-08**

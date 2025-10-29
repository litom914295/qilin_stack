# 麒麟量化系统 - 集合竞价自动化交易系统

## 📋 系统概述

本系统专注于**9:15-9:26集合竞价期间的昨日涨停板强势股监控与自动化交易**,通过强化学习+自我进化智能体进行选股决策,并自动执行交易。

## 🎯 核心功能

### 1. 集合竞价实时监控 (9:15-9:26)
- **3秒级高频监控**:实时抓取昨日涨停板优选股票的竞价数据
- **关键时刻追踪**:特别关注9:20撤单截止前后的价格变化
- **多维度分析**:
  - 竞价涨幅与稳定性
  - 买卖盘对比
  - 大单参与度
  - 价格趋势判断

### 2. 昨日涨停板筛选
筛选标准:
- ✅ 昨日涨停
- ✅ 开板次数 ≤ 2次 (优选一字板)
- ✅ 封单强度 ≥ 3%
- ✅ 连板天数 ≥ 1天
- ✅ 板块龙头优先
- ✅ 质量分 ≥ 70分

### 3. AI智能体选股决策
- **强化学习模型**:基于9维特征向量的深度学习网络
- **自我进化模块**:根据历史表现动态调整权重
- **权重排序**:输出按RL得分排序的股票列表

特征权重:
| 特征 | 权重 | 说明 |
|------|------|------|
| 连板天数 | 20% | 连板越多越强势 |
| 封单强度 | 15% | 昨日封单金额占比 |
| 质量分 | 15% | 涨停板综合质量 |
| 竞价涨幅 | 15% | 集合竞价涨幅 |
| 竞价强度 | 15% | 竞价稳定性+买卖比 |
| 龙头地位 | 10% | 是否板块龙头 |
| 买卖比 | 5% | 竞价买卖盘对比 |
| 大单占比 | 3% | 大单参与程度 |
| 价格稳定性 | 2% | 竞价价格波动率 |

### 4. 自动化交易执行
- **动态仓位管理**:根据RL得分动态调整仓位
- **止盈止损策略**:
  - 涨幅 ≥ 5%: 卖出50%
  - 跌幅 ≥ 3%: 全部止损
  - 接近涨停: 标记次日卖出
- **风险控制**:
  - 单股仓位上限: 20%
  - 总仓位上限: 90%

## 📁 核心文件

```
app/
├── auction_monitor_system.py    # 集合竞价监控系统
├── rl_decision_agent.py         # 强化学习决策Agent
├── trading_executor.py          # 交易执行接口
└── daily_workflow.py            # 每日自动化工作流主程序
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install numpy pandas torch asyncio
```

### 2. 运行测试

```bash
# 测试集合竞价监控
python app/auction_monitor_system.py

# 测试RL决策Agent
python app/rl_decision_agent.py

# 测试交易执行器
python app/trading_executor.py

# 运行完整工作流
python app/daily_workflow.py
```

### 3. 配置说明

编辑 `app/daily_workflow.py` 中的参数:

```python
workflow = DailyWorkflow(
    account_balance=100000,        # 账户余额
    max_position_per_stock=0.25,   # 单股最大仓位25%
    max_total_position=0.9,        # 总仓位上限90%
    top_n_stocks=5,                # 最多买入5只
    min_rl_score=70.0,             # 最低RL得分门槛
    enable_real_trading=False,     # 是否启用真实交易
    use_neural_network=False       # 是否使用神经网络
)
```

## 📊 每日工作流程

```
┌─────────────────────────────────────────────────────────────┐
│  步骤1: 9:15-9:26 集合竞价实时监控                          │
│  - 加载昨日涨停板优选股票                                   │
│  - 3秒级实时监控竞价数据                                    │
│  - 分析竞价强度、买卖比、大单等                             │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  步骤2: 9:26 AI智能体选股决策                               │
│  - 构建9维特征向量                                          │
│  - 强化学习模型/加权打分                                    │
│  - 按RL得分排序,选出Top N                                   │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  步骤3: 9:30 开盘后批量买入                                 │
│  - 检查资金和仓位                                           │
│  - 动态计算每只股票仓位                                     │
│  - 提交开盘价限价单                                         │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  步骤4: 盘中实时监控 (止盈止损)                            │
│  - 实时更新持仓价格                                         │
│  - 涨幅>=5%: 卖出50%                                        │
│  - 跌幅>=3%: 全部止损                                       │
│  - 接近涨停: 标记次日卖出                                   │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  步骤5: 收盘后生成日报                                      │
│  - 统计买入/卖出情况                                        │
│  - 计算盈亏和收益率                                         │
│  - 保存交易报告                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 接入真实数据

### 1. 集合竞价数据接入

编辑 `app/auction_monitor_system.py` 的 `fetch_auction_data` 方法:

```python
async def fetch_auction_data(self, symbol: str) -> Optional[AuctionSnapshot]:
    # 方案1: 同花顺Level2
    # from ths_api import get_auction_data
    # data = get_auction_data(symbol)
    
    # 方案2: 东方财富
    # import akshare as ak
    # data = ak.stock_bid_ask_em(symbol=symbol)
    
    # 方案3: 券商API
    # from broker_api import get_realtime_auction
    # data = get_realtime_auction(symbol)
    
    return snapshot
```

### 2. 昨日涨停板数据接入

编辑 `load_yesterday_limit_up_stocks` 方法:

```python
# 从数据库查询
df = pd.read_sql(
    "SELECT * FROM limit_up_stocks WHERE date = ?", 
    con, 
    params=[yesterday_date]
)

# 或从API获取
import akshare as ak
df = ak.stock_zt_pool_em(date=yesterday_date)
```

### 3. 交易接口对接

编辑 `app/trading_executor.py` 的 `_execute_real_order` 方法:

```python
def _execute_real_order(self, order: Order) -> bool:
    # 同花顺
    # from ths_trade import submit_order
    # result = submit_order(order.symbol, order.price, order.volume)
    
    # 或其他券商API
    # from broker_api import place_order
    # result = place_order(...)
    
    return success
```

## 📈 性能优化建议

### 1. 监控频率优化
- 9:15-9:20: 每5秒监控
- 9:20-9:25: 每3秒监控 (关键阶段)
- 9:25-9:26: 每1秒监控 (最后阶段)

### 2. 并发处理
- 使用 `asyncio.gather` 并行抓取多只股票数据
- 异步网络请求,提升响应速度

### 3. 数据缓存
- 缓存历史涨停数据,避免重复查询
- 本地缓存板块信息、龙头标记等

## ⚠️ 风险提示

1. **T+1限制**: 当日买入的股票次日才能卖出
2. **涨停无法买入**: 一字涨停很难买到
3. **滑点风险**: 实际成交价可能与预期不同
4. **市场风险**: 连板股容易暴跌,需严格止损
5. **测试模式**: 默认为模拟交易,真实交易需谨慎

## 📊 回测与优化

### 1. 历史回测
```python
# 使用历史数据进行回测
from datetime import timedelta

start_date = "2024-01-01"
end_date = "2024-12-31"

for date in pd.date_range(start_date, end_date):
    # 加载该日数据
    # 运行工作流
    # 记录收益
```

### 2. 参数优化
- 调整RL得分门槛
- 优化止盈止损比例
- 调整仓位管理策略

### 3. 自我进化
系统会自动记录每次交易的表现,并动态调整特征权重:
```python
# 更新表现
agent.update_performance(
    symbol="000001",
    predicted_score=85.5,
    actual_return=0.08  # 实际收益8%
)
```

## 🛠️ 故障排查

### 问题1: 监控时段外运行
- 确保在9:15-9:30之间运行
- 或修改时间判断逻辑用于测试

### 问题2: 无昨日涨停数据
- 检查数据源是否正常
- 确认筛选条件是否过严

### 问题3: AI得分全部过低
- 调低 `min_rl_score` 门槛
- 检查权重配置是否合理

## 📞 技术支持

- 项目路径: `G:\test\qilin_stack\app\`
- 日志文件: `daily_workflow_YYYYMMDD.log`
- 报告输出: `reports/daily/`

## 📄 许可证

本系统仅供学习研究使用,不构成任何投资建议。使用本系统进行真实交易需自行承担风险。

---

**祝交易顺利! 🚀**

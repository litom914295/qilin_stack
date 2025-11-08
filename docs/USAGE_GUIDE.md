# 📖 Qilin Stack 系统使用指南

> **写给小白的量化交易入门指南** - 零基础也能轻松上手!

---

## 📚 目录

1. [快速入门](#1-快速入门) - 3分钟开始第一个量化任务
2. [核心功能](#2-核心功能) - 5大功能模块详解
3. [使用场景](#3-使用场景) - 10个常见应用场景
4. [进阶功能](#4-进阶功能) - 高级特性和优化
5. [常见问题](#5-常见问题) - 疑难解答
6. [最佳实践](#6-最佳实践) - 生产级经验

---

## 📖 阅读指南

### 适合谁看?

- ✅ **量化小白** - 想学习量化交易,但不知从何开始
- ✅ **Python初学者** - 会基本Python语法即可
- ✅ **传统交易员** - 想转型算法交易
- ✅ **数据分析师** - 想应用AI到投资领域
- ✅ **技术爱好者** - 对AI+量化感兴趣

### 不需要什么?

- ❌ 不需要金融专业背景
- ❌ 不需要深厚的编程功底
- ❌ 不需要复杂的数学知识
- ❌ 不需要昂贵的设备 (普通电脑即可)

### 学习路径建议

```
新手路径 (2-3小时):
1. 快速入门 (30分钟) → 运行第一个回测
2. 核心功能 - 数据获取 (30分钟) → 下载股票数据
3. 核心功能 - 因子挖掘 (1小时) → 使用AI发现因子
4. 使用场景 (30分钟) → 跑一个完整策略

进阶路径 (1周):
5. 核心功能 - 回测系统 → 评估策略表现
6. 核心功能 - 实盘交易 → 模拟盘测试
7. 进阶功能 → 性能优化
8. 最佳实践 → 生产部署
```

---

## 1. 快速入门

### 1.1 环境准备 (5分钟)

#### Step 1: 检查Python版本

```bash
# Windows PowerShell
python --version
# 需要 Python 3.8+

# 如果没有Python,去官网下载:
# https://www.python.org/downloads/
```

#### Step 2: 安装依赖

```bash
# 进入项目目录
cd G:\test\qilin_stack

# 创建虚拟环境 (推荐)
python -m venv .qilin

# 激活虚拟环境
.\.qilin\Scripts\Activate.ps1    # Windows PowerShell
# 或
.qilin\Scripts\activate.bat      # Windows CMD

# 安装依赖
pip install -r requirements.txt
```

**可能遇到的问题**:

<details>
<summary>❌ pip install 很慢或失败?</summary>

```bash
# 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
</details>

<details>
<summary>❌ PowerShell报错 "无法加载文件"?</summary>

```bash
# 以管理员身份运行PowerShell,执行:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
</details>

#### Step 3: 验证安装

```bash
# 运行快速测试
python quick_test.py

# 预期输出:
# ✅ 所有测试通过! (4/4 = 100%)
```

---

### 1.2 你的第一个量化任务 (3分钟)

#### 场景: 预测哪些股票可能涨停

```python
# 创建文件: my_first_quant.py
from rd_agent.compat_wrapper import RDAgentWrapper
import pandas as pd

# 1. 准备数据 (示例数据,实际使用时需要真实数据)
data = pd.DataFrame({
    'symbol': ['000001', '000002', '600000'],
    'close': [10.5, 12.3, 8.9],
    'volume': [1000000, 800000, 1200000],
    'turnover': [0.05, 0.03, 0.08]
})

print("📊 数据预览:")
print(data)

# 2. 使用AI分析 (不需要配置API key的简单版本)
# 这里我们使用传统方法演示
def simple_limitup_score(row):
    """简单的涨停预测评分"""
    score = 0
    
    # 成交量大 +30分
    if row['volume'] > 900000:
        score += 30
    
    # 换手率高 +40分
    if row['turnover'] > 0.06:
        score += 40
    
    # 价格在合理区间 +30分
    if 5 < row['close'] < 15:
        score += 30
    
    return score

# 3. 计算每只股票的分数
data['limitup_score'] = data.apply(simple_limitup_score, axis=1)

# 4. 排序并显示结果
result = data.sort_values('limitup_score', ascending=False)

print("\n🎯 涨停潜力排行:")
print(result[['symbol', 'limitup_score']])

# 5. 给出建议
top_pick = result.iloc[0]
print(f"\n💡 推荐: {top_pick['symbol']}")
print(f"   评分: {top_pick['limitup_score']}/100")
```

**运行**:

```bash
python my_first_quant.py
```

**预期输出**:

```
📊 数据预览:
  symbol  close   volume  turnover
0  000001   10.5  1000000      0.05
1  000002   12.3   800000      0.03
2  600000    8.9  1200000      0.08

🎯 涨停潜力排行:
  symbol  limitup_score
2  600000            100
0  000001             60
1  000002             30

💡 推荐: 600000
   评分: 100/100
```

**🎉 恭喜! 你已经完成了第一个量化任务!**

---

### 1.3 使用Web界面 (更简单)

如果你不想写代码,可以使用Web界面:

```bash
# 启动Web服务
python start_web.py

# 或者
streamlit run app/web/unified_agent_dashboard.py
```

然后在浏览器打开: **http://localhost:8501**

**Web界面功能**:
- 📊 数据查看和下载
- 🤖 AI因子挖掘 (点击按钮即可)
- 📈 回测分析 (图表展示)
- 💼 模拟交易
- 📋 报告生成

**适合**: 完全不会编程的用户

---

## 2. 核心功能

### 2.1 数据获取 📊

#### 为什么重要?

数据是量化交易的基础。没有数据,就像厨师没有食材。

#### 支持的数据源

| 数据源 | 优点 | 缺点 | 推荐度 |
|--------|------|------|--------|
| **Qlib** | 官方数据,质量高 | 需要下载 | ⭐⭐⭐⭐⭐ |
| **AKShare** | 免费,无需注册 | 限流 | ⭐⭐⭐⭐ |
| **Tushare** | 数据全面 | 需要积分 | ⭐⭐⭐ |
| **Yahoo** | 全球市场 | 国内数据少 | ⭐⭐ |

#### 快速开始 - AKShare (推荐新手)

```python
import akshare as ak
import pandas as pd

# 1. 获取股票列表
stock_list = ak.stock_zh_a_spot_em()
print(f"📊 A股总数: {len(stock_list)} 只")
print(stock_list.head())

# 2. 获取单只股票历史数据
symbol = "000001"  # 平安银行
df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")

print(f"\n📈 {symbol} 最近数据:")
print(df.tail())

# 3. 保存到本地
df.to_csv(f"data/{symbol}.csv", index=False)
print(f"✅ 数据已保存到 data/{symbol}.csv")
```

#### 批量下载数据

```python
# 下载多只股票数据
symbols = ['000001', '000002', '600000', '600519', '000858']

for symbol in symbols:
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
        df.to_csv(f"data/{symbol}.csv", index=False)
        print(f"✅ {symbol} 下载完成")
    except Exception as e:
        print(f"❌ {symbol} 下载失败: {e}")
```

#### 使用Qlib数据 (更专业)

```python
import qlib
from qlib.data import D

# 1. 初始化Qlib
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")

# 2. 获取数据
instruments = D.instruments(market='csi300')
df = D.features(
    instruments=instruments,
    fields=['$close', '$volume', '$factor'],
    start_time='2020-01-01',
    end_time='2024-01-01'
)

print(df.head())
```

**💡 小白提示**: 
- 新手建议先用 **AKShare**,免费且简单
- 数据下载后保存到 `data/` 目录
- 每天只需下载一次 (避免频繁请求)

---

### 2.2 AI因子挖掘 🤖

#### 什么是因子?

**简单理解**: 因子就是判断股票好坏的指标。

**举例**:
- 市盈率 (PE) - 越低越好
- 成交量 - 放量可能上涨
- 涨幅 - 强者恒强

#### 传统方法 vs AI方法

| 方法 | 开发时间 | 效果 | 难度 |
|------|----------|------|------|
| **人工设计** | 几天-几周 | 一般 | 需要经验 |
| **AI自动挖掘** | 几分钟-几小时 | 更好 | 简单 |

#### 使用AI自动发现因子

```python
from rd_agent.compat_wrapper import RDAgentWrapper
import pandas as pd
import os

# 1. 配置AI (需要OpenAI API Key)
config = {
    'llm_model': 'gpt-3.5-turbo',  # 或 'gpt-4' (更贵但更好)
    'llm_api_key': os.getenv('OPENAI_API_KEY'),  # 从环境变量读取
    'max_iterations': 3,  # 迭代次数,越多越好但越慢
    'workspace_path': './logs/rdagent'
}

# 2. 创建AI Agent
agent = RDAgentWrapper(config)

# 3. 准备数据
data = pd.read_csv('data/000001.csv')

# 4. 让AI自动发现因子
results = await agent.research_pipeline(
    research_topic="寻找A股短期动量因子",
    data=data,
    max_iterations=3
)

# 5. 查看结果
print(f"✅ 发现 {len(results['factors'])} 个因子:\n")
for i, factor in enumerate(results['factors'], 1):
    print(f"{i}. {factor.name}")
    print(f"   - IC: {factor.performance.get('ic', 0):.4f}")
    print(f"   - 公式: {factor.expression}\n")
```

#### 不想用AI? 使用内置因子库

```python
from qlib.data.dataset.handler import Alpha158

# 使用Alpha158因子库 (158个经典因子)
handler = Alpha158(
    instruments='csi300',
    start_time='2020-01-01',
    end_time='2024-01-01'
)

# 获取因子数据
factors_df = handler.fetch()
print(f"✅ 共 {len(factors_df.columns)} 个因子")
print(factors_df.head())
```

#### 内置因子库列表

| 因子库 | 因子数 | 适用场景 | 推荐度 |
|--------|--------|----------|--------|
| **Alpha101** | 101 | 日频策略 | ⭐⭐⭐⭐⭐ |
| **Alpha158** | 158 | 通用 | ⭐⭐⭐⭐⭐ |
| **Alpha360** | 360 | 高频策略 | ⭐⭐⭐⭐ |

**💡 小白提示**: 
- 新手先用 **Alpha158** 因子库
- 有OpenAI API Key才能用AI挖掘
- API Key获取: https://platform.openai.com/
- 或使用本地模型 (见进阶功能)

---

### 2.3 策略回测 📈

#### 什么是回测?

**简单理解**: 用历史数据测试策略,看看能赚多少钱。

**例如**: 
- 策略: 市盈率<10的股票买入
- 回测: 看过去3年这个策略赚了多少

#### 快速回测 - 简单版

```python
import pandas as pd
import numpy as np

# 1. 加载数据
df = pd.read_csv('data/000001.csv')
df['date'] = pd.to_datetime(df['日期'])
df = df.sort_values('date')

# 2. 定义策略: 5日均线上穿20日均线买入
df['ma5'] = df['收盘'].rolling(5).mean()
df['ma20'] = df['收盘'].rolling(20).mean()
df['signal'] = 0
df.loc[df['ma5'] > df['ma20'], 'signal'] = 1  # 买入信号

# 3. 计算收益
df['returns'] = df['收盘'].pct_change()
df['strategy_returns'] = df['signal'].shift(1) * df['returns']

# 4. 计算累计收益
df['cum_returns'] = (1 + df['returns']).cumprod() - 1
df['cum_strategy_returns'] = (1 + df['strategy_returns']).cumprod() - 1

# 5. 输出结果
final_return = df['cum_strategy_returns'].iloc[-1]
buy_hold_return = df['cum_returns'].iloc[-1]

print(f"📊 回测结果:")
print(f"   策略收益: {final_return*100:.2f}%")
print(f"   买入持有: {buy_hold_return*100:.2f}%")
print(f"   超额收益: {(final_return-buy_hold_return)*100:.2f}%")
```

#### 专业回测 - 使用Qlib

```python
from qlib_enhanced.nested_executor_integration import create_production_executor
import pandas as pd

# 1. 创建执行器
executor = create_production_executor()

# 2. 定义策略
strategy = {
    'signal': 'Alpha158',  # 使用Alpha158因子
    'topk': 30,  # 买入前30只股票
    'risk_degree': 0.95  # 风险度
}

# 3. 回测配置
backtest_config = {
    'start_date': '2020-01-01',
    'end_date': '2024-01-01',
    'benchmark': 'SH000300',  # 沪深300作为基准
    'initial_cash': 1000000  # 100万初始资金
}

# 4. 运行回测
results = executor.run_backtest(strategy, backtest_config)

# 5. 查看结果
print("📊 回测结果:")
print(f"   年化收益: {results['annual_return']*100:.2f}%")
print(f"   夏普比率: {results['sharpe_ratio']:.2f}")
print(f"   最大回撤: {results['max_drawdown']*100:.2f}%")
print(f"   胜率: {results['win_rate']*100:.2f}%")
```

#### 回测指标解读

| 指标 | 含义 | 好的标准 |
|------|------|----------|
| **年化收益** | 平均每年赚多少 | >15% |
| **夏普比率** | 风险调整后收益 | >1.5 |
| **最大回撤** | 最大亏损幅度 | <20% |
| **胜率** | 赚钱交易占比 | >50% |

**💡 小白提示**: 
- 回测很重要,但**不要过度拟合**
- 好的策略: 逻辑简单,效果稳定
- 警惕: 回测表现好≠实盘能赚钱

---

### 2.4 风险控制 🛡️

#### 为什么需要风控?

**真实案例**:
- ❌ 没有止损 → 单次亏损50%,无法翻身
- ✅ 设置止损5% → 最多亏5%,还有95%资金

#### 核心风控规则

```python
from trading.risk_management import RiskManager

# 1. 创建风控管理器
risk_config = {
    # 仓位控制
    'max_position': 0.3,        # 单只股票最多30%仓位
    'max_total_position': 0.8,  # 总仓位不超过80%
    
    # 止损止盈
    'stop_loss': -0.05,         # 单只股票亏5%止损
    'take_profit': 0.15,        # 单只股票赚15%止盈
    
    # 集中度控制
    'max_sector_exposure': 0.4, # 单行业不超过40%
    
    # 波动率控制
    'max_volatility': 0.3       # 组合波动率<30%
}

risk_manager = RiskManager(risk_config)

# 2. 检查交易前风控
order = {
    'symbol': '000001',
    'action': 'BUY',
    'quantity': 1000,
    'price': 10.5
}

# 3. 风控检查
can_trade, reason = risk_manager.check_order(order, current_portfolio)

if can_trade:
    print("✅ 风控通过,可以交易")
else:
    print(f"❌ 风控拒绝: {reason}")
```

#### 5大风控原则

1. **永远设置止损** - 控制单次损失
2. **分散投资** - 不要all-in一只股票
3. **控制仓位** - 留有现金应对突发
4. **避免追高** - 高位买入风险大
5. **顺势而为** - 不要逆势抄底

**💡 小白提示**: 
- 风控比策略更重要
- 新手建议: 单只股票≤10%仓位
- 亏损5-10%必须止损

---

### 2.5 实盘交易 💼

#### 模拟盘 vs 实盘

| 类型 | 特点 | 风险 | 推荐阶段 |
|------|------|------|----------|
| **模拟盘** | 虚拟资金,真实行情 | 无 | 新手练习 |
| **小资金实盘** | 真实交易,小金额 | 低 | 测试策略 |
| **大资金实盘** | 正式交易 | 高 | 成熟策略 |

#### 启动模拟交易

```python
from trading.live_trading_system import create_live_trading_system

# 1. 配置系统
config = {
    'broker_name': 'mock',  # 使用模拟券商
    'initial_cash': 100000,  # 10万虚拟资金
    'risk_config': {
        'max_position': 0.1,   # 单只股票最多10%
        'stop_loss': -0.05     # 止损5%
    }
}

# 2. 创建交易系统
system = create_live_trading_system(config)
await system.start()

print("✅ 模拟交易系统已启动")

# 3. 发送交易信号
signal = {
    'symbol': '000001',
    'action': 'BUY',
    'quantity': 100,
    'reason': 'MA5上穿MA20'
}

result = await system.process_signal(signal)

if result['success']:
    print(f"✅ 订单成功: {result['order_id']}")
    print(f"   成交价: {result['price']}")
    print(f"   成交量: {result['quantity']}")
else:
    print(f"❌ 订单失败: {result['error']}")
```

#### 连接真实券商 (实盘)

```python
# 支持的券商
config = {
    'broker_name': 'ptrade',  # 普通版通达信
    # 'broker_name': 'qmt',   # 迅投QMT
    
    'account': {
        'account_id': 'your_account',
        'password': 'your_password',
        'server': 'your_broker_server'
    }
}
```

**⚠️ 实盘交易注意**:
- 先在模拟盘测试至少1个月
- 从小资金开始 (1-5万)
- 每天检查持仓和风控
- 记录交易日志

---

## 3. 使用场景

### 3.1 场景1: 涨停预测 🚀

**目标**: 预测明天可能涨停的股票

```python
import akshare as ak
import pandas as pd

# 1. 获取今日行情
df = ak.stock_zh_a_spot_em()

# 2. 筛选条件
# - 换手率 > 5% (活跃)
# - 涨幅 > 3% (强势)
# - 流通市值 < 100亿 (小盘)
# - 量比 > 2 (放量)

candidates = df[
    (df['换手率'] > 5) & 
    (df['涨跌幅'] > 3) & 
    (df['流通市值'] < 100e8) & 
    (df['量比'] > 2)
]

# 3. 计算涨停概率得分
def limitup_score(row):
    score = 0
    score += min(row['换手率'] * 5, 40)  # 换手率
    score += min(row['涨跌幅'] * 3, 30)  # 涨幅
    score += min(row['量比'] * 10, 30)  # 量比
    return score

candidates['score'] = candidates.apply(limitup_score, axis=1)

# 4. 排序输出
result = candidates.nlargest(10, 'score')
print("🎯 明日涨停潜力股TOP10:")
print(result[['代码', '名称', 'score', '涨跌幅', '换手率']])
```

---

### 3.2 场景2: 因子组合策略 📊

**目标**: 组合多个因子选股

```python
from qlib.data.dataset.handler import Alpha158
import pandas as pd

# 1. 获取因子数据
handler = Alpha158(instruments='csi300')
factors = handler.fetch()

# 2. 选择5个因子
selected_factors = [
    'RESI5',    # 5日收益残差
    'WVMA5',    # 5日加权均价
    'RSQR5',    # 5日R方
    'QTLU5',    # 5日上分位数
    'RSI6'      # 6日RSI
]

# 3. 因子标准化
for factor in selected_factors:
    factors[f'{factor}_norm'] = (
        factors[factor] - factors[factor].mean()
    ) / factors[factor].std()

# 4. 计算综合得分 (等权重)
factors['score'] = sum(
    factors[f'{f}_norm'] for f in selected_factors
) / len(selected_factors)

# 5. 选出得分最高的30只股票
top_stocks = factors.nlargest(30, 'score')
print("✅ 选出30只股票:")
print(top_stocks.index.get_level_values(1).unique())
```

---

### 3.3 场景3: 智能止损系统 🛡️

**目标**: 动态调整止损点

```python
class DynamicStopLoss:
    """动态止损系统"""
    
    def __init__(self, initial_stop=0.05):
        self.initial_stop = initial_stop  # 初始止损5%
        self.positions = {}
    
    def update_stop(self, symbol, current_price, highest_price):
        """更新止损价"""
        # 1. 初始止损
        if symbol not in self.positions:
            self.positions[symbol] = {
                'entry_price': current_price,
                'stop_price': current_price * (1 - self.initial_stop),
                'highest_price': current_price
            }
            return self.positions[symbol]['stop_price']
        
        pos = self.positions[symbol]
        
        # 2. 更新最高价
        if current_price > pos['highest_price']:
            pos['highest_price'] = current_price
        
        # 3. 浮盈>10%: 移动止损到保本
        profit = (current_price - pos['entry_price']) / pos['entry_price']
        if profit > 0.10:
            new_stop = max(
                pos['entry_price'],  # 保本
                pos['highest_price'] * 0.95  # 或最高价-5%
            )
            pos['stop_price'] = max(pos['stop_price'], new_stop)
        
        return pos['stop_price']
    
    def should_stop(self, symbol, current_price):
        """是否应该止损"""
        if symbol not in self.positions:
            return False
        
        return current_price < self.positions[symbol]['stop_price']

# 使用示例
stop_loss = DynamicStopLoss(initial_stop=0.05)

# 买入
stop_price = stop_loss.update_stop('000001', 10.0, 10.0)
print(f"✅ 买入 000001, 止损价: {stop_price}")

# 价格上涨到12元
stop_price = stop_loss.update_stop('000001', 12.0, 12.0)
print(f"📈 价格上涨, 新止损价: {stop_price}")

# 检查是否止损
if stop_loss.should_stop('000001', 11.5):
    print("❌ 触发止损")
else:
    print("✅ 继续持有")
```

---

### 3.4 场景4: 行业轮动 🔄

**目标**: 买入当前最强的行业

```python
import akshare as ak

# 1. 获取行业数据
industries = ak.stock_board_industry_name_em()

# 2. 获取每个行业的涨幅
industry_performance = []

for industry in industries['板块名称'][:10]:  # 前10个行业
    df = ak.stock_board_industry_cons_em(symbol=industry)
    avg_change = df['涨跌幅'].mean()
    
    industry_performance.append({
        'industry': industry,
        'avg_change': avg_change,
        'top_stocks': df.nlargest(5, '涨跌幅')['代码'].tolist()
    })

# 3. 选择表现最好的行业
result = pd.DataFrame(industry_performance)
result = result.sort_values('avg_change', ascending=False)

print("🔥 最强行业TOP3:")
for i, row in result.head(3).iterrows():
    print(f"{row['industry']}: {row['avg_change']:.2f}%")
    print(f"   龙头股: {', '.join(row['top_stocks'][:3])}\n")
```

---

### 3.5 场景5: 财报选股 📊

**目标**: 根据财报数据选优质股

```python
import akshare as ak

# 1. 获取财务数据
df = ak.stock_financial_abstract_em()

# 2. 筛选条件 (价值投资)
good_stocks = df[
    (df['净利润'] > 0) &           # 盈利
    (df['净利润同比增长'] > 20) &   # 增长>20%
    (df['净资产收益率'] > 15) &     # ROE>15%
    (df['资产负债率'] < 60)         # 负债率<60%
]

# 3. 计算价值得分
def value_score(row):
    score = 0
    score += min(row['净利润同比增长'], 50)  # 增长率
    score += row['净资产收益率']             # ROE
    score -= row['资产负债率'] / 2           # 负债率惩罚
    return score

good_stocks['value_score'] = good_stocks.apply(value_score, axis=1)

# 4. 排序输出
result = good_stocks.nlargest(20, 'value_score')
print("💎 优质价值股TOP20:")
print(result[['股票代码', '股票简称', 'value_score', '净资产收益率']])
```

---

### 3.6 场景6: 量化网格交易 🎯

**目标**: 在震荡行情中高抛低吸

```python
class GridTrading:
    """网格交易策略"""
    
    def __init__(self, symbol, base_price, grid_size=0.02, grids=10):
        self.symbol = symbol
        self.base_price = base_price      # 基准价
        self.grid_size = grid_size        # 网格大小2%
        self.grids = grids                # 网格数量
        self.positions = {}
        
        # 生成网格价格
        self.grid_prices = [
            base_price * (1 + grid_size * i) 
            for i in range(-grids//2, grids//2 + 1)
        ]
        print(f"✅ 网格价格: {self.grid_prices}")
    
    def on_price_change(self, current_price):
        """价格变化时的操作"""
        signals = []
        
        for i, price in enumerate(self.grid_prices):
            # 价格下穿网格线: 买入
            if current_price <= price and i not in self.positions:
                signals.append({
                    'action': 'BUY',
                    'price': price,
                    'reason': f'价格触及买入网格{i}'
                })
                self.positions[i] = price
            
            # 价格上穿网格线: 卖出
            elif current_price >= price and i in self.positions:
                signals.append({
                    'action': 'SELL',
                    'price': price,
                    'reason': f'价格触及卖出网格{i}'
                })
                del self.positions[i]
        
        return signals

# 使用示例
grid = GridTrading(
    symbol='000001',
    base_price=10.0,
    grid_size=0.02,  # 2%网格
    grids=10
)

# 模拟价格变化
prices = [10.0, 9.8, 9.6, 9.8, 10.0, 10.2, 10.4, 10.2, 10.0]

for price in prices:
    signals = grid.on_price_change(price)
    if signals:
        print(f"📊 价格: {price}")
        for signal in signals:
            print(f"   {signal['action']}: {signal['reason']}")
```

---

### 3.7 场景7: 机器学习预测 🤖

**目标**: 使用机器学习预测涨跌

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# 1. 准备数据
df = pd.read_csv('data/000001.csv')

# 2. 构造特征
df['returns'] = df['收盘'].pct_change()
df['ma5'] = df['收盘'].rolling(5).mean()
df['ma20'] = df['收盘'].rolling(20).mean()
df['volume_ratio'] = df['成交量'] / df['成交量'].rolling(5).mean()

# 3. 构造标签 (明天涨跌)
df['target'] = (df['returns'].shift(-1) > 0).astype(int)

# 4. 准备训练数据
features = ['ma5', 'ma20', 'volume_ratio']
df = df.dropna()

X = df[features]
y = df['target']

# 划分训练集和测试集
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. 训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. 预测
y_pred = model.predict(X_test)

# 7. 评估
accuracy = (y_pred == y_test).mean()
print(f"✅ 预测准确率: {accuracy*100:.2f}%")

# 8. 今天的预测
today_features = X.iloc[-1:].values
prediction = model.predict(today_features)[0]
print(f"📈 明天预测: {'上涨' if prediction == 1 else '下跌'}")
```

---

### 3.8 场景8: 自动化交易日报 📧

**目标**: 每日生成交易报告

```python
from datetime import datetime
import pandas as pd

class DailyReport:
    """交易日报生成器"""
    
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.date = datetime.now().strftime('%Y-%m-%d')
    
    def generate(self):
        """生成报告"""
        report = f"""
# 📊 交易日报 - {self.date}

## 1. 账户概况
- 总资产: ¥{self.portfolio['total_value']:,.2f}
- 可用资金: ¥{self.portfolio['cash']:,.2f}
- 持仓市值: ¥{self.portfolio['position_value']:,.2f}
- 今日盈亏: ¥{self.portfolio['daily_pnl']:,.2f} ({self.portfolio['daily_pnl_pct']:.2f}%)

## 2. 持仓明细
"""
        for pos in self.portfolio['positions']:
            report += f"""
### {pos['symbol']} - {pos['name']}
- 持仓: {pos['quantity']} 股
- 成本: ¥{pos['cost']:.2f}
- 现价: ¥{pos['current_price']:.2f}
- 盈亏: ¥{pos['pnl']:,.2f} ({pos['pnl_pct']:.2f}%)
"""
        
        report += f"""
## 3. 今日交易
"""
        for trade in self.portfolio['today_trades']:
            report += f"- {trade['time']} {trade['action']} {trade['symbol']} {trade['quantity']}股 @¥{trade['price']:.2f}\n"
        
        report += f"""
## 4. 风险提示
- 仓位: {self.portfolio['position_ratio']:.1f}%
- 最大持仓: {self.portfolio['max_position_stock']} ({self.portfolio['max_position_ratio']:.1f}%)
"""
        
        return report
    
    def save(self, filename=None):
        """保存报告"""
        if filename is None:
            filename = f"reports/daily_report_{self.date}.md"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.generate())
        
        print(f"✅ 报告已保存: {filename}")

# 使用示例
portfolio = {
    'total_value': 105000,
    'cash': 30000,
    'position_value': 75000,
    'daily_pnl': 2000,
    'daily_pnl_pct': 1.94,
    'position_ratio': 71.4,
    'max_position_stock': '000001',
    'max_position_ratio': 28.5,
    'positions': [
        {
            'symbol': '000001',
            'name': '平安银行',
            'quantity': 1000,
            'cost': 10.0,
            'current_price': 10.5,
            'pnl': 500,
            'pnl_pct': 5.0
        }
    ],
    'today_trades': [
        {
            'time': '09:35',
            'action': 'BUY',
            'symbol': '000001',
            'quantity': 1000,
            'price': 10.0
        }
    ]
}

report = DailyReport(portfolio)
report.save()
```

---

### 3.9 场景9: 多策略组合 🎯

**目标**: 组合多个策略分散风险

```python
class StrategyPortfolio:
    """多策略组合"""
    
    def __init__(self, strategies, weights):
        self.strategies = strategies
        self.weights = weights  # 每个策略的权重
    
    def get_signals(self, data):
        """获取所有策略的信号"""
        all_signals = []
        
        for strategy, weight in zip(self.strategies, self.weights):
            signals = strategy.generate_signals(data)
            # 加权信号
            for signal in signals:
                signal['weight'] = weight
            all_signals.extend(signals)
        
        return all_signals
    
    def combine_signals(self, signals):
        """合并信号"""
        # 按股票分组
        grouped = {}
        for signal in signals:
            symbol = signal['symbol']
            if symbol not in grouped:
                grouped[symbol] = {
                    'buy_score': 0,
                    'sell_score': 0
                }
            
            if signal['action'] == 'BUY':
                grouped[symbol]['buy_score'] += signal['weight']
            elif signal['action'] == 'SELL':
                grouped[symbol]['sell_score'] += signal['weight']
        
        # 生成最终信号
        final_signals = []
        for symbol, scores in grouped.items():
            if scores['buy_score'] > 0.5:  # 超过50%权重建议买入
                final_signals.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'confidence': scores['buy_score']
                })
            elif scores['sell_score'] > 0.5:
                final_signals.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'confidence': scores['sell_score']
                })
        
        return final_signals

# 使用示例
# 策略1: 动量策略
class MomentumStrategy:
    def generate_signals(self, data):
        signals = []
        if data['ma5'] > data['ma20']:
            signals.append({'symbol': data['symbol'], 'action': 'BUY'})
        return signals

# 策略2: 均值回归
class MeanReversionStrategy:
    def generate_signals(self, data):
        signals = []
        if data['close'] < data['ma20'] * 0.95:
            signals.append({'symbol': data['symbol'], 'action': 'BUY'})
        return signals

# 组合策略
portfolio = StrategyPortfolio(
    strategies=[MomentumStrategy(), MeanReversionStrategy()],
    weights=[0.6, 0.4]  # 动量60%, 均值回归40%
)

# 获取信号
data = {'symbol': '000001', 'close': 10.0, 'ma5': 10.2, 'ma20': 10.5}
signals = portfolio.get_signals(data)
final = portfolio.combine_signals(signals)

print("🎯 最终信号:")
for signal in final:
    print(f"{signal['symbol']}: {signal['action']} (置信度: {signal['confidence']:.2f})")
```

---

### 3.10 场景10: 实时监控告警 🔔

**目标**: 监控持仓,异常时告警

```python
class MarketMonitor:
    """市场监控"""
    
    def __init__(self, positions, alert_config):
        self.positions = positions
        self.alert_config = alert_config
    
    def check_alerts(self, market_data):
        """检查告警"""
        alerts = []
        
        for pos in self.positions:
            symbol = pos['symbol']
            current_price = market_data[symbol]['price']
            
            # 1. 止损告警
            loss_pct = (current_price - pos['cost']) / pos['cost']
            if loss_pct < -self.alert_config['stop_loss']:
                alerts.append({
                    'type': 'STOP_LOSS',
                    'symbol': symbol,
                    'message': f"{symbol} 触发止损 ({loss_pct*100:.2f}%)",
                    'urgency': 'HIGH'
                })
            
            # 2. 止盈告警
            if loss_pct > self.alert_config['take_profit']:
                alerts.append({
                    'type': 'TAKE_PROFIT',
                    'symbol': symbol,
                    'message': f"{symbol} 达到止盈 ({loss_pct*100:.2f}%)",
                    'urgency': 'MEDIUM'
                })
            
            # 3. 大幅波动告警
            volatility = market_data[symbol]['volatility']
            if volatility > self.alert_config['max_volatility']:
                alerts.append({
                    'type': 'HIGH_VOLATILITY',
                    'symbol': symbol,
                    'message': f"{symbol} 波动率异常 ({volatility:.2f}%)",
                    'urgency': 'MEDIUM'
                })
            
            # 4. 成交量异常
            volume_ratio = market_data[symbol]['volume_ratio']
            if volume_ratio > 3:
                alerts.append({
                    'type': 'VOLUME_SPIKE',
                    'symbol': symbol,
                    'message': f"{symbol} 成交量放大 ({volume_ratio:.1f}倍)",
                    'urgency': 'LOW'
                })
        
        return alerts
    
    def send_alerts(self, alerts):
        """发送告警"""
        for alert in alerts:
            # 这里可以集成邮件、微信、钉钉等
            print(f"🔔 [{alert['urgency']}] {alert['message']}")

# 使用示例
positions = [
    {'symbol': '000001', 'cost': 10.0, 'quantity': 1000}
]

alert_config = {
    'stop_loss': 0.05,        # 5%止损
    'take_profit': 0.15,      # 15%止盈
    'max_volatility': 5.0     # 波动率>5%
}

monitor = MarketMonitor(positions, alert_config)

# 模拟市场数据
market_data = {
    '000001': {
        'price': 9.4,           # 当前价
        'volatility': 6.5,      # 波动率
        'volume_ratio': 3.5     # 量比
    }
}

alerts = monitor.check_alerts(market_data)
monitor.send_alerts(alerts)
```

---

## 4. 进阶功能

### 4.1 使用本地大模型 (不需要API Key) 🆓

#### 为什么使用本地模型?

- ✅ **免费** - 不需要API Key
- ✅ **隐私** - 数据不出本地
- ✅ **无限制** - 不限调用次数

#### 快速部署本地模型

**Step 1: 安装vllm**

```bash
pip install vllm
```

**Step 2: 下载模型** (推荐Qwen-14B)

```bash
# 使用modelscope下载 (国内快)
pip install modelscope
modelscope download --model qwen/Qwen-14B-Chat
```

**Step 3: 启动模型服务**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model qwen/Qwen-14B-Chat \
    --port 8000
```

**Step 4: 配置Qilin Stack**

```python
config = {
    'llm_model': 'qwen/Qwen-14B-Chat',
    'llm_provider': 'openai',  # vllm兼容OpenAI API
    'llm_base_url': 'http://localhost:8000/v1',
    'llm_api_key': 'EMPTY',  # 本地模型不需要key
    'max_iterations': 10
}

agent = RDAgentWrapper(config)
```

**💡 小白提示**: 
- 本地模型需要较好的GPU (至少8GB显存)
- 推荐模型: Qwen-14B (中文好), Llama-2-13B (英文好)
- 没有GPU? 可以用CPU,但会很慢

---

### 4.2 性能优化 ⚡

#### 使用Numba加速因子计算

```python
import numba as nb
import numpy as np

# 普通Python (慢)
def calc_factor_slow(close_prices):
    results = []
    for i in range(20, len(close_prices)):
        ma20 = sum(close_prices[i-20:i]) / 20
        results.append(close_prices[i] / ma20 - 1)
    return results

# Numba加速 (快50-100倍)
@nb.jit(nopython=True)
def calc_factor_fast(close_prices):
    n = len(close_prices)
    results = np.empty(n - 20)
    
    for i in range(20, n):
        ma20 = np.mean(close_prices[i-20:i])
        results[i-20] = close_prices[i] / ma20 - 1
    
    return results

# 测试
close = np.random.random(10000)

%timeit calc_factor_slow(close)  # 100ms
%timeit calc_factor_fast(close)  # 1ms (快100倍!)
```

#### 批量计算因子

```python
import pandas as pd
import numpy as np

# 一次性计算多个因子
def calc_factors_batch(df):
    """批量计算所有因子"""
    
    # 使用向量化操作
    df['returns'] = df['close'].pct_change()
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['volume_ma5'] = df['volume'].rolling(5).mean()
    
    # 避免循环
    df['momentum'] = df['close'] / df['close'].shift(20) - 1
    df['volatility'] = df['returns'].rolling(20).std()
    
    return df

# 使用
df = calc_factors_batch(df)  # 快!
```

---

### 4.3 分布式回测 🚀

#### 使用多进程加速回测

```python
from multiprocessing import Pool
import pandas as pd

def backtest_single_stock(symbol):
    """单只股票回测"""
    df = pd.read_csv(f'data/{symbol}.csv')
    # ... 回测逻辑
    return result

# 串行 (慢)
results = []
for symbol in symbols:
    result = backtest_single_stock(symbol)
    results.append(result)

# 并行 (快4-8倍)
with Pool(processes=8) as pool:
    results = pool.map(backtest_single_stock, symbols)
```

---

### 4.4 数据库存储 💾

#### 使用SQLite存储数据

```python
import sqlite3
import pandas as pd

# 1. 创建数据库
conn = sqlite3.connect('qilin_data.db')

# 2. 保存数据
df.to_sql('stock_000001', conn, if_exists='replace', index=False)

# 3. 读取数据
df = pd.read_sql('SELECT * FROM stock_000001 WHERE date > "2023-01-01"', conn)

# 4. 关闭连接
conn.close()
```

#### 使用ClickHouse (大数据)

```python
from clickhouse_driver import Client

# 连接ClickHouse
client = Client('localhost')

# 批量插入
client.execute(
    'INSERT INTO stocks VALUES',
    df.to_dict('records')
)

# 查询
result = client.execute('SELECT * FROM stocks WHERE symbol = "000001"')
```

---

### 4.5 自定义因子 🔧

#### 创建自己的因子

```python
from qlib.data.dataset.handler import DataHandlerLP

class MyCustomFactor(DataHandlerLP):
    """自定义因子"""
    
    def __init__(self, instruments='csi300', **kwargs):
        # 定义因子表达式
        fields = [
            # 价格动量
            "($close / Ref($close, 20) - 1)",  # 20日收益率
            
            # 成交量
            "($volume / Mean($volume, 5) - 1)",  # 量比
            
            # 波动率
            "Std($close, 20) / Mean($close, 20)",
            
            # 自定义: 价量背离
            "(Rank($close) - Rank($volume))",
        ]
        
        super().__init__(
            instruments=instruments,
            fields=fields,
            **kwargs
        )

# 使用
handler = MyCustomFactor()
factors = handler.fetch()
```

---

### 4.6 实时数据接入 📡

#### WebSocket实时行情

```python
import websocket
import json

class RealtimeData:
    """实时行情"""
    
    def __init__(self, symbols):
        self.symbols = symbols
        self.ws = None
    
    def on_message(self, ws, message):
        """接收消息"""
        data = json.loads(message)
        print(f"📊 {data['symbol']}: {data['price']}")
        
        # 这里可以触发交易逻辑
        self.on_price_update(data)
    
    def on_price_update(self, data):
        """价格更新时的处理"""
        # 检查交易信号
        # 发送订单
        pass
    
    def start(self):
        """启动连接"""
        self.ws = websocket.WebSocketApp(
            "ws://your-data-provider/stream",
            on_message=self.on_message
        )
        self.ws.run_forever()

# 使用
realtime = RealtimeData(['000001', '000002'])
realtime.start()
```

---

## 5. 常见问题

### 5.1 安装问题

**Q1: pip install 报错 "No matching distribution"?**

```bash
# 升级pip
python -m pip install --upgrade pip

# 或指定Python版本
python3.8 -m pip install -r requirements.txt
```

**Q2: 虚拟环境激活失败?**

```bash
# Windows PowerShell可能被禁止,使用CMD
.qilin\Scripts\activate.bat

# 或关闭PowerShell限制
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### 5.2 数据问题

**Q3: AKShare下载数据失败?**

```python
# 可能是网络问题,加重试
import time

def download_with_retry(symbol, max_retries=3):
    for i in range(max_retries):
        try:
            df = ak.stock_zh_a_hist(symbol=symbol)
            return df
        except Exception as e:
            print(f"重试 {i+1}/{max_retries}: {e}")
            time.sleep(2)
    return None
```

**Q4: 数据格式不对?**

```python
# 统一数据格式
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
```

---

### 5.3 策略问题

**Q5: 回测效果好,实盘亏损?**

**原因**:
- ❌ 过度拟合 - 策略太复杂
- ❌ 未来函数 - 使用了未来数据
- ❌ 滑点成本 - 没考虑交易成本
- ❌ 流动性 - 小盘股买不进卖不出

**解决**:
```python
# 1. 添加交易成本
returns = returns - 0.0003  # 万3手续费

# 2. 检查未来函数
# 错误: df['signal'] = df['close'].shift(-1) > df['close']
# 正确: df['signal'] = df['close'] > df['close'].shift(1)

# 3. 只交易流通市值>50亿的股票
df = df[df['流通市值'] > 50e8]
```

**Q6: 策略收益不稳定?**

```python
# 使用滚动回测验证
def rolling_backtest(df, window=252):
    """滚动回测"""
    results = []
    
    for i in range(window, len(df), window//4):
        train = df[i-window:i]
        test = df[i:i+window//4]
        
        # 在train上训练,在test上测试
        result = backtest(train, test)
        results.append(result)
    
    return pd.DataFrame(results)
```

---

### 5.4 性能问题

**Q7: 因子计算太慢?**

```python
# 使用Numba或向量化
# 慢: for循环
# 快: df['factor'] = (df['close'] / df['close'].shift(20) - 1)

# 或使用多进程
from multiprocessing import Pool
with Pool(8) as pool:
    results = pool.map(calc_factor, symbols)
```

**Q8: 内存不够?**

```python
# 分批处理
chunk_size = 100
for i in range(0, len(symbols), chunk_size):
    batch = symbols[i:i+chunk_size]
    process_batch(batch)
```

---

### 5.5 实盘问题

**Q9: 订单成交价和预期不一致?**

**原因**: 市场波动 + 滑点

**解决**:
```python
# 1. 使用限价单
order = {
    'type': 'LIMIT',
    'price': current_price * 1.01  # 最多贵1%
}

# 2. 避免盘口薄的股票
if bid_ask_spread > 0.01:  # 买卖价差>1%
    print("⚠️ 流动性差,放弃")
```

**Q10: 实盘止损没触发?**

```python
# 实时监控价格
import schedule

def check_stop_loss():
    """定时检查止损"""
    for pos in positions:
        current_price = get_realtime_price(pos['symbol'])
        if current_price < pos['stop_price']:
            execute_stop_loss(pos)

# 每5秒检查一次
schedule.every(5).seconds.do(check_stop_loss)

while True:
    schedule.run_pending()
    time.sleep(1)
```

---

## 6. 最佳实践

### 6.1 新手建议 ✅

#### 第1周: 学习基础

- [ ] 跑通快速入门示例
- [ ] 下载历史数据
- [ ] 理解回测概念
- [ ] 运行简单策略

#### 第2-4周: 模拟交易

- [ ] 使用模拟盘测试
- [ ] 记录每笔交易
- [ ] 分析亏损原因
- [ ] 优化策略

#### 第2-3个月: 实盘小金额

- [ ] 1-5万小资金
- [ ] 严格执行风控
- [ ] 记录交易日志
- [ ] 持续学习改进

#### 6个月后: 增加资金

- [ ] 策略稳定盈利
- [ ] 心态平稳
- [ ] 逐步加仓

---

### 6.2 风险控制 🛡️

#### 黄金法则

1. **永远不要满仓** - 至少留20%现金
2. **单只股票≤10%** - 分散风险
3. **止损必须设** - 5-10%强制止损
4. **不要加死杠杆** - 尤其是高杠杆
5. **远离ST和退市股** - 除非特别策略

#### 风控检查清单

```python
def risk_check(portfolio):
    """风控检查"""
    warnings = []
    
    # 1. 仓位过高
    if portfolio['position_ratio'] > 0.9:
        warnings.append("⚠️ 仓位过高 (>90%)")
    
    # 2. 单股集中
    for pos in portfolio['positions']:
        if pos['ratio'] > 0.15:
            warnings.append(f"⚠️ {pos['symbol']} 仓位过重 (>{15}%)")
    
    # 3. 没有止损
    for pos in portfolio['positions']:
        if 'stop_price' not in pos:
            warnings.append(f"⚠️ {pos['symbol']} 未设置止损")
    
    # 4. 浮亏过大
    for pos in portfolio['positions']:
        if pos['pnl_pct'] < -0.10:
            warnings.append(f"🚨 {pos['symbol']} 浮亏>10%,建议止损")
    
    return warnings
```

---

### 6.3 交易纪律 📋

#### 制定交易计划

```markdown
# 我的交易计划

## 策略
- 因子: Alpha158 + 动量
- 选股: CSI300成分股
- 持仓: 20-30只
- 换手: 每周调仓

## 风控
- 单股: ≤10%
- 总仓位: ≤80%
- 止损: -5%
- 止盈: +15%

## 执行
- 开盘前: 计算信号
- 09:35-09:45: 集中交易
- 收盘后: 复盘总结

## 纪律
- 严格执行止损
- 不频繁调仓
- 不追涨杀跌
- 亏损不加仓
```

---

### 6.4 持续学习 📚

#### 推荐资源

**书籍**:
- 《量化投资：以Python为工具》
- 《打开量化投资的黑箱》
- 《主动投资组合管理》

**网站**:
- 聚宽 (www.joinquant.com)
- 优矿 (uqer.datayes.com)
- Qlib文档 (qlib.readthedocs.io)

**社区**:
- GitHub - microsoft/qlib
- 知乎 - 量化交易话题
- Reddit - r/algotrading

---

### 6.5 代码规范 💻

#### 良好习惯

```python
# ✅ 好的代码
def calc_momentum(df, window=20):
    """
    计算动量因子
    
    Args:
        df: 股票数据
        window: 计算窗口
    
    Returns:
        动量值
    """
    return df['close'] / df['close'].shift(window) - 1

# ❌ 不好的代码
def cm(d, w=20):
    return d['close'] / d['close'].shift(w) - 1
```

#### 版本控制

```bash
# 使用Git管理代码
git init
git add .
git commit -m "初始版本"

# 每次修改都提交
git commit -m "添加止损功能"
```

---

## 🎓 结语

恭喜你读完这份指南! 

### 记住三点

1. **量化不是圣杯** - 需要持续学习和改进
2. **风控永远第一** - 活下来才能盈利
3. **实践出真知** - 理论要结合实盘

### 下一步

1. 完成快速入门 (1.2节)
2. 选一个感兴趣的场景 (第3章)
3. 开始模拟交易 (至少1个月)
4. 记录每笔交易和思考
5. 持续优化策略

### 获取帮助

- 📖 查看 [故障排查指南](TROUBLESHOOTING.md)
- 💬 提交 [GitHub Issues](https://github.com/your-org/qilin_stack/issues)
- 📧 邮件: support@example.com

---

**祝你量化交易之路顺利! 🚀**

**最后更新**: 2024-11-08 | **版本**: 2.0

# 统一数据流管道

## 🎯 概述

统一数据流管道为Qlib、TradingAgents、RD-Agent三个系统提供一致的数据访问接口，支持多数据源融合和自动降级。

## ✨ 核心功能

### 1. 多数据源支持
- ✅ **Qlib**: 历史回测数据
- ✅ **AKShare**: 实时行情数据
- ⏳ **Tushare**: 备用数据源
- ⏳ **JoinQuant**: 备用数据源

### 2. 自动降级策略
```
Primary: Qlib → Fallback: AKShare → Fallback: Tushare
```

### 3. 统一数据格式
```python
@dataclass
class MarketData:
    symbol: str
    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    frequency: DataFrequency
    source: DataSource
```

### 4. 三系统桥接
- **QlibDataBridge**: Qlib格式数据转换
- **TradingAgentsDataBridge**: 市场状态格式转换
- **RDAgentDataBridge**: 因子数据格式转换

---

## 🚀 快速开始

### 1. 基础使用

```python
from data_pipeline.unified_data import get_unified_pipeline, DataFrequency

# 获取统一数据管道
pipeline = get_unified_pipeline()

# 获取K线数据
data = pipeline.get_bars(
    symbols=['000001.SZ', '600000.SH'],
    start_date='2024-01-01',
    end_date='2024-06-30',
    frequency=DataFrequency.DAY
)

print(data.head())
```

### 2. 使用桥接器

```python
from data_pipeline.system_bridge import get_unified_bridge

# 获取统一桥接管理器
bridge = get_unified_bridge()

# Qlib格式数据
qlib_bridge = bridge.get_qlib_bridge()
qlib_data = qlib_bridge.get_qlib_format_data(
    instruments=['000001.SZ'],
    fields=['$open', '$close', '$volume'],
    start_time='2024-01-01',
    end_time='2024-06-30'
)

# TradingAgents市场状态
ta_bridge = bridge.get_tradingagents_bridge()
market_state = ta_bridge.get_market_state(
    symbols=['000001.SZ'],
    date='2024-06-30'
)

# RD-Agent因子数据
rd_bridge = bridge.get_rdagent_bridge()
factors = rd_bridge.get_factor_data(
    symbols=['000001.SZ'],
    start_date='2024-01-01',
    end_date='2024-06-30'
)
```

---

## 📊 架构设计

```
统一数据流管道
├── 数据源层 (unified_data.py)
│   ├── DataSourceAdapter (抽象基类)
│   ├── QlibDataAdapter
│   ├── AKShareDataAdapter
│   └── TushareDataAdapter (TODO)
├── 统一管道层 (unified_data.py)
│   └── UnifiedDataPipeline
│       ├── get_bars()
│       ├── get_ticks()
│       ├── get_fundamentals()
│       └── get_realtime_quote()
└── 桥接层 (system_bridge.py)
    ├── QlibDataBridge
    ├── TradingAgentsDataBridge
    ├── RDAgentDataBridge
    └── UnifiedDataBridge (管理器)
```

---

## 🔧 数据源配置

### Qlib配置

```python
# Qlib自动初始化，数据路径：~/.qlib/qlib_data/cn_data
# 如需自定义：
import qlib
qlib.init(provider_uri="your_data_path", region=REG_CN)
```

### AKShare配置

```python
# AKShare无需配置，直接使用
# 支持：
# - 日线数据（前复权）
# - 分钟数据
# - 实时行情
```

---

## 📚 API参考

### UnifiedDataPipeline

#### get_bars()
获取K线数据（支持多数据源降级）

```python
def get_bars(
    symbols: Union[str, List[str]],
    start_date: str,
    end_date: str,
    frequency: DataFrequency = DataFrequency.DAY,
    source: Optional[DataSource] = None
) -> pd.DataFrame
```

**参数**:
- `symbols`: 股票代码或列表
- `start_date`: 开始日期 (YYYY-MM-DD)
- `end_date`: 结束日期 (YYYY-MM-DD)
- `frequency`: 数据频率
- `source`: 指定数据源 (None=自动选择)

**返回**: MultiIndex DataFrame (symbol, datetime)

#### get_ticks()
获取tick数据

```python
def get_ticks(
    symbol: str,
    date: str,
    source: Optional[DataSource] = None
) -> pd.DataFrame
```

#### get_fundamentals()
获取基本面数据

```python
def get_fundamentals(
    symbols: Union[str, List[str]],
    date: str,
    source: Optional[DataSource] = None
) -> pd.DataFrame
```

#### get_realtime_quote()
获取实时行情

```python
def get_realtime_quote(
    symbols: Union[str, List[str]]
) -> pd.DataFrame
```

---

### QlibDataBridge

#### get_qlib_format_data()
获取Qlib格式数据

```python
def get_qlib_format_data(
    instruments: List[str],
    fields: List[str],
    start_time: str,
    end_time: str,
    freq: str = 'day'
) -> pd.DataFrame
```

**字段映射**:
- `$open`, `$high`, `$low`, `$close`
- `$volume`, `$amount`
- `$turnover_rate`, `$vwap`

#### get_features_for_model()
获取模型训练特征（含技术指标）

```python
def get_features_for_model(
    instruments: List[str],
    start_time: str,
    end_time: str
) -> pd.DataFrame
```

**包含特征**:
- 基础: open, high, low, close, volume
- 技术指标: ma5, ma20, rsi, volatility
- 衍生: returns, price_to_ma

---

### TradingAgentsDataBridge

#### get_market_state()
获取市场状态

```python
def get_market_state(
    symbols: List[str],
    date: str
) -> Dict[str, Any]
```

**返回格式**:
```python
{
    'timestamp': '2024-06-30',
    'prices': {
        '000001.SZ': {
            'current': 10.5,
            'open': 10.2,
            'high': 10.8,
            'low': 10.1,
            'history': [...]
        }
    },
    'volumes': {...},
    'fundamentals': {...}
}
```

---

### RDAgentDataBridge

#### get_factor_data()
获取因子数据

```python
def get_factor_data(
    symbols: List[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame
```

**包含因子**:
- 价格因子: price_to_ma5, price_to_ma20
- 动量因子: momentum_5d, momentum_20d
- 波动率因子: volatility_20d
- 成交量因子: volume_ratio
- 振幅因子: amplitude

#### get_limit_up_data()
获取涨停板数据

```python
def get_limit_up_data(
    symbols: List[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame
```

---

## 🔍 测试

### 测试统一管道

```bash
python data_pipeline/unified_data.py
```

### 测试桥接层

```bash
python data_pipeline/system_bridge.py
```

### 测试输出示例

```
=== 统一数据管道测试 ===

1️⃣ 测试数据源连通性:
  ✅ qlib: 可用
  ✅ akshare: 可用

2️⃣ 测试获取K线数据:
  获取到 20 条数据
  数据列: ['open', 'high', 'low', 'close', 'volume', 'amount']

3️⃣ 可用数据源:
  - qlib
  - akshare

✅ 测试完成
```

---

## 🎨 使用案例

### 案例1: Qlib模型训练

```python
from data_pipeline.system_bridge import get_unified_bridge

bridge = get_unified_bridge()
qlib_bridge = bridge.get_qlib_bridge()

# 获取训练数据
train_data = qlib_bridge.get_features_for_model(
    instruments=['000001.SZ', '000002.SZ'],
    start_time='2020-01-01',
    end_time='2023-12-31'
)

# 使用Qlib训练模型
# ... (省略模型训练代码)
```

### 案例2: TradingAgents实时交易

```python
from data_pipeline.system_bridge import get_unified_bridge

bridge = get_unified_bridge()
ta_bridge = bridge.get_tradingagents_bridge()

# 获取实时市场状态
market_state = ta_bridge.get_market_state(
    symbols=['000001.SZ'],
    date='2024-06-30'
)

# TradingAgents决策
# ... (省略决策代码)
```

### 案例3: RD-Agent因子研究

```python
from data_pipeline.system_bridge import get_unified_bridge

bridge = get_unified_bridge()
rd_bridge = bridge.get_rdagent_bridge()

# 获取涨停板数据
limit_ups = rd_bridge.get_limit_up_data(
    symbols=['000001.SZ', '000002.SZ'],
    start_date='2024-01-01',
    end_date='2024-06-30'
)

# 分析涨停板特征
# ... (省略分析代码)
```

---

## 🐛 故障排除

### 问题1: Qlib初始化失败

```python
# 检查Qlib数据路径
import os
qlib_data_path = os.path.expanduser("~/.qlib/qlib_data/cn_data")
print(f"Qlib数据路径: {qlib_data_path}")
print(f"路径存在: {os.path.exists(qlib_data_path)}")
```

### 问题2: AKShare数据获取失败

```python
# 测试AKShare连通性
import akshare as ak
try:
    df = ak.stock_zh_a_spot_em()
    print(f"AKShare可用，获取到 {len(df)} 只股票")
except Exception as e:
    print(f"AKShare失败: {e}")
```

### 问题3: 数据缓存问题

```python
# 清除缓存
import shutil
from pathlib import Path

cache_dir = Path("./cache/data")
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print("缓存已清除")
```

---

## 🔮 后续计划

### 短期
- [ ] 添加Tushare数据源支持
- [ ] 实现数据质量检查
- [ ] 添加数据对齐功能

### 中期
- [ ] 支持更多数据频率（分钟、小时）
- [ ] 实现数据流式更新
- [ ] 添加数据监控面板

### 长期
- [ ] 分布式数据缓存
- [ ] 实时数据流处理
- [ ] 智能数据源选择

---

**状态**: ✅ 可用
**版本**: 1.0.0
**更新日期**: 2024

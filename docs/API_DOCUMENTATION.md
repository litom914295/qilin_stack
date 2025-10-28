# Qilin Stack API 文档

## 概述

Qilin Stack 是一个量化交易系统,提供完整的数据获取、策略开发、回测和实盘交易功能。

## 核心模块

### 1. 输入验证模块 (`app/core/validators.py`)

提供统一的数据验证和清洗功能。

#### 主要类

**`Validator`** - 通用验证器

主要方法:
- `normalize_symbol(symbol, output_format)` - 标准化股票代码格式
- `validate_symbol(symbol, market)` - 验证股票代码
- `validate_quantity(qty)` - 验证交易数量  
- `validate_price(price, symbol)` - 验证价格
- `validate_order(order)` - 验证订单数据
- `validate_dataframe(df, required_columns)` - 验证DataFrame数据
- `validate_parameter(param_name, value, min_val, max_val, allowed_values)` - 参数验证
- `validate_config(config, config_schema)` - 验证配置文件
- `sanitize_input(input_str, max_length)` - 清理和净化输入字符串

**`RiskValidator`** - 风险验证器

主要方法:
- `validate_position_size(position, capital, max_ratio)` - 验证仓位大小
- `validate_stop_loss(entry_price, stop_price, max_loss)` - 验证止损设置
- `validate_leverage(leverage, max_leverage)` - 验证杠杆率

示例:
```python
from app.core.validators import Validator, ValidationError

# 标准化股票代码
symbol = Validator.normalize_symbol("600000.SH", output_format="qlib")
# 结果: "SH600000"

# 验证订单
order = {
    'symbol': 'SH600000',
    'side': 'BUY',
    'quantity': 100,
    'price': 10.5
}
validated_order = Validator.validate_order(order)
```

---

### 2. 配置管理模块 (`app/core/config_manager.py`)

使用 Pydantic V2 提供类型安全的配置管理。

#### 主要类

**`BaseConfig`** - 基础配置模型

**`DatabaseConfig`** - 数据库配置

**`TradingConfig`** - 交易配置

**`RiskConfig`** - 风控配置

**`BacktestConfig`** - 回测配置

**`SystemConfig`** - 系统配置(整合所有配置)

**`ConfigManager`** - 配置管理器

主要方法:
- `load_config(config_path)` - 从文件加载配置
- `save_config(config, config_path)` - 保存配置到文件
- `get(key, default)` - 获取配置值
- `set(key, value)` - 设置配置值
- `validate()` - 验证配置
- `to_dict()` - 转换为字典
- `from_dict(data)` - 从字典创建配置

示例:
```python
from app.core.config_manager import ConfigManager

# 加载配置
config = ConfigManager.load_config("config.yaml")

# 访问配置
print(config.trading.initial_capital)
print(config.risk.max_position_ratio)

# 修改配置
config.trading.commission_rate = 0.0003
config.save("updated_config.yaml")
```

---

### 3. 日志管理模块 (`app/core/logging_manager.py`)

提供统一的日志记录功能,支持敏感信息过滤。

#### 主要类

**`SensitiveDataFilter`** - 敏感数据过滤器

**`LoggingManager`** - 日志管理器

主要方法:
- `setup_logging(log_level, log_dir, app_name)` - 设置日志
- `get_logger(name)` - 获取日志记录器
- `set_level(level)` - 设置日志级别
- `add_handler(handler)` - 添加处理器
- `remove_handler(handler)` - 移除处理器

示例:
```python
from app.core.logging_manager import LoggingManager

# 设置日志
logger = LoggingManager.setup_logging(
    log_level="INFO",
    log_dir="./logs",
    app_name="qilin_stack"
)

# 使用日志
logger.info("系统启动")
logger.error("发生错误", exc_info=True)

# 敏感信息会自动被过滤
logger.info(f"API Key: {api_key}")  # 输出: API Key: ***
```

---

### 4. 缓存管理模块 (`app/core/cache_manager.py`)

提供多级缓存支持,优化数据访问性能。

#### 主要类

**`CacheManager`** - 缓存管理器

主要方法:
- `get(key, use_disk)` - 获取缓存
- `set(key, value, ttl, use_disk)` - 设置缓存
- `delete(key)` - 删除缓存
- `clear(memory_only)` - 清空缓存
- `cleanup_expired()` - 清理过期缓存

**装饰器:**
- `@cached(ttl, use_disk, key_func)` - 缓存装饰器
- `@memoize` - 记忆化装饰器(仅内存)

示例:
```python
from app.core.cache_manager import cached, memoize

# 使用缓存装饰器
@cached(ttl=3600)
def get_stock_data(symbol, date):
    # 耗时的数据获取操作
    return fetch_data(symbol, date)

# 使用记忆化装饰器
@memoize
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 第一次调用会执行函数
result1 = get_stock_data("SH600000", "2024-01-01")

# 第二次调用会从缓存获取
result2 = get_stock_data("SH600000", "2024-01-01")
```

---

### 5. 交易执行模块 (`app/core/trade_executor.py`)

处理订单生成、执行、状态跟踪。

#### 主要类

**`Order`** - 订单数据结构

属性:
- `symbol` - 股票代码
- `side` - 买卖方向(BUY/SELL)
- `order_type` - 订单类型(MARKET/LIMIT/STOP等)
- `quantity` - 数量
- `price` - 价格
- `status` - 订单状态
- `filled_quantity` - 成交数量
- `commission` - 手续费
- `slippage` - 滑点

**`Position`** - 持仓信息

**`ExecutionEngine`** - 交易执行引擎

主要方法:
- `connect()` - 连接引擎
- `disconnect()` - 断开引擎
- `execute_order(symbol, side, quantity, order_type, price, strategy)` - 执行订单
- `get_execution_report()` - 获取执行报告

支持的执行策略:
- `direct` - 直接执行
- `vwap` - 成交量加权平均价格
- `twap` - 时间加权平均价格
- `iceberg` - 冰山订单
- `smart` - 智能路由

示例:
```python
from app.core.trade_executor import ExecutionEngine, OrderSide, OrderType

# 创建执行引擎
engine = ExecutionEngine(broker_type="simulated", config={
    'initial_capital': 1000000,
    'commission_rate': 0.0003,
    'slippage': 0.001
})

await engine.connect()

# 执行市价单
order_id = await engine.execute_order(
    symbol="SH600000",
    side=OrderSide.BUY,
    quantity=1000,
    order_type=OrderType.MARKET
)

# 执行限价单
order_id = await engine.execute_order(
    symbol="SZ000001",
    side=OrderSide.BUY,
    quantity=500,
    order_type=OrderType.LIMIT,
    price=12.50,
    strategy="vwap"
)

# 获取执行报告
report = engine.get_execution_report()
print(report['summary'])
```

---

## 数据模块

### 数据获取 (`app/data/`)

- `realtime_data_fetcher.py` - 实时数据获取
- `level2_adapter.py` - Level-2 行情适配器
- `lhb_parser.py` - 龙虎榜数据解析
- `data_quality.py` - 数据质量检查

---

## Agent 模块

### 交易 Agent (`app/agents/`)

- `trading_agents_impl.py` - 交易 Agent 实现
- `enhanced_agents.py` - 增强型 Agent
- `enhanced_zt_quality_agent.py` - 涨停板质量评估 Agent
- `refined_auction_agent.py` - 集合竞价 Agent

---

## 回测模块

### 回测引擎 (`app/backtest/`)

- `backtest_engine.py` - 回测引擎核心
- `simple_backtest.py` - 简化回测接口

---

## 监控模块

### 监控和健康检查 (`app/monitoring/`)

- `metrics.py` - 指标收集
- `health.py` - 健康检查
- `api_routes.py` - 监控 API 路由

---

## 集成模块

### RD-Agent 集成 (`app/integration/`)

- `rdagent_adapter.py` - RD-Agent 适配器,用于模型开发和优化

---

## 使用示例

### 完整的交易流程

```python
import asyncio
from app.core.config_manager import ConfigManager
from app.core.validators import Validator
from app.core.trade_executor import ExecutionEngine, OrderSide, OrderType
from app.core.logging_manager import LoggingManager

async def main():
    # 1. 设置日志
    logger = LoggingManager.setup_logging()
    
    # 2. 加载配置
    config = ConfigManager.load_config("config.yaml")
    
    # 3. 创建执行引擎
    engine = ExecutionEngine(
        broker_type="simulated",
        config=config.trading.to_dict()
    )
    
    await engine.connect()
    
    try:
        # 4. 验证订单
        order_data = {
            'symbol': '600000.SH',
            'side': 'BUY',
            'quantity': 1000,
            'price': 10.5
        }
        
        validated_order = Validator.validate_order(order_data)
        logger.info(f"订单验证通过: {validated_order}")
        
        # 5. 执行订单
        order_id = await engine.execute_order(
            symbol=validated_order['symbol'],
            side=OrderSide[validated_order['side']],
            quantity=validated_order['quantity'],
            order_type=OrderType.LIMIT,
            price=validated_order['price'],
            strategy="smart"
        )
        
        logger.info(f"订单已提交: {order_id}")
        
        # 6. 等待执行
        await asyncio.sleep(5)
        
        # 7. 获取报告
        report = engine.get_execution_report()
        logger.info(f"执行报告: {report['summary']}")
        
    finally:
        await engine.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
```

---

## 配置文件示例

### config.yaml

```yaml
trading:
  initial_capital: 1000000
  commission_rate: 0.0003
  min_commission: 5
  slippage_rate: 0.001
  
risk:
  max_position_ratio: 0.3
  max_single_stock_ratio: 0.1
  stop_loss_ratio: 0.05
  take_profit_ratio: 0.15
  max_drawdown: 0.2

backtest:
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  benchmark: "SH000300"
  
database:
  host: "localhost"
  port: 3306
  username: "trader"
  password: "password"
  database: "qilin"
```

---

## 测试

所有核心模块都配备了完整的单元测试:

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_improvements.py -v
pytest tests/test_cache_manager.py -v

# 生成覆盖率报告
pytest tests/ --cov=app --cov-report=html
```

---

## 性能优化建议

1. **使用缓存** - 对频繁访问的数据使用 `@cached` 装饰器
2. **批量操作** - 使用 DataFrame 批量处理数据
3. **异步执行** - 对 I/O 密集型操作使用 async/await
4. **连接池** - 数据库和 API 连接使用连接池
5. **监控指标** - 使用 metrics 模块监控系统性能

---

## 贡献指南

1. 所有新功能必须包含单元测试
2. 代码必须通过 ruff 检查和类型检查
3. 使用类型注解(Type Hints)
4. 编写清晰的文档字符串
5. 遵循现有的代码风格

---

## 许可证

[添加许可证信息]

---

## 联系方式

[添加联系方式]

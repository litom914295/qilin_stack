# RD-Agent API 参考文档

**版本**: v1.0  
**最后更新**: 2024

---

## 概述

本文档提供麒麟项目中 RD-Agent 集成模块的完整 API 参考,包括:
- 配置管理 (`RDAgentConfig`)
- 官方集成 (`OfficialRDAgentManager`)
- 涨停板专属集成 (`LimitUpRDAgentIntegration`)
- 数据接口 (`LimitUpDataInterface`)
- 代码沙盒 (`CodeSandbox`)

---

## 目录

- [配置管理](#配置管理)
- [官方集成](#官方集成)
- [涨停板集成](#涨停板集成)
- [数据接口](#数据接口)
- [代码沙盒](#代码沙盒)
- [示例代码](#示例代码)

---

## 配置管理

### RDAgentConfig

配置类,管理 RD-Agent 的所有配置项。

**模块**: `rd_agent.config`

#### 初始化

```python
from rd_agent.config import RDAgentConfig

# 使用默认配置
config = RDAgentConfig()

# 自定义配置
config = RDAgentConfig(
    llm_model="gpt-4-turbo",
    max_iterations=20,
    checkpoint_path="./checkpoints/factor.pkl"
)
```

#### 主要字段

##### LLM 配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `llm_provider` | str | "openai" | LLM 提供商 (openai/azure/anthropic/local) |
| `llm_model` | str | "gpt-4-turbo" | LLM 模型名称 |
| `llm_api_key` | str | "" | API 密钥 |
| `llm_api_base` | Optional[str] | None | API 基础 URL |
| `llm_temperature` | float | 0.7 | 温度参数 (0-1) |
| `llm_max_tokens` | int | 4000 | 最大生成 token 数 |

##### 研究配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `research_mode` | str | "factor" | 研究模式 (factor/model/strategy) |
| `max_iterations` | int | 10 | 最大迭代次数 |
| `parallel_tasks` | int | 3 | 并行任务数 |
| `enable_cache` | bool | True | 是否启用缓存 |

##### P0-1: 会话恢复配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `checkpoint_path` | Optional[str] | None | Checkpoint 文件路径 |
| `enable_auto_checkpoint` | bool | True | 是否自动保存 checkpoint |
| `checkpoint_interval` | int | 5 | Checkpoint 保存间隔 (轮次) |

**示例**:
```python
config = RDAgentConfig(
    checkpoint_path="./checkpoints/factor_loop.pkl",
    enable_auto_checkpoint=True,
    checkpoint_interval=5
)
```

##### 因子研究配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `factor_pool_size` | int | 20 | 因子池大小 |
| `factor_selection_top_k` | int | 5 | 选择 top-K 因子 |
| `factor_ic_threshold` | float | 0.03 | IC 阈值 |
| `factor_ir_threshold` | float | 0.5 | IR 阈值 |

#### 方法

##### from_yaml(config_file: str) → RDAgentConfig

从 YAML 文件加载配置。

**参数**:
- `config_file` (str): YAML 配置文件路径

**返回**: `RDAgentConfig` 实例

**示例**:
```python
config = RDAgentConfig.from_yaml("config/rdagent_limitup.yaml")
```

##### from_json(config_file: str) → RDAgentConfig

从 JSON 文件加载配置。

**参数**:
- `config_file` (str): JSON 配置文件路径

**返回**: `RDAgentConfig` 实例

##### validate() → bool

验证配置有效性。

**返回**: `bool` - 配置是否有效

**验证项**:
- RD-Agent 路径存在性
- LLM 提供商有效性
- API 密钥完整性
- 研究模式有效性
- 参数范围检查

**示例**:
```python
config = RDAgentConfig(llm_provider="openai", llm_api_key="sk-...")

if config.validate():
    print("配置有效")
else:
    print("配置验证失败")
```

##### to_dict() → Dict[str, Any]

转换为字典。

**返回**: `Dict[str, Any]` - 配置字典

##### get_qlib_config() → Dict[str, Any]

获取 Qlib 相关配置。

**返回**: 包含数据周期等信息的字典

---

## 官方集成

### OfficialRDAgentManager

官方 RD-Agent 组件管理器,负责初始化和管理 FactorRDLoop/ModelRDLoop。

**模块**: `rd_agent.official_integration`

#### 初始化

```python
from rd_agent.official_integration import OfficialRDAgentManager

config = {
    "llm_model": "gpt-4-turbo",
    "max_iterations": 10,
    "checkpoint_path": "./checkpoints/factor.pkl"
}

manager = OfficialRDAgentManager(config)
```

**参数**:
- `config` (Dict[str, Any]): 配置字典
- `checkpoint_path` (Optional[str]): Checkpoint 路径 (可选)

#### 主要方法

##### get_factor_loop(resume: bool = False) → FactorRDLoop

获取因子研发循环实例 (懒加载)。

**参数**:
- `resume` (bool): 是否从 checkpoint 恢复,默认 False

**返回**: `FactorRDLoop` 实例

**P0-1 会话恢复示例**:
```python
manager = OfficialRDAgentManager(config)

# 首次创建
factor_loop = manager.get_factor_loop()

# 从 checkpoint 恢复
factor_loop = manager.get_factor_loop(resume=True)
```

##### get_model_loop(resume: bool = False) → ModelRDLoop

获取模型研发循环实例 (懒加载)。

**参数**:
- `resume` (bool): 是否从 checkpoint 恢复,默认 False

**返回**: `ModelRDLoop` 实例

##### resume_from_checkpoint(checkpoint_path: str = None, mode: str = "factor")

从 checkpoint 恢复研发循环。

**参数**:
- `checkpoint_path` (str, 可选): Checkpoint 文件路径,若不提供则使用配置中的路径
- `mode` (str): 恢复模式,"factor" 或 "model"

**返回**: 恢复的 Loop 实例

**异常**:
- `OfficialIntegrationError`: 恢复失败
- `ValueError`: 不支持的 mode

**P0-1 示例**:
```python
# 方式 1: 使用配置中的 checkpoint_path
config = {"checkpoint_path": "./checkpoints/factor.pkl"}
manager = OfficialRDAgentManager(config)

factor_loop = manager.resume_from_checkpoint(mode="factor")

# 方式 2: 指定 checkpoint_path
factor_loop = manager.resume_from_checkpoint(
    checkpoint_path="./saved_states/iter_10.pkl",
    mode="factor"
)
```

##### validate_config(config: Dict[str, Any]) (静态方法)

验证配置完整性。

**参数**:
- `config` (Dict[str, Any]): 配置字典

**异常**:
- `ConfigValidationError`: 配置无效

---

## 涨停板集成

### LimitUpRDAgentIntegration

涨停板专属 RD-Agent 集成,针对"一进二"选股策略优化。

**模块**: `rd_agent.limitup_integration`

#### 初始化

```python
from rd_agent.limitup_integration import LimitUpRDAgentIntegration

# 使用默认配置
integration = LimitUpRDAgentIntegration()

# 指定配置文件
integration = LimitUpRDAgentIntegration("config/rdagent_limitup.yaml")
```

#### 主要方法

##### discover_limit_up_factors(start_date: str, end_date: str, n_factors: int = 10) → List[Dict]

发现涨停板因子。

**参数**:
- `start_date` (str): 开始日期 (YYYY-MM-DD)
- `end_date` (str): 结束日期 (YYYY-MM-DD)
- `n_factors` (int): 生成因子数量,默认 10

**返回**: 因子列表,每个因子包含:
- `name` (str): 因子名称
- `code` (str): 因子代码
- `category` (str): 因子类别
- `performance` (Dict): 性能指标
  - `ic` (float): 信息系数
  - `ir` (float): 信息比率
  - `next_day_limit_up_rate` (float): 次日涨停率 (P0-6)
  - `sample_count` (int): 样本数

**P0-3 示例** (真实因子评估):
```python
integration = LimitUpRDAgentIntegration()

factors = await integration.discover_limit_up_factors(
    start_date="2024-01-01",
    end_date="2024-01-31",
    n_factors=10
)

for factor in factors:
    print(f"因子: {factor['name']}")
    print(f"  IC: {factor['performance']['ic']:.4f}")
    print(f"  次日涨停率: {factor['performance']['next_day_limit_up_rate']:.2%}")
    print(f"  样本数: {factor['performance']['sample_count']}")
```

---

## 数据接口

### LimitUpDataInterface

涨停板数据接口,提供封单金额、连板天数、题材热度等数据。

**模块**: `rd_agent.limit_up_data`

#### 初始化

```python
from rd_agent.limit_up_data import LimitUpDataInterface

data_interface = LimitUpDataInterface(data_source="qlib")
```

#### P0-5: 核心方法

##### get_seal_amount(symbol: str, date: str, prev_close: float) → float

计算涨停板封单金额。

**方法**:
1. 分钟数据精确计算 (首选)
2. 日线数据近似估算 (fallback)

**参数**:
- `symbol` (str): 股票代码 (如 "000001.SZ")
- `date` (str): 日期 (YYYY-MM-DD)
- `prev_close` (float): 前一交易日收盘价

**返回**: `float` - 封单金额 (万元)

**缓存**: 自动缓存,避免重复计算

**示例**:
```python
seal_amount = data_interface.get_seal_amount(
    symbol="000001.SZ",
    date="2024-01-15",
    prev_close=10.0
)

print(f"封单金额: {seal_amount:.2f} 万元")
```

##### get_continuous_board(symbol: str, date: str, lookback_days: int = 30) → int

计算连续涨停天数。

**算法**:
1. 从 date 往前遍历
2. 计数连续涨停天数 (涨幅 >= 9.9% 且 收盘价 == 最高价)
3. 遇到非涨停则停止

**参数**:
- `symbol` (str): 股票代码
- `date` (str): 日期 (YYYY-MM-DD)
- `lookback_days` (int): 回望天数,默认 30

**返回**: `int` - 连续涨停天数 (1=首板, 2=二板, 0=未涨停)

**示例**:
```python
continuous_days = data_interface.get_continuous_board(
    symbol="000001.SZ",
    date="2024-01-15"
)

if continuous_days == 1:
    print("首板")
elif continuous_days == 2:
    print("二板")
elif continuous_days >= 3:
    print(f"{continuous_days}连板")
```

##### get_concept_heat(symbol: str, date: str) → float

计算股票所属题材的热度。

**热度定义**: 同题材当日涨停股票数量

**参数**:
- `symbol` (str): 股票代码
- `date` (str): 日期 (YYYY-MM-DD)

**返回**: `float` - 热度值 (涨停股票数量)

**示例**:
```python
concept_heat = data_interface.get_concept_heat(
    symbol="000001.SZ",
    date="2024-01-15"
)

print(f"题材热度: {concept_heat:.0f} 只涨停")
```

##### get_limit_up_features(symbols: List[str], date: str, lookback_days: int = 20) → pd.DataFrame

获取涨停相关特征 (P0-5 集成)。

**参数**:
- `symbols` (List[str]): 股票代码列表
- `date` (str): 日期 (YYYY-MM-DD)
- `lookback_days` (int): 回望天数,默认 20

**返回**: `pd.DataFrame` - 特征矩阵,索引为股票代码,列包括:
- `limit_up_strength` (float): 涨停强度 (0-100)
- `seal_quality` (float): 封板质量 (0-10)
- `seal_amount` (float): 封单金额 (万元) **[P0-5]**
- `continuous_board` (int): 连板天数 **[P0-5]**
- `concept_heat` (float): 题材热度 **[P0-5]**
- `volume_surge` (float): 量比
- `market_cap` (float): 流通市值 (亿元)
- `turnover_rate` (float): 换手率
- `industry` (str): 行业

**示例**:
```python
symbols = ["000001.SZ", "000002.SZ"]
features = data_interface.get_limit_up_features(
    symbols=symbols,
    date="2024-01-15",
    lookback_days=20
)

print(features[['seal_amount', 'continuous_board', 'concept_heat']])
```

---

## 代码沙盒

### CodeSandbox

安全代码执行沙盒,防止代码注入攻击。

**模块**: `rd_agent.code_sandbox`

#### 初始化

```python
from rd_agent.code_sandbox import CodeSandbox, SecurityLevel

sandbox = CodeSandbox(
    security_level=SecurityLevel.STRICT,
    timeout=5,
    enable_logging=True
)
```

**参数**:
- `security_level` (SecurityLevel): 安全级别 (STRICT/MODERATE/PERMISSIVE)
- `timeout` (int): 执行超时时间 (秒),默认 5
- `enable_logging` (bool): 是否启用日志,默认 True

#### 主要方法

##### execute(code: str, context: Dict[str, Any], allowed_modules: List[str] = None) → CodeExecutionResult

安全执行代码。

**参数**:
- `code` (str): 要执行的 Python 代码
- `context` (Dict[str, Any]): 执行上下文 (可用变量)
- `allowed_modules` (List[str], 可选): 额外允许的模块列表

**返回**: `CodeExecutionResult` 对象,包含:
- `success` (bool): 是否执行成功
- `locals` (Dict[str, Any]): 执行后的局部变量
- `error` (str, 可选): 错误信息
- `warnings` (List[str]): 警告列表

**安全机制**:
1. AST 静态分析 (白名单检查)
2. 关键字扫描
3. 限定命名空间 (safe builtins)
4. 执行超时控制 (Unix)
5. 异常捕获

**示例**:
```python
result = sandbox.execute(
    code="""
result = df['close'].mean()
factor = df['close'] / df['volume']
""",
    context={'df': dataframe}
)

if result.success:
    print(f"Result: {result.locals['result']}")
    print(f"Factor: {result.locals['factor']}")
else:
    print(f"Error: {result.error}")
    print(f"Warnings: {result.warnings}")
```

#### 便捷函数

##### execute_safe(code: str, context: Dict[str, Any], timeout: int = 5) → CodeExecutionResult

快捷方式,使用默认安全设置执行代码。

**示例**:
```python
from rd_agent.code_sandbox import execute_safe

result = execute_safe(
    code="result = df['close'].mean()",
    context={'df': dataframe},
    timeout=5
)
```

---

## 示例代码

### 完整因子发现流程

```python
import asyncio
from rd_agent.limitup_integration import LimitUpRDAgentIntegration

async def discover_factors():
    # 1. 初始化
    integration = LimitUpRDAgentIntegration()
    
    # 2. 因子发现 (P0-3: 真实评估)
    factors = await integration.discover_limit_up_factors(
        start_date="2024-01-01",
        end_date="2024-01-31",
        n_factors=10
    )
    
    # 3. 结果分析
    for factor in factors:
        perf = factor['performance']
        
        print(f"\n因子: {factor['name']}")
        print(f"类别: {factor['category']}")
        print(f"IC: {perf['ic']:.4f}")
        print(f"IR: {perf['ir']:.4f}")
        print(f"次日涨停率: {perf.get('next_day_limit_up_rate', 0):.2%}")
        print(f"样本数: {perf['sample_count']}")

# 运行
asyncio.run(discover_factors())
```

### P0-1: 会话恢复

```python
from rd_agent.official_integration import OfficialRDAgentManager

# 配置 checkpoint
config = {
    "checkpoint_path": "./checkpoints/factor_loop.pkl",
    "enable_auto_checkpoint": True,
    "checkpoint_interval": 5,
    "max_iterations": 20
}

manager = OfficialRDAgentManager(config)

# 从 checkpoint 恢复
factor_loop = manager.resume_from_checkpoint(mode="factor")

# 继续研发
# factor_loop.run()
```

### P0-5: 数据字段完整性

```python
from rd_agent.limit_up_data import LimitUpDataInterface

data_interface = LimitUpDataInterface()

symbols = ["000001.SZ", "000002.SZ", "000003.SZ"]
date = "2024-01-15"

# 获取完整特征
features = data_interface.get_limit_up_features(
    symbols=symbols,
    date=date
)

# 验证 P0-5 新增字段
print("封单金额:", features['seal_amount'].values)
print("连板天数:", features['continuous_board'].values)
print("题材热度:", features['concept_heat'].values)
```

### 代码沙盒安全执行

```python
from rd_agent.code_sandbox import execute_safe
import pandas as pd

# 测试数据
test_df = pd.DataFrame({
    'close': [10.0, 11.0, 12.0],
    'volume': [1000, 1100, 1200]
})

# 安全执行因子代码
factor_code = """
factor = df['close'] / df['volume']
result = factor.mean()
"""

result = execute_safe(
    code=factor_code,
    context={'df': test_df},
    timeout=5
)

if result.success:
    print(f"因子值: {result.locals['factor']}")
    print(f"平均值: {result.locals['result']}")
else:
    print(f"执行失败: {result.error}")
```

---

## 异常处理

### 常见异常

| 异常类 | 模块 | 说明 |
|--------|------|------|
| `OfficialIntegrationError` | official_integration | 官方集成错误 |
| `ConfigValidationError` | config | 配置验证错误 |
| `SecurityError` | code_sandbox | 代码安全检查失败 |
| `TimeoutError` | code_sandbox | 代码执行超时 |

### 异常处理示例

```python
from rd_agent.official_integration import (
    OfficialRDAgentManager,
    OfficialIntegrationError,
    ConfigValidationError
)

try:
    # 验证配置
    OfficialRDAgentManager.validate_config(config)
    
    # 创建管理器
    manager = OfficialRDAgentManager(config)
    
    # 获取 Loop
    factor_loop = manager.get_factor_loop(resume=True)

except ConfigValidationError as e:
    print(f"配置验证失败: {e}")

except OfficialIntegrationError as e:
    print(f"集成错误: {e}")

except Exception as e:
    print(f"未知错误: {e}")
```

---

## 最佳实践

### 1. 配置管理

- ✅ 使用 YAML 文件管理配置
- ✅ 敏感信息使用环境变量
- ✅ 在生产环境验证配置

```python
# 推荐方式
config = RDAgentConfig.from_yaml("config/prod.yaml")
if not config.validate():
    raise ValueError("配置无效")
```

### 2. 会话恢复

- ✅ 定期保存 checkpoint
- ✅ 设置合理的 checkpoint_interval
- ✅ 妥善管理 checkpoint 文件

```python
# 推荐配置
config = RDAgentConfig(
    checkpoint_path="./checkpoints/factor_{timestamp}.pkl",
    enable_auto_checkpoint=True,
    checkpoint_interval=5  # 每5轮保存
)
```

### 3. 代码安全

- ✅ 始终使用 CodeSandbox 执行不可信代码
- ✅ 设置合理的超时时间
- ✅ 记录执行日志

```python
# 推荐方式
result = execute_safe(
    code=llm_generated_code,
    context={'df': data},
    timeout=10  # 充足的时间
)

if not result.success:
    logger.error(f"代码执行失败: {result.error}")
```

---

## 性能优化

### 缓存机制

LimitUpDataInterface 自动缓存计算结果:
- 封单金额: `_seal_amount_cache`
- 连板天数: `_continuous_board_cache`
- 股票概念: `_concept_cache`

**清理缓存**:
```python
data_interface._seal_amount_cache.clear()
data_interface._continuous_board_cache.clear()
```

### 批量操作

优先使用批量接口:
```python
# ✅ 推荐: 批量获取
features = data_interface.get_limit_up_features(symbols, date)

# ❌ 避免: 逐个获取
for symbol in symbols:
    seal_amount = data_interface.get_seal_amount(symbol, date, prev_close)
```

---

## 版本兼容性

### Python 版本

- **要求**: Python >= 3.8
- **推荐**: Python 3.11+

### 依赖版本

- numpy >= 1.20
- pandas >= 1.3
- qlib >= 0.9
- akshare >= 1.10

---

**文档版本**: v1.0  
**维护者**: 麒麟量化团队  
**反馈**: 请提交 Issue 或 PR

# ⚙️ 配置指南

完整的系统配置说明文档。

---

## 配置文件结构

```
config/
├── config.yaml              # 主配置文件
├── tradingagents.yaml       # TradingAgents配置
├── rdagent_limitup.yaml     # RD-Agent涨停板配置
└── qlib_enhanced.yaml       # Qlib增强配置
```

---

## 主配置文件

### config.yaml

```yaml
# 系统模式
environment: "development"  # development, production

# LLM配置
llm:
  provider: "openai"        # openai, azure, anthropic
  model: "gpt-5-thinking-all"
  api_key: "${LLM_API_KEY}"  # 支持环境变量
  api_base: "https://api.tu-zi.com"
  timeout: 30
  max_retries: 3
  temperature: 0.7

# 日志配置
logging:
  level: "INFO"             # DEBUG, INFO, WARNING, ERROR
  file: "logs/qilin_stack.log"
  max_size: "100MB"
  backup_count: 10
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 数据源配置
data_sources:
  primary: "qlib"           # qlib, akshare, tushare
  fallback: ["akshare", "tushare"]
  cache_enabled: true
  cache_ttl: 3600           # 秒

# 决策引擎配置
decision_engine:
  weights:
    qlib: 0.40
    trading_agents: 0.35
    rd_agent: 0.25
  
  thresholds:
    min_confidence: 0.5     # 最小置信度
    min_strength: 0.3       # 最小信号强度
  
  risk_filters:
    max_position_size: 0.2  # 最大单次仓位20%
    max_single_stock: 0.1   # 单股最大10%
    max_correlation: 0.8    # 最大相关性

# 监控配置
monitoring:
  enabled: true
  port: 8000
  metrics_interval: 60      # 秒
  export_format: "prometheus"

# 性能配置
performance:
  worker_threads: 8
  max_concurrent_requests: 100
  connection_pool_size: 20
  timeout: 30
```

---

## 数据源配置

### Qlib配置

```yaml
qlib:
  provider_uri: "~/.qlib/qlib_data/cn_data"
  region: "cn"
  market: "csi300"          # csi300, csi500, all
  
  features:
    - "$close"
    - "$volume"
    - "$open"
    - "$high"
    - "$low"
  
  model:
    type: "LGBModel"
    params:
      num_leaves: 31
      learning_rate: 0.05
      n_estimators: 100
```

### AKShare配置

```yaml
akshare:
  cache_dir: "./cache/akshare"
  cache_ttl: 3600
  rate_limit: 60            # 每分钟请求数
  timeout: 10
  retry_times: 3
  retry_delay: 2            # 秒
```

### Tushare配置

```yaml
tushare:
  token: "${TUSHARE_TOKEN}"
  cache_dir: "./cache/tushare"
  cache_ttl: 3600
  api_url: "http://api.tushare.pro"
```

---

## 决策引擎高级配置

### 权重动态优化

```yaml
weight_optimizer:
  enabled: true
  strategy: "daily"         # daily, weekly, monthly
  
  constraints:
    min_weight: 0.1
    max_weight: 0.6
  
  metrics:
    - accuracy
    - f1_score
    - sharpe_ratio
    - win_rate
  
  weights_combination:       # 指标权重
    accuracy: 0.3
    f1_score: 0.3
    sharpe_ratio: 0.3
    win_rate: 0.1
```

### 风险管理

```yaml
risk_management:
  position_sizing:
    method: "kelly"          # kelly, fixed, volatility
    kelly_fraction: 0.5
    max_leverage: 2.0
  
  stop_loss:
    method: "atr"            # atr, fixed, trailing
    atr_multiplier: 2.0
    fixed_percent: 0.05
  
  take_profit:
    method: "trailing"
    trailing_percent: 0.02
    fixed_percent: 0.10
```

---

## 自适应系统配置

### 市场状态检测

```yaml
market_state_detector:
  indicators:
    ma_short: 20
    ma_long: 60
    rsi_period: 14
    macd_fast: 12
    macd_slow: 26
    macd_signal: 9
  
  thresholds:
    bull_rsi: 60
    bear_rsi: 40
    volatility_threshold: 0.02
```

### 策略自适应

```yaml
adaptive_strategy:
  bull_market:
    position_size: 0.7
    stop_loss: -0.08
    take_profit: 0.15
    holding_period: 10
  
  bear_market:
    position_size: 0.3
    stop_loss: -0.03
    take_profit: 0.08
    holding_period: 3
  
  sideways:
    position_size: 0.4
    stop_loss: -0.04
    take_profit: 0.10
    holding_period: 5
  
  volatile:
    position_size: 0.2
    stop_loss: -0.02
    take_profit: 0.06
    holding_period: 2
```

---

## TradingAgents配置

### 智能体配置

```yaml
trading_agents:
  analyst_agent:
    model: "gpt-5-thinking-all"
    temperature: 0.7
    max_tokens: 2000
    tools:
      - market_data
      - technical_analysis
      - news_sentiment
  
  risk_agent:
    model: "gpt-5-thinking-all"
    temperature: 0.3
    max_tokens: 1500
    risk_tolerance: "moderate"  # conservative, moderate, aggressive
  
  execution_agent:
    model: "gpt-5-thinking-all"
    temperature: 0.1
    max_tokens: 1000
    order_type: "limit"         # market, limit
```

---

## RD-Agent配置

### 因子研究配置

```yaml
rd_agent:
  # P0-1: 会话恢复配置
  checkpoint:
    checkpoint_path: "./checkpoints/limitup_factor_loop.pkl"  # Checkpoint文件路径
    enable_auto_checkpoint: true                               # 启用自动保存
    checkpoint_interval: 5                                     # 每5轮保存一次
  
  # 因子研究配置
  factor_research:
    max_factors: 50
    min_ic: 0.03
    min_icir: 0.5
    correlation_threshold: 0.8
    
    # P0-6: 扩展字段配置
    factor_categories:                  # 因子类别列表
      - "technical"                     # 技术指标
      - "momentum"                      # 动量因子
      - "volume"                        # 成交量因子
      - "limit_up"                      # 涨停板专属因子
      - "sentiment"                     # 情绪因子
    
    prediction_targets:                 # 预测目标列表
      - "next_day_return"               # 次日收益率
      - "next_day_limit_up"             # 次日是否涨停
  
  model_experiment:
    models:
      - LightGBM
      - XGBoost
      - RandomForest
    cv_folds: 5
    optimization_metric: "ic"
  
  llm_config:
    model: "gpt-5-thinking-all"
    api_base: "https://api.tu-zi.com"
```

### P0-1: 会话恢复详细说明

会话恢复功能允许从 checkpoint 恢复研发流程,防止意外中断导致进度丢失。

#### checkpoint_path

**类型**: `Optional[str]`  
**默认值**: `None`  
**说明**: Checkpoint 文件保存路径

**推荐配置**:
```yaml
# 固定路径
checkpoint_path: "./checkpoints/factor_loop.pkl"

# 动态路径 (带时间戳)
checkpoint_path: "./checkpoints/factor_{timestamp}.pkl"
```

**使用场景**:
- ✅ 长时间运行的因子发现任务 (> 30分钟)
- ✅ 需要增量式研发的场景
- ✅ 不稳定网络环境下的 LLM 调用
- ❌ 短任务 (< 5轮) 不需要 checkpoint

#### enable_auto_checkpoint

**类型**: `bool`  
**默认值**: `True`  
**说明**: 是否启用自动 checkpoint 保存

**配置示例**:
```yaml
# 启用自动保存 (推荐)
enable_auto_checkpoint: true

# 禁用自动保存 (需手动保存)
enable_auto_checkpoint: false
```

#### checkpoint_interval

**类型**: `int`  
**默认值**: `5`  
**说明**: Checkpoint 保存间隔 (单位: 研发轮次)

**推荐值**:
- 短任务 (< 10 轮): `2-3`
- 中等任务 (10-50 轮): `5`
- 长任务 (> 50 轮): `10`

**示例**:
```yaml
# 每5轮保存一次 (推荐)
checkpoint_interval: 5
```

**恢复示例**:
```python
from rd_agent.official_integration import OfficialRDAgentManager

config = {
    "checkpoint_path": "./checkpoints/factor_loop.pkl",
    "enable_auto_checkpoint": True,
    "checkpoint_interval": 5
}

manager = OfficialRDAgentManager(config)

# 从 checkpoint 恢复
factor_loop = manager.resume_from_checkpoint(mode="factor")
```

### P0-6: 扩展字段详细说明

涨停板专属配置,支持自定义因子类别和预测目标。

#### factor_categories

**类型**: `List[str]`  
**默认值**: `["technical", "momentum", "volume", "limit_up"]`  
**说明**: 因子类别列表,指导 LLM 生成特定类别的因子

**支持类别**:
- `technical`: 技术指标 (RSI, MACD, 布林带等)
- `momentum`: 动量因子 (收益率, 动量指标)
- `volume`: 成交量因子 (量比, 换手率)
- `limit_up`: 涨停板专属因子 (封单金额、连板天数、题材热度)
- `fundamental`: 基本面因子 (市盈率, 市净率)
- `sentiment`: 情绪因子 (题材热度, 资金流向)

**配置示例**:
```yaml
# 涨停板策略: 强调 limit_up 和 sentiment
factor_categories:
  - "limit_up"          # 优先级最高
  - "sentiment"
  - "momentum"
  - "volume"

# 全市场策略: 均衡配置
factor_categories:
  - "technical"
  - "momentum"
  - "volume"
  - "fundamental"
```

**影响**: LLM 会根据类别列表生成对应的因子代码,类别越靠前优先级越高。

#### prediction_targets

**类型**: `List[str]`  
**默认值**: `["next_day_return", "next_day_limit_up"]`  
**说明**: 预测目标列表,定义因子要预测的指标

**支持目标**:
- `next_day_return`: 次日收益率 (连续值)
- `next_day_limit_up`: 次日是否涨停 (0/1 分类)
- `intraday_return`: 日内收益率
- `max_drawdown`: 最大回撤

**配置示例**:
```yaml
# 涨停板策略: 关注次日涨停
prediction_targets:
  - "next_day_limit_up"   # 主要目标
  - "next_day_return"     # 辅助目标

# 全市场策略: 关注收益率
prediction_targets:
  - "next_day_return"
  - "intraday_return"
```

**影响**: 因子评估时会计算对应的性能指标 (如 `next_day_limit_up_rate`)

---

## 监控配置

### Prometheus配置

```yaml
prometheus:
  scrape_interval: 15s
  evaluation_interval: 15s
  scrape_configs:
    - job_name: 'qilin-stack'
      static_configs:
        - targets: ['localhost:8000']
      metrics_path: '/metrics'
```

### Grafana配置

```yaml
grafana:
  port: 3000
  admin_user: "admin"
  admin_password: "${GRAFANA_PASSWORD}"
  dashboards:
    - name: "Qilin Stack Overview"
      file: "grafana/dashboards/overview.json"
```

### 告警规则

```yaml
alerting:
  rules:
    - name: "HighErrorRate"
      expr: "rate(error_count_total[5m]) > 0.1"
      for: "5m"
      severity: "critical"
      annotations:
        summary: "错误率过高"
    
    - name: "LowConfidence"
      expr: "avg(signal_confidence) < 0.4"
      for: "10m"
      severity: "warning"
      annotations:
        summary: "信号置信度过低"
```

---

## 环境变量

- Windows PowerShell 环境变量与启动指南：参见 [docs/ENV_SETUP_WINDOWS.md](ENV_SETUP_WINDOWS.md)
- 安全与合规配置：参见 [docs/security/audit_compliance.md](security/audit_compliance.md)
- 在线服务/集成指南：
  - Qlib 功能分析与缺失项： [docs/QLIB_FEATURE_ANALYSIS.md](QLIB_FEATURE_ANALYSIS.md)
  - TradingAgents 集成说明： [tradingagents_integration/README.md](../tradingagents_integration/README.md)
  - RD-Agent 集成指南： [docs/RD-Agent_Integration_Guide.md](RD-Agent_Integration_Guide.md)
- 监控/运维：
  - 监控指标： [docs/MONITORING_METRICS.md](MONITORING_METRICS.md)
  - SLO/接受标准： [docs/sla/slo.yaml](sla/slo.yaml)
  - 部署指南： [docs/DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

### 必需变量

```bash
# LLM服务
export LLM_API_KEY="your-api-key"
export LLM_API_BASE="https://api.tu-zi.com"

# Qlib数据路径
export QLIB_DATA_PATH="~/.qlib/qlib_data/cn_data"
```

### 可选变量

```bash
# AKShare
export AKSHARE_TOKEN="your-token"

# Tushare
export TUSHARE_TOKEN="your-token"

# 监控
export PROMETHEUS_PORT="8000"
export GRAFANA_PASSWORD="your-password"

# 数据库
export DB_HOST="localhost"
export DB_PORT="5432"
export DB_NAME="qilin_stack"
export DB_USER="admin"
export DB_PASSWORD="your-password"
```

---

## 配置加载

### Python代码加载配置

```python
import yaml
from pathlib import Path

def load_config(config_file='config/config.yaml'):
    """加载配置文件"""
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 替换环境变量
    config = _replace_env_vars(config)
    
    return config

def _replace_env_vars(config):
    """递归替换环境变量"""
    import os
    import re
    
    if isinstance(config, dict):
        return {k: _replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_replace_env_vars(item) for item in config]
    elif isinstance(config, str):
        # 替换 ${VAR_NAME} 格式
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, config)
        for var_name in matches:
            var_value = os.getenv(var_name, '')
            config = config.replace(f'${{{var_name}}}', var_value)
        return config
    else:
        return config

# 使用示例
config = load_config()
print(f"LLM Model: {config['llm']['model']}")
```

---

## 配置验证

### 验证脚本

创建 `scripts/validate_config.py`:

```python
import yaml
from pathlib import Path

def validate_config(config_file='config/config.yaml'):
    """验证配置文件"""
    try:
        config = yaml.safe_load(open(config_file))
        
        # 检查必需字段
        required_fields = ['llm', 'data_sources', 'decision_engine']
        for field in required_fields:
            assert field in config, f"缺少必需字段: {field}"
        
        # 检查权重总和
        weights = config['decision_engine']['weights']
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, f"权重总和应为1.0，实际为{total}"
        
        print("✅ 配置验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 配置验证失败: {e}")
        return False

if __name__ == '__main__':
    validate_config()
```

运行验证：
```bash
python scripts/validate_config.py
```

---

## 最佳实践

### 1. 环境分离

```
config/
├── development.yaml    # 开发环境
├── staging.yaml        # 测试环境
└── production.yaml     # 生产环境
```

加载对应配置：
```python
import os
env = os.getenv('ENVIRONMENT', 'development')
config = load_config(f'config/{env}.yaml')
```

### 2. 敏感信息管理

- ❌ 不要将API密钥硬编码在配置文件中
- ✅ 使用环境变量
- ✅ 使用密钥管理服务（AWS Secrets Manager, Azure Key Vault）

### 3. 配置版本控制

- ✅ 提交配置文件模板
- ❌ 不要提交包含真实密钥的配置
- ✅ 使用 `.gitignore` 忽略本地配置

```gitignore
config/local.yaml
config/**/secrets.yaml
.env
```

---

## 故障排查

### 常见配置问题

**Q: "LLM API key not found"**
```bash
# 检查环境变量
echo $LLM_API_KEY

# 设置环境变量
export LLM_API_KEY="your-key"
```

**Q: "Invalid weights, sum must be 1.0"**
```yaml
# 确保权重总和为1.0
decision_engine:
  weights:
    qlib: 0.40
    trading_agents: 0.35
    rd_agent: 0.25  # 0.40 + 0.35 + 0.25 = 1.0
```

**Q: "Data source not configured"**
```yaml
# 检查数据源配置
data_sources:
  primary: "qlib"
  fallback: ["akshare"]
```

---

## 下一步

- 📖 [快速开始](QUICKSTART.md)
- 🚢 [部署指南](DEPLOYMENT.md)
- 📊 [监控系统](MONITORING.md)

---

**有问题？** 查看 [FAQ](FAQ.md) 或提交 [Issue](https://github.com/your-repo/issues)

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
  factor_research:
    max_factors: 50
    min_ic: 0.03
    min_icir: 0.5
    correlation_threshold: 0.8
  
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

# 📋 Qilin Stack 完整实施计划

## 总体目标

将当前的**原型系统**升级为**生产就绪系统**，具备完整的测试、文档、监控、性能优化和部署能力。

**总工期**: 25-35 个工作日  
**团队规模**: 1-2 人  
**优先级**: 高 → 低（阶段1最高）

---

## 📊 进度总览

| 阶段 | 任务数 | 预计天数 | 状态 | 优先级 |
|------|--------|----------|------|--------|
| 阶段1: 测试体系 | 3 | 4-6天 | 🔄 待开始 | 🔴 P0 |
| 阶段2: 文档完善 | 3 | 5天 | 🔄 待开始 | 🔴 P0 |
| 阶段3: 数据接入 | 3 | 3-5天 | 🔄 待开始 | 🟠 P1 |
| 阶段4: 监控部署 | 3 | 2天 | 🔄 待开始 | 🟠 P1 |
| 阶段5: 性能优化 | 3 | 4-5天 | 🔄 待开始 | 🟡 P2 |
| 阶段6: 回测系统 | 2 | 5-7天 | 🔄 待开始 | 🟡 P2 |
| 阶段7: 生产部署 | 1 | 2-3天 | 🔄 待开始 | 🟢 P3 |

**图例**:
- 🔴 P0 = 必须完成，系统基础
- 🟠 P1 = 重要，影响可用性
- 🟡 P2 = 有价值，提升体验
- 🟢 P3 = 可选，长期优化

---

## 阶段1: 完善测试体系 (4-6天) 🔴

### 目标
建立完整的自动化测试体系，确保代码质量和系统稳定性。

### 任务清单

#### 1.1 单元测试 (2-3天)
**文件结构**:
```
tests/
├── unit/
│   ├── test_decision_engine.py      # 决策引擎测试
│   ├── test_signal_generators.py    # 信号生成器测试
│   ├── test_weight_optimizer.py     # 权重优化器测试
│   ├── test_market_state.py         # 市场状态检测测试
│   ├── test_monitoring.py           # 监控系统测试
│   ├── test_data_pipeline.py        # 数据管道测试
│   └── test_system_bridge.py        # 系统桥接测试
├── integration/
│   └── test_end_to_end.py           # 端到端测试
├── fixtures/
│   ├── sample_data.py               # 测试数据
│   └── mock_responses.py            # Mock响应
├── conftest.py                      # pytest配置
└── requirements-test.txt            # 测试依赖
```

**测试覆盖目标**:
- 核心逻辑覆盖率: **80%+**
- 关键路径覆盖率: **100%**
- 边界条件测试: **完整**

**技术栈**:
- `pytest` - 测试框架
- `pytest-asyncio` - 异步测试
- `pytest-cov` - 覆盖率报告
- `pytest-mock` - Mock工具
- `hypothesis` - 属性测试（可选）

**关键测试用例**:
```python
# tests/unit/test_decision_engine.py
import pytest
from decision_engine.core import DecisionEngine, SignalType

@pytest.mark.asyncio
async def test_decision_engine_basic():
    """测试基本决策流程"""
    engine = DecisionEngine()
    decisions = await engine.make_decisions(['000001.SZ'], '2024-06-30')
    assert len(decisions) == 1
    assert decisions[0].final_signal in SignalType
    assert 0 <= decisions[0].confidence <= 1

@pytest.mark.asyncio
async def test_decision_engine_empty_symbols():
    """测试空股票列表"""
    engine = DecisionEngine()
    decisions = await engine.make_decisions([], '2024-06-30')
    assert len(decisions) == 0

@pytest.mark.asyncio
async def test_signal_fusion_weighted():
    """测试加权信号融合"""
    from decision_engine.core import SignalFuser, Signal
    fuser = SignalFuser()
    signals = [
        Signal(SignalType.BUY, 0.8, 0.9, 'qlib'),
        Signal(SignalType.SELL, 0.6, 0.7, 'trading_agents'),
        Signal(SignalType.HOLD, 0.5, 0.6, 'rd_agent')
    ]
    result = fuser.fuse_signals(signals)
    assert result.type in SignalType
    assert 0 <= result.confidence <= 1
```

**执行方式**:
```bash
# 运行所有测试
pytest tests/ -v

# 运行并生成覆盖率报告
pytest tests/ --cov=decision_engine --cov=adaptive_system --cov=monitoring --cov=data_pipeline --cov-report=html

# 运行特定模块
pytest tests/unit/test_decision_engine.py -v
```

---

#### 1.2 集成测试 (1-2天)
**目标**: 测试多模块协同工作

**测试场景**:
1. **完整决策流程**
   - 数据获取 → 信号生成 → 信号融合 → 决策输出 → 监控记录
2. **市场状态自适应**
   - 检测牛市 → 调整参数 → 生成决策
   - 检测熊市 → 调整参数 → 生成决策
3. **权重动态优化**
   - 性能评估 → 权重更新 → 新决策生成
4. **错误处理和降级**
   - 数据源失败 → 自动降级 → 继续运行
   - 单个系统失败 → 其他系统补偿

**示例测试**:
```python
# tests/integration/test_end_to_end.py
import pytest
from decision_engine.core import get_decision_engine
from adaptive_system.market_state import AdaptiveStrategyAdjuster
from monitoring.metrics import get_monitor

@pytest.mark.asyncio
async def test_full_decision_pipeline():
    """测试完整决策流程"""
    # 1. 初始化
    engine = get_decision_engine()
    adjuster = AdaptiveStrategyAdjuster()
    monitor = get_monitor()
    
    # 2. 市场状态检测
    market_data = create_test_market_data()
    state = adjuster.detector.detect_state(market_data)
    assert state is not None
    
    # 3. 生成决策
    symbols = ['000001.SZ', '600000.SH']
    decisions = await engine.make_decisions(symbols, '2024-06-30')
    assert len(decisions) == 2
    
    # 4. 验证监控指标
    metrics = monitor.export_metrics()
    assert 'decision_made_total' in metrics
    assert monitor.get_summary()['total_decisions'] == 2
```

---

#### 1.3 CI配置 (1天)
**目标**: 自动化测试和代码质量检查

**GitHub Actions 配置**:
```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r tests/requirements-test.txt
    
    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --max-complexity=10 --max-line-length=120 --statistics
    
    - name: Type check with mypy
      run: |
        pip install mypy
        mypy decision_engine adaptive_system monitoring data_pipeline
    
    - name: Run tests with coverage
      run: |
        pytest tests/ --cov=. --cov-report=xml --cov-report=html
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true

  quality:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install pylint black isort
    
    - name: Check code formatting
      run: |
        black --check .
        isort --check-only .
    
    - name: Lint with pylint
      run: |
        pylint decision_engine adaptive_system monitoring data_pipeline --fail-under=8.0
```

**代码质量配置文件**:

```ini
# .flake8
[flake8]
max-line-length = 120
exclude = .git,__pycache__,build,dist,venv
ignore = E203,W503

# pyproject.toml (black + isort)
[tool.black]
line-length = 120
target-version = ['py39', 'py310', 'py311']

[tool.isort]
profile = "black"
line_length = 120

# .pylintrc
[MASTER]
max-line-length=120
disable=missing-docstring,too-few-public-methods
```

**预期成果**:
- ✅ 每次提交自动运行测试
- ✅ Pull Request检查门禁
- ✅ 覆盖率报告自动生成
- ✅ 代码质量评分≥8.0

---

## 阶段2: 文档完善 (5天) 🔴

### 目标
提供完整的开发者文档和用户文档，降低使用门槛。

### 任务清单

#### 2.1 API文档 (2天)
**目标**: 自动生成专业的API文档

**工具**: Sphinx + autodoc + napoleon

**文档结构**:
```
docs/
├── source/
│   ├── conf.py                 # Sphinx配置
│   ├── index.rst               # 首页
│   ├── api/
│   │   ├── decision_engine.rst # 决策引擎API
│   │   ├── adaptive_system.rst # 自适应系统API
│   │   ├── monitoring.rst      # 监控系统API
│   │   └── data_pipeline.rst   # 数据管道API
│   ├── tutorials/
│   │   ├── quickstart.rst      # 快速开始
│   │   ├── basic_usage.rst     # 基础使用
│   │   └── advanced.rst        # 高级特性
│   └── guides/
│       ├── configuration.rst   # 配置指南
│       ├── deployment.rst      # 部署指南
│       └── troubleshooting.rst # 故障排查
├── build/
└── Makefile
```

**Docstring 规范** (Google Style):
```python
class DecisionEngine:
    """智能决策引擎，融合多个系统的交易信号。
    
    该类负责从Qlib、TradingAgents和RD-Agent三个系统获取信号，
    进行加权融合，并应用风险过滤规则，最终输出交易决策。
    
    Attributes:
        qlib_generator (QlibSignalGenerator): Qlib信号生成器
        ta_generator (TradingAgentsSignalGenerator): TradingAgents信号生成器
        rd_generator (RDAgentSignalGenerator): RD-Agent信号生成器
        fuser (SignalFuser): 信号融合器
        weights (Dict[str, float]): 系统权重配置
    
    Examples:
        基本使用::
        
            >>> import asyncio
            >>> from decision_engine.core import get_decision_engine
            >>> 
            >>> async def main():
            ...     engine = get_decision_engine()
            ...     decisions = await engine.make_decisions(
            ...         symbols=['000001.SZ', '600000.SH'],
            ...         date='2024-06-30'
            ...     )
            ...     for decision in decisions:
            ...         print(f"{decision.symbol}: {decision.final_signal.value}")
            >>> 
            >>> asyncio.run(main())
    
    Note:
        - 默认权重为 Qlib:40%, TradingAgents:35%, RD-Agent:25%
        - 可通过 `update_weights()` 方法动态调整
        - 所有信号生成均为异步操作
    """
    
    async def make_decisions(
        self,
        symbols: List[str],
        date: str,
        min_confidence: float = 0.5
    ) -> List[Decision]:
        """生成交易决策。
        
        Args:
            symbols (List[str]): 股票代码列表，例如 ['000001.SZ', '600000.SH']
            date (str): 决策日期，格式 'YYYY-MM-DD'
            min_confidence (float, optional): 最小置信度阈值。默认 0.5
        
        Returns:
            List[Decision]: 决策列表，每个决策包含：
                - symbol: 股票代码
                - final_signal: 最终信号（BUY/SELL/HOLD等）
                - confidence: 置信度 [0, 1]
                - strength: 信号强度 [0, 1]
                - reasoning: 决策推理说明
                - source_signals: 原始信号列表
        
        Raises:
            ValueError: 如果 date 格式不正确
            RuntimeError: 如果所有信号生成器都失败
        
        Examples:
            生成单个股票决策::
            
                >>> decisions = await engine.make_decisions(
                ...     symbols=['000001.SZ'],
                ...     date='2024-06-30',
                ...     min_confidence=0.6
                ... )
                >>> print(decisions[0].final_signal)
                SignalType.BUY
            
            批量生成决策::
            
                >>> symbols = ['000001.SZ', '600000.SH', '600519.SH']
                >>> decisions = await engine.make_decisions(symbols, '2024-06-30')
                >>> high_conf = [d for d in decisions if d.confidence > 0.7]
        
        Note:
            - 方法会并行调用三个信号生成器以提高效率
            - 低于 min_confidence 的信号会被过滤为 HOLD
            - 建议处理可能的异常情况
        """
        pass
```

**生成文档**:
```bash
# 初始化Sphinx
cd docs
sphinx-quickstart

# 配置conf.py
# 添加扩展: 'sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode'

# 生成HTML文档
make html

# 查看文档
open build/html/index.html
```

**预期成果**:
- ✅ 所有公共API有完整docstring
- ✅ 自动生成的HTML文档
- ✅ 代码示例可运行
- ✅ 交叉引用正确

---

#### 2.2 用户指南 (2天)
**目标**: 编写面向最终用户的使用文档

**内容大纲**:

**快速开始 (QUICKSTART.md)**:
```markdown
# 快速开始

## 5分钟上手

### 1. 安装
```bash
pip install -r requirements.txt
```

### 2. 配置
```yaml
# config/config.yaml
llm_provider: "openai"
llm_api_key: "your-key"
llm_api_base: "https://api.tu-zi.com"
```

### 3. 运行第一个决策
```python
import asyncio
from decision_engine.core import get_decision_engine

async def main():
    engine = get_decision_engine()
    decisions = await engine.make_decisions(['000001.SZ'], '2024-06-30')
    print(decisions[0].final_signal)

asyncio.run(main())
```

## 核心概念

### 信号类型
- BUY: 买入信号
- SELL: 卖出信号
- HOLD: 持有
- STRONG_BUY: 强买
- STRONG_SELL: 强卖

### 三大系统
1. **Qlib**: 量化模型预测
2. **TradingAgents**: 多智能体协同
3. **RD-Agent**: 因子研究和发现

### 信号融合
默认权重: Qlib 40%, TradingAgents 35%, RD-Agent 25%
```

**配置指南 (CONFIGURATION.md)**:
```markdown
# 配置指南

## 系统配置

### LLM配置
```yaml
llm_provider: "openai"  # 或 "azure", "anthropic"
llm_model: "gpt-5-thinking-all"
llm_api_key: "${LLM_API_KEY}"  # 支持环境变量
llm_api_base: "https://api.tu-zi.com"
llm_timeout: 30
llm_max_retries: 3
```

### 数据源配置
```yaml
data_sources:
  primary: "qlib"
  fallback: ["akshare", "tushare"]
  
qlib:
  provider_uri: "~/.qlib/qlib_data/cn_data"
  
akshare:
  cache_dir: "./cache/akshare"
  cache_ttl: 3600
```

### 决策引擎配置
```yaml
decision_engine:
  weights:
    qlib: 0.40
    trading_agents: 0.35
    rd_agent: 0.25
  
  thresholds:
    min_confidence: 0.5
    min_strength: 0.3
  
  risk_filters:
    max_position_size: 0.2
    max_single_stock: 0.1
```

## 环境变量

必需:
- `LLM_API_KEY`: LLM服务密钥
- `QLIB_DATA_PATH`: Qlib数据路径

可选:
- `AKSHARE_TOKEN`: AKShare令牌
- `TUSHARE_TOKEN`: Tushare令牌
- `PROMETHEUS_PORT`: 监控端口 (默认8000)
```

**最佳实践 (BEST_PRACTICES.md)**:
```markdown
# 最佳实践

## 性能优化

### 1. 批量处理
```python
# ❌ 不推荐：逐个处理
for symbol in symbols:
    decisions = await engine.make_decisions([symbol], date)

# ✅ 推荐：批量处理
decisions = await engine.make_decisions(symbols, date)
```

### 2. 缓存使用
```python
# 启用缓存
from data_pipeline.unified_data import UnifiedDataPipeline
pipeline = UnifiedDataPipeline(cache_enabled=True, cache_ttl=3600)
```

### 3. 并发控制
```python
# 限制并发数
import asyncio
semaphore = asyncio.Semaphore(10)

async def process_with_limit(symbol):
    async with semaphore:
        return await engine.make_decisions([symbol], date)
```

## 错误处理

### 健壮的错误处理
```python
from decision_engine.core import DecisionEngineError

try:
    decisions = await engine.make_decisions(symbols, date)
except DecisionEngineError as e:
    logger.error(f"决策失败: {e}")
    # 降级策略
    decisions = fallback_decisions(symbols)
except Exception as e:
    logger.critical(f"未知错误: {e}")
    raise
```

## 监控集成

### 记录关键指标
```python
from monitoring.metrics import get_monitor

monitor = get_monitor()

# 记录决策
for decision in decisions:
    monitor.record_decision(
        symbol=decision.symbol,
        decision=decision.final_signal.value,
        latency=elapsed_time,
        confidence=decision.confidence
    )

# 定期导出
metrics = monitor.export_metrics()
```
```

---

#### 2.3 部署文档 (1天)
**DEPLOYMENT_PRODUCTION.md**:
```markdown
# 生产环境部署指南

## 环境准备

### 系统要求
- Python 3.9+
- 8GB+ RAM
- 50GB+ 磁盘空间
- Linux (推荐 Ubuntu 20.04+)

### 依赖安装
```bash
# 系统依赖
sudo apt-get update
sudo apt-get install -y build-essential python3-dev

# Python依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 数据初始化

### Qlib数据下载
```bash
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

### 数据验证
```bash
python scripts/validate_data.py
```

## 配置优化

### 生产配置模板
```yaml
# config/production.yaml
environment: "production"

logging:
  level: "INFO"
  file: "/var/log/qilin_stack/app.log"
  max_size: "100MB"
  backup_count: 10

performance:
  worker_threads: 8
  max_concurrent_requests: 100
  connection_pool_size: 20

monitoring:
  enabled: true
  port: 8000
  metrics_interval: 60
```

### 安全配置
```bash
# 使用环境变量存储密钥
export LLM_API_KEY="your-secret-key"
export AKSHARE_TOKEN="your-token"

# 或使用secrets管理器
aws secretsmanager get-secret-value --secret-id qilin-stack-keys
```

## 服务部署

### 使用systemd
```ini
# /etc/systemd/system/qilin-stack.service
[Unit]
Description=Qilin Stack Decision Engine
After=network.target

[Service]
Type=simple
User=qilin
WorkingDirectory=/opt/qilin-stack
Environment="LLM_API_KEY=xxx"
ExecStart=/opt/qilin-stack/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# 启动服务
sudo systemctl enable qilin-stack
sudo systemctl start qilin-stack
sudo systemctl status qilin-stack
```

### 使用Docker
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  qilin-stack:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LLM_API_KEY=${LLM_API_KEY}
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    restart: unless-stopped
  
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

## 健康检查

### 健康检查端点
```python
# health_check.py
from fastapi import FastAPI
from monitoring.metrics import get_monitor

app = FastAPI()

@app.get("/health")
async def health():
    monitor = get_monitor()
    summary = monitor.get_summary()
    return {
        "status": "healthy" if summary['total_errors'] == 0 else "degraded",
        "uptime": summary['uptime'],
        "total_decisions": summary['total_decisions']
    }

@app.get("/ready")
async def ready():
    # 检查依赖服务
    from decision_engine.core import get_decision_engine
    engine = get_decision_engine()
    return {"ready": True}
```

### 监控检查
```bash
# 检查服务状态
curl http://localhost:8000/health

# 检查指标导出
curl http://localhost:8000/metrics
```

## 故障排查

### 常见问题

**Q: 决策延迟过高**
```bash
# 检查并发配置
# 增加worker数量
# 启用缓存

# 监控CPU/内存
htop
```

**Q: LLM调用失败**
```bash
# 检查网络连接
curl https://api.tu-zi.com

# 检查API密钥
echo $LLM_API_KEY

# 查看错误日志
tail -f /var/log/qilin_stack/app.log | grep ERROR
```

**Q: 数据加载失败**
```bash
# 验证数据完整性
python scripts/validate_data.py

# 重新下载数据
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

## 性能基准

### 预期性能指标
- 决策延迟: < 100ms (P95)
- 吞吐量: > 1000 决策/秒
- 内存使用: < 4GB
- CPU使用: < 50% (8核)

### 压测命令
```bash
# 使用locust进行压测
locust -f tests/load_test.py --host=http://localhost:8000
```
```

---

## 阶段3: 实际数据接入 (3-5天) 🟠

### 目标
接入真实数据源，替换模拟数据。

### 任务清单

#### 3.1 Qlib数据配置 (1-2天)
**任务**:
1. 下载并初始化Qlib cn_data
2. 配置provider_uri
3. 测试数据加载性能
4. 验证因子计算正确性

**实施步骤**:
```bash
# 1. 下载数据
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

# 2. 验证数据
python scripts/validate_qlib_data.py

# 3. 性能测试
python scripts/benchmark_qlib.py
```

**验证脚本**:
```python
# scripts/validate_qlib_data.py
import qlib
from qlib.data import D

qlib.init(provider_uri='~/.qlib/qlib_data/cn_data')

# 测试数据加载
instruments = D.instruments(market='csi300')
print(f"加载股票数: {len(instruments)}")

# 测试特征获取
features = D.features(instruments[:10], ['$close', '$volume'], '2024-01-01', '2024-06-30')
print(f"特征形状: {features.shape}")
print("✅ Qlib数据验证通过")
```

---

#### 3.2 AKShare集成 (1-2天)
**任务**:
1. 实现真实的AKShare API调用
2. 处理API限流（每分钟60次）
3. 实现重试机制和错误处理
4. 添加本地缓存

**实现**:
```python
# data_pipeline/adapters/akshare_adapter.py
import akshare as ak
import time
from functools import lru_cache
from typing import Optional
import pandas as pd

class RateLimiter:
    """API调用频率限制器"""
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    def wait_if_needed(self):
        now = time.time()
        # 移除1分钟前的记录
        self.calls = [t for t in self.calls if now - t < 60]
        
        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.calls[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.calls.append(now)

class RealAKShareAdapter:
    """真实的AKShare数据适配器"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter(calls_per_minute=60)
        self.cache = {}
    
    def _call_with_retry(self, func, *args, max_retries=3, **kwargs):
        """带重试的API调用"""
        for attempt in range(max_retries):
            try:
                self.rate_limiter.wait_if_needed()
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # 指数退避
    
    def get_realtime_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取实时行情"""
        cache_key = f"realtime_{symbol}_{int(time.time() // 60)}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # 转换股票代码格式
            code = symbol.split('.')[0]
            df = self._call_with_retry(ak.stock_zh_a_spot_em)
            data = df[df['代码'] == code]
            
            self.cache[cache_key] = data
            return data
        except Exception as e:
            print(f"获取实时数据失败 {symbol}: {e}")
            return None
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """获取历史数据"""
        cache_key = f"hist_{symbol}_{start_date}_{end_date}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            code = symbol.split('.')[0]
            df = self._call_with_retry(
                ak.stock_zh_a_hist,
                symbol=code,
                period="daily",
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust="qfq"
            )
            
            self.cache[cache_key] = df
            return df
        except Exception as e:
            print(f"获取历史数据失败 {symbol}: {e}")
            return None
```

**测试**:
```python
# tests/test_akshare_real.py
import pytest
from data_pipeline.adapters.akshare_adapter import RealAKShareAdapter

def test_akshare_realtime():
    adapter = RealAKShareAdapter()
    data = adapter.get_realtime_data('000001.SZ')
    assert data is not None
    assert len(data) > 0

def test_akshare_historical():
    adapter = RealAKShareAdapter()
    data = adapter.get_historical_data('000001.SZ', '2024-01-01', '2024-06-30')
    assert data is not None
    assert len(data) > 0
```

---

#### 3.3 数据质量检查 (1天)
**任务**:
1. 实现数据完整性检查
2. 检测异常值
3. 处理缺失数据
4. 数据清洗流程

**实现**:
```python
# data_pipeline/quality_check.py
from typing import List, Dict
import pandas as pd
import numpy as np

class DataQualityChecker:
    """数据质量检查器"""
    
    def check_completeness(self, df: pd.DataFrame) -> Dict:
        """检查数据完整性"""
        total = len(df)
        missing = df.isnull().sum()
        
        return {
            'total_rows': total,
            'missing_values': missing.to_dict(),
            'completeness': 1 - (missing.sum() / (total * len(df.columns)))
        }
    
    def detect_outliers(self, df: pd.DataFrame, column: str) -> List[int]:
        """检测异常值（使用IQR方法）"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers.index.tolist()
    
    def check_time_series_gaps(self, df: pd.DataFrame, date_column: str) -> List[str]:
        """检查时间序列断点"""
        dates = pd.to_datetime(df[date_column])
        date_range = pd.date_range(dates.min(), dates.max(), freq='D')
        
        # 排除周末
        business_days = date_range[date_range.dayofweek < 5]
        missing_dates = set(business_days) - set(dates)
        
        return [d.strftime('%Y-%m-%d') for d in missing_dates]
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        df_clean = df.copy()
        
        # 1. 删除全空行
        df_clean = df_clean.dropna(how='all')
        
        # 2. 填充缺失值（前向填充）
        df_clean = df_clean.fillna(method='ffill')
        
        # 3. 处理异常值（限制在3个标准差内）
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            df_clean[col] = df_clean[col].clip(mean - 3*std, mean + 3*std)
        
        return df_clean
    
    def generate_quality_report(self, df: pd.DataFrame) -> Dict:
        """生成数据质量报告"""
        return {
            'completeness': self.check_completeness(df),
            'shape': df.shape,
            'dtypes': df.dtypes.to_dict(),
            'summary': df.describe().to_dict()
        }
```

---

## 阶段4: 监控部署 (2天) 🟠

### 目标
部署完整的监控栈，实现可视化和告警。

### 任务清单

#### 4.1 Prometheus配置 (0.5天)
**配置文件**:
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'qilin-stack'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'

rule_files:
  - 'alerts.yml'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']
```

**启动**:
```bash
# Docker方式
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

---

#### 4.2 Grafana面板 (1天)
**面板配置JSON** (`grafana/qilin-stack-dashboard.json`):
```json
{
  "dashboard": {
    "title": "Qilin Stack 监控面板",
    "panels": [
      {
        "title": "决策数量",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(decision_made_total[5m])",
            "legendFormat": "{{symbol}}"
          }
        ]
      },
      {
        "title": "信号置信度",
        "type": "gauge",
        "targets": [
          {
            "expr": "avg(signal_confidence)",
            "legendFormat": "平均置信度"
          }
        ]
      },
      {
        "title": "系统权重分布",
        "type": "pie",
        "targets": [
          {
            "expr": "system_weight",
            "legendFormat": "{{system}}"
          }
        ]
      },
      {
        "title": "决策延迟 (P95)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(decision_latency_seconds_bucket[5m]))",
            "legendFormat": "P95延迟"
          }
        ]
      }
    ]
  }
}
```

**导入步骤**:
1. 访问 http://localhost:3000
2. 添加Prometheus数据源
3. 导入dashboard JSON
4. 配置刷新间隔

---

#### 4.3 告警规则 (0.5天)
**配置**:
```yaml
# alerts.yml
groups:
  - name: qilin_stack_alerts
    interval: 30s
    rules:
      # 高错误率告警
      - alert: HighErrorRate
        expr: rate(error_count_total[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "错误率过高"
          description: "过去5分钟错误率: {{ $value }}"
      
      # 低置信度告警
      - alert: LowConfidence
        expr: avg(signal_confidence) < 0.4
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "信号置信度过低"
          description: "平均置信度: {{ $value }}"
      
      # 决策延迟告警
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(decision_latency_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "决策延迟过高"
          description: "P95延迟: {{ $value }}秒"
      
      # 服务宕机告警
      - alert: ServiceDown
        expr: up{job="qilin-stack"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "服务宕机"
          description: "Qilin Stack服务无响应"
```

**AlertManager配置**:
```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default'

receivers:
  - name: 'default'
    webhook_configs:
      - url: 'http://localhost:5001/alert'  # 自定义webhook
    email_configs:
      - to: 'alert@example.com'
        from: 'prometheus@example.com'
        smarthost: 'smtp.example.com:587'
        auth_username: 'user'
        auth_password: 'pass'
```

---

## 阶段5-7 (详细计划见后续任务...)

由于篇幅限制，阶段5-7的详细实施步骤将在开始执行时逐步展开。

---

## 📅 时间表

### 第1-2周: 测试与文档 (P0)
- Day 1-3: 单元测试
- Day 4-5: 集成测试 + CI
- Day 6-8: API文档
- Day 9-10: 用户指南 + 部署文档

### 第3周: 数据与监控 (P1)
- Day 11-12: Qlib数据接入
- Day 13-14: AKShare集成
- Day 15: 数据质量检查
- Day 16-17: 监控部署

### 第4-5周: 优化与回测 (P2)
- Day 18-19: 并发优化
- Day 20: 缓存策略
- Day 21-22: 数据库持久化
- Day 23-26: 回测系统
- Day 27-29: 实盘模拟

### 第6周: 生产部署 (P3)
- Day 30-32: 容器化和编排
- Day 33-35: 部署验证和优化

---

## 🎯 成功标准

### 阶段1完成标准
- [ ] 测试覆盖率 ≥ 80%
- [ ] CI全部通过
- [ ] 代码质量评分 ≥ 8.0

### 阶段2完成标准
- [ ] API文档完整可浏览
- [ ] 用户指南清晰易懂
- [ ] 部署文档可操作

### 阶段3完成标准
- [ ] 真实数据加载成功
- [ ] 数据质量检查通过
- [ ] 性能符合预期

### 阶段4完成标准
- [ ] Prometheus正常采集指标
- [ ] Grafana面板展示正确
- [ ] 告警规则触发正常

### 最终交付标准
- [ ] 所有测试通过
- [ ] 文档完整
- [ ] 监控正常
- [ ] 性能达标
- [ ] 可生产部署

---

## 📞 联系与支持

如有问题，请参考:
- 📖 [API文档](docs/build/html/index.html)
- 💬 [Issue Tracker](https://github.com/your-repo/issues)
- 📧 support@example.com

---

**准备开始第一个任务 → 阶段1.1: 单元测试**

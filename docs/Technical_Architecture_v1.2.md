# 麒麟量化系统技术架构文档 v1.2

## 1. 系统架构概览

### 1.1 整体架构图
```
┌──────────────────────────────────────────────────────────┐
│                     客户端层                              │
│  Web Dashboard | API Client | Trading Terminal           │
└────────────────────┬─────────────────────────────────────┘
                     │ HTTPS/WSS
┌────────────────────▼─────────────────────────────────────┐
│                     API网关层                             │
│  Authentication | Rate Limiting | Load Balancing         │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│                   应用服务层                              │
│  ┌──────────────────────────────────────────────────┐   │
│  │            Multi-Agent决策系统                    │   │
│  │  ┌────────────────────────────────────────────┐  │   │
│  │  │ 10个专业Agent协同工作                      │  │   │
│  │  │ • 涨停质量Agent  • 龙头识别Agent          │  │   │
│  │  │ • 龙虎榜Agent    • 新闻情绪Agent          │  │   │
│  │  │ • 筹码分析Agent  • 缠论Agent              │  │   │
│  │  │ • 波浪理论Agent  • 斐波那契Agent          │  │   │
│  │  │ • 市场情绪Agent  • 风控守门Agent          │  │   │
│  │  └────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│                   Qlib量化引擎层                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │Data Handler │ │Factor Engine│ │Model Training│       │
│  │数据处理     │ │因子计算     │ │模型训练     │       │
│  └─────────────┘ └─────────────┘ └─────────────┘       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │Prediction   │ │Backtest     │ │Portfolio    │       │
│  │在线预测     │ │回测系统     │ │组合管理     │       │
│  └─────────────┘ └─────────────┘ └─────────────┘       │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│                   数据接入层                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │AkShare      │ │TuShare      │ │Exchange API │       │
│  │Adapter      │ │Adapter      │ │Adapter      │       │
│  └─────────────┘ └─────────────┘ └─────────────┘       │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│                   存储层                                  │
│  PostgreSQL | Redis | ClickHouse | MinIO                 │
└───────────────────────────────────────────────────────────┘
```

### 1.2 技术栈选型

| 层级 | 技术选型 | 选择理由 |
|------|----------|----------|
| **前端** | React + TypeScript | 组件化开发，类型安全 |
| **API层** | FastAPI | 高性能，自动文档生成 |
| **业务层** | Python 3.9+ | AI/ML生态完善 |
| **Agent框架** | 自研 + AutoGen | 灵活定制，易扩展 |
| **量化引擎** | Qlib | 微软开源，功能完善 |
| **数据源** | AkShare/TuShare | 数据全面，接口稳定 |
| **关系数据库** | PostgreSQL | ACID，复杂查询支持 |
| **缓存** | Redis | 高性能，支持多种数据结构 |
| **时序数据库** | ClickHouse | 高效的时序数据处理 |
| **对象存储** | MinIO | 兼容S3，私有部署 |
| **消息队列** | RabbitMQ | 可靠性高，功能丰富 |
| **容器** | Docker + K8s | 标准化部署，弹性伸缩 |
| **监控** | Prometheus + Grafana | 开源，生态完善 |

## 2. 核心模块设计

### 2.1 Multi-Agent系统

#### 2.1.1 Agent基类设计
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
from dataclasses import dataclass

@dataclass
class AgentContext:
    """Agent执行上下文"""
    stock_code: str
    date: str
    historical_data: pd.DataFrame
    market_data: Dict[str, Any]
    config: Dict[str, Any]

class BaseAgent(ABC):
    """Agent基类"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = self._setup_logger()
        
    @abstractmethod
    async def analyze(self, context: AgentContext) -> Dict[str, Any]:
        """执行分析
        
        Returns:
            {
                'score': float,  # 0-1评分
                'confidence': float,  # 置信度
                'factors': Dict,  # 关键因素
                'suggestion': str  # 建议
            }
        """
        pass
    
    @abstractmethod
    def get_weight(self) -> float:
        """获取Agent权重"""
        pass
    
    async def validate_input(self, context: AgentContext) -> bool:
        """验证输入数据"""
        return True
    
    def _setup_logger(self):
        """设置日志"""
        import logging
        return logging.getLogger(f"Agent.{self.name}")
```

#### 2.1.2 Agent协调器
```python
class AgentCoordinator:
    """Agent协调器"""
    
    def __init__(self):
        self.agents: List[BaseAgent] = []
        self.fusion_strategy = WeightedFusion()
        
    def register_agent(self, agent: BaseAgent):
        """注册Agent"""
        self.agents.append(agent)
        
    async def run_analysis(self, context: AgentContext) -> Dict[str, Any]:
        """运行所有Agent分析"""
        results = {}
        
        # 并行执行所有Agent
        tasks = []
        for agent in self.agents:
            if await agent.validate_input(context):
                tasks.append(agent.analyze(context))
        
        agent_results = await asyncio.gather(*tasks)
        
        # 融合结果
        final_score = self.fusion_strategy.fuse(agent_results)
        
        # 生成归因分析
        attribution = self.generate_attribution(agent_results)
        
        return {
            'final_score': final_score,
            'agent_results': agent_results,
            'attribution': attribution,
            'timestamp': datetime.now()
        }
```

### 2.2 Qlib集成层

#### 2.2.1 数据管理器
```python
import qlib
from qlib.data import D
from qlib.data.dataset import DatasetH

class QlibDataManager:
    """Qlib数据管理器"""
    
    def __init__(self, provider_uri: str, region: str = "cn"):
        qlib.init(provider_uri=provider_uri, region=region)
        self.data_api = D
        
    def get_stock_data(self, 
                      stock_code: str,
                      start_date: str,
                      end_date: str,
                      fields: List[str] = None) -> pd.DataFrame:
        """获取股票数据"""
        if fields is None:
            fields = ["$open", "$high", "$low", "$close", "$volume"]
            
        return self.data_api.features(
            instruments=[stock_code],
            fields=fields,
            start_time=start_date,
            end_time=end_date
        )
    
    def calculate_factors(self, 
                         stock_data: pd.DataFrame,
                         factor_expressions: List[str]) -> pd.DataFrame:
        """计算技术因子"""
        factors = {}
        for expr in factor_expressions:
            factors[expr] = self.data_api.features(
                instruments=stock_data.index,
                fields=[expr]
            )
        return pd.DataFrame(factors)
```

#### 2.2.2 模型管理器
```python
from qlib.contrib.model.gbdt import LGBModel
from qlib.model.trainer import task_train

class QlibModelManager:
    """Qlib模型管理器"""
    
    def __init__(self):
        self.models = {}
        self.model_registry = ModelRegistry()
        
    def train_model(self, 
                   dataset: DatasetH,
                   model_type: str = "lightgbm") -> str:
        """训练模型"""
        if model_type == "lightgbm":
            model = LGBModel(
                loss="mse",
                colsample_bytree=0.8,
                learning_rate=0.01,
                n_estimators=1000
            )
        
        # 训练
        model.fit(dataset)
        
        # 注册模型
        model_id = self.model_registry.register(model)
        self.models[model_id] = model
        
        return model_id
    
    def predict(self, model_id: str, dataset: DatasetH) -> pd.DataFrame:
        """模型预测"""
        model = self.models.get(model_id)
        if not model:
            model = self.model_registry.load(model_id)
            
        return model.predict(dataset)
```

### 2.3 数据适配层

#### 2.3.1 统一数据接口
```python
from abc import ABC, abstractmethod

class DataAdapter(ABC):
    """数据适配器基类"""
    
    @abstractmethod
    def get_daily_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取日线数据"""
        pass
    
    @abstractmethod
    def get_tick_data(self, stock_code: str, date: str) -> pd.DataFrame:
        """获取tick数据"""
        pass
    
    @abstractmethod
    def get_limit_up_list(self, date: str) -> pd.DataFrame:
        """获取涨停板数据"""
        pass
    
    @abstractmethod
    def get_dragon_tiger_list(self, date: str) -> pd.DataFrame:
        """获取龙虎榜数据"""
        pass
```

#### 2.3.2 AkShare适配器
```python
import akshare as ak

class AkShareAdapter(DataAdapter):
    """AkShare数据适配器"""
    
    def get_daily_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取日线数据"""
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", "")
        )
        return self._normalize_columns(df)
    
    def get_limit_up_list(self, date: str) -> pd.DataFrame:
        """获取涨停板数据"""
        df = ak.stock_zt_pool_em(date=date.replace("-", ""))
        return self._normalize_limit_up_data(df)
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        column_mapping = {
            '日期': 'date',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume'
        }
        return df.rename(columns=column_mapping)
```

### 2.4 执行引擎

#### 2.4.1 下单网关
```python
import hmac
import hashlib
from typing import Dict, List

class OrderGateway:
    """下单网关"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.secret_key = config.get('secret_key')
        self.broker_api = self._init_broker_api()
        
    def place_order(self, order: Dict[str, Any]) -> str:
        """下单"""
        # 签名验证
        signature = self._sign_order(order)
        order['signature'] = signature
        
        # 风控检查
        if not self._risk_check(order):
            raise RiskException("Risk check failed")
        
        # 发送订单
        if self.config.get('mode') == 'production':
            return self.broker_api.send_order(order)
        else:
            return self._simulate_order(order)
    
    def _sign_order(self, order: Dict[str, Any]) -> str:
        """订单签名"""
        message = json.dumps(order, sort_keys=True)
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _risk_check(self, order: Dict[str, Any]) -> bool:
        """风控检查"""
        # 检查仓位限制
        if order['quantity'] > self.config.get('max_quantity', 10000):
            return False
        
        # 检查金额限制
        if order['amount'] > self.config.get('max_amount', 1000000):
            return False
        
        return True
```

## 3. 数据流设计

### 3.1 实时数据流
```
Market Data Source
        │
        ▼
   Data Adapter
        │
        ▼
   Message Queue (RabbitMQ)
        │
        ├──► Real-time Processing
        │         │
        │         ▼
        │    Agent Analysis
        │         │
        │         ▼
        │    Decision Making
        │
        └──► Data Storage (ClickHouse)
```

### 3.2 批处理数据流
```
Historical Data
        │
        ▼
   Qlib Data Engine
        │
        ▼
   Factor Calculation
        │
        ▼
   Model Training
        │
        ▼
   Model Registry
        │
        ▼
   Online Prediction
```

## 4. 安全架构

### 4.1 安全层级
```
┌─────────────────────────────────┐
│      应用安全                    │
│  • 输入验证                      │
│  • SQL注入防护                   │
│  • XSS防护                       │
├─────────────────────────────────┤
│      认证授权                    │
│  • JWT Token                     │
│  • RBAC权限模型                  │
│  • API密钥管理                   │
├─────────────────────────────────┤
│      数据安全                    │
│  • 传输加密 (TLS)                │
│  • 存储加密                      │
│  • 敏感数据脱敏                  │
├─────────────────────────────────┤
│      下单安全                    │
│  • HMAC签名                      │
│  • 时间戳验证                    │
│  • 重放攻击防护                  │
└─────────────────────────────────┘
```

### 4.2 安全措施实现
```python
class SecurityManager:
    """安全管理器"""
    
    def __init__(self):
        self.encryptor = AESEncryptor()
        self.validator = InputValidator()
        
    def encrypt_sensitive_data(self, data: str) -> str:
        """加密敏感数据"""
        return self.encryptor.encrypt(data)
    
    def validate_request(self, request: Dict) -> bool:
        """验证请求"""
        # 验证签名
        if not self._verify_signature(request):
            return False
        
        # 验证时间戳
        if not self._verify_timestamp(request):
            return False
        
        # 验证输入
        if not self.validator.validate(request):
            return False
        
        return True
    
    def _verify_signature(self, request: Dict) -> bool:
        """验证签名"""
        signature = request.pop('signature', None)
        if not signature:
            return False
            
        expected_sig = self._calculate_signature(request)
        return hmac.compare_digest(signature, expected_sig)
```

## 5. 性能优化策略

### 5.1 缓存策略
```python
class CacheManager:
    """缓存管理器"""
    
    def __init__(self):
        self.redis_client = redis.Redis()
        self.local_cache = TTLCache(maxsize=1000, ttl=300)
        
    def get_or_compute(self, key: str, compute_func, ttl: int = 300):
        """获取或计算"""
        # L1缓存 - 本地内存
        if key in self.local_cache:
            return self.local_cache[key]
        
        # L2缓存 - Redis
        value = self.redis_client.get(key)
        if value:
            self.local_cache[key] = value
            return value
        
        # 计算并缓存
        value = compute_func()
        self.redis_client.setex(key, ttl, value)
        self.local_cache[key] = value
        
        return value
```

### 5.2 异步处理
```python
class AsyncProcessor:
    """异步处理器"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.loop = asyncio.get_event_loop()
        
    async def process_batch(self, items: List, process_func):
        """批量异步处理"""
        tasks = []
        for item in items:
            task = self.loop.run_in_executor(
                self.executor,
                process_func,
                item
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
```

## 6. 监控与告警

### 6.1 监控指标
```python
from prometheus_client import Counter, Histogram, Gauge

# 业务指标
trade_signals_total = Counter('trade_signals_total', 'Total trade signals generated')
prediction_accuracy = Gauge('prediction_accuracy', 'Model prediction accuracy')
agent_performance = Histogram('agent_performance_seconds', 'Agent execution time', ['agent_name'])

# 系统指标
api_requests_total = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
api_request_duration = Histogram('api_request_duration_seconds', 'API request duration')
system_memory_usage = Gauge('system_memory_usage_bytes', 'System memory usage')
```

### 6.2 告警规则
```yaml
# alerting_rules.yml
groups:
  - name: qilin_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(api_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          
      - alert: SlowAgentExecution
        expr: histogram_quantile(0.95, agent_performance_seconds) > 10
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Agent execution is slow"
          
      - alert: LowPredictionAccuracy
        expr: prediction_accuracy < 0.5
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Model prediction accuracy is low"
```

## 7. 部署架构

### 7.1 容器化部署
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# 启动应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 7.2 Kubernetes部署
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qilin-stack
spec:
  replicas: 3
  selector:
    matchLabels:
      app: qilin-stack
  template:
    metadata:
      labels:
        app: qilin-stack
    spec:
      containers:
      - name: qilin-stack
        image: qilin-stack:v1.2
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

## 8. 扩展性设计

### 8.1 插件系统
```python
class PluginManager:
    """插件管理器"""
    
    def __init__(self):
        self.plugins = {}
        
    def register_plugin(self, name: str, plugin_class):
        """注册插件"""
        self.plugins[name] = plugin_class
        
    def load_plugin(self, name: str, config: Dict):
        """加载插件"""
        plugin_class = self.plugins.get(name)
        if not plugin_class:
            raise ValueError(f"Plugin {name} not found")
        return plugin_class(config)
    
    def discover_plugins(self, plugin_dir: str):
        """自动发现插件"""
        for file in os.listdir(plugin_dir):
            if file.endswith('.py'):
                module = importlib.import_module(f"plugins.{file[:-3]}")
                if hasattr(module, 'Plugin'):
                    self.register_plugin(file[:-3], module.Plugin)
```

### 8.2 策略扩展
```python
class StrategyFactory:
    """策略工厂"""
    
    strategies = {}
    
    @classmethod
    def register_strategy(cls, name: str):
        """注册策略装饰器"""
        def decorator(strategy_class):
            cls.strategies[name] = strategy_class
            return strategy_class
        return decorator
    
    @classmethod
    def create_strategy(cls, name: str, config: Dict):
        """创建策略实例"""
        strategy_class = cls.strategies.get(name)
        if not strategy_class:
            raise ValueError(f"Strategy {name} not found")
        return strategy_class(config)

# 使用示例
@StrategyFactory.register_strategy("limit_up")
class LimitUpStrategy(BaseStrategy):
    pass
```

## 9. 故障恢复机制

### 9.1 断点续传
```python
class CheckpointManager:
    """检查点管理器"""
    
    def save_checkpoint(self, state: Dict):
        """保存检查点"""
        checkpoint = {
            'state': state,
            'timestamp': datetime.now().isoformat(),
            'version': '1.2'
        }
        with open('checkpoint.json', 'w') as f:
            json.dump(checkpoint, f)
    
    def load_checkpoint(self) -> Optional[Dict]:
        """加载检查点"""
        if not os.path.exists('checkpoint.json'):
            return None
        with open('checkpoint.json', 'r') as f:
            return json.load(f)
    
    def recover_from_checkpoint(self):
        """从检查点恢复"""
        checkpoint = self.load_checkpoint()
        if checkpoint:
            self.restore_state(checkpoint['state'])
            return True
        return False
```

### 9.2 熔断机制
```python
class CircuitBreaker:
    """熔断器"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        
    def call(self, func, *args, **kwargs):
        """调用函数"""
        if self.state == 'open':
            if self._should_attempt_reset():
                self.state = 'half-open'
            else:
                raise CircuitBreakerOpenException()
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """成功回调"""
        self.failure_count = 0
        self.state = 'closed'
    
    def _on_failure(self):
        """失败回调"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
```

## 10. 性能基准

### 10.1 性能指标
| 指标 | 目标值 | 测试方法 |
|------|--------|----------|
| Agent分析延迟 | < 1s | 单Agent独立测试 |
| 全流程延迟 | < 45s | 端到端测试 |
| 并发用户数 | > 100 | 压力测试 |
| 日处理量 | > 10000 | 批量测试 |
| 内存占用 | < 8GB | 资源监控 |
| CPU使用率 | < 70% | 资源监控 |

### 10.2 优化建议
1. 使用连接池管理数据库连接
2. 实现多级缓存策略
3. 异步处理非关键路径
4. 使用消息队列解耦模块
5. 定期清理历史数据
6. 使用索引优化查询
7. 实现负载均衡
8. 使用CDN加速静态资源

---
*文档版本：1.2*
*更新时间：2024-10-15*
*作者：技术架构团队*
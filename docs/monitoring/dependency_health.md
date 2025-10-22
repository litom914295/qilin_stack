# 依赖健康探针（P0-12）

## 概述
建立外部依赖健康监控系统，主动探测数据库、缓存、消息队列、外部API等关键依赖的健康状态。

## 监控依赖

### 关键依赖（Critical）
1. **PostgreSQL**: 主数据库
2. **Redis**: 缓存和会话存储
3. **Kafka**: 消息队列

### 非关键依赖（Non-Critical）
1. **MLflow**: 模型服务
2. **TuShare API**: 外部行情数据
3. **AKShare API**: 备用数据源
4. **S3/MinIO**: 对象存储

## 健康状态

### 状态定义
- **HEALTHY (1.0)**: 依赖正常工作
- **DEGRADED (0.5)**: 依赖部分功能受限但可用
- **UNHEALTHY (0.0)**: 依赖不可用
- **UNKNOWN**: 未知状态（探针失败）

### 探测规则
- **数据库**: 执行 `SELECT 1` 查询
- **Redis**: 执行 `PING` 命令
- **HTTP端点**: 请求健康检查端点
- **超时**: 5-10秒（可配置）
- **探测间隔**: 30秒

## Prometheus指标

### Gauge指标
```promql
# 依赖健康状态 (0/0.5/1)
qilin_dependency_health_status{dependency="postgres", type="database"}

# 依赖可用性
qilin_dependency_availability{dependency="redis", type="cache"}
```

### Histogram指标
```promql
# 探针响应时间分布
qilin_dependency_probe_duration_seconds_bucket{dependency="mlflow", type="ml_service"}
```

### Counter指标
```promql
# 探针失败总数
qilin_dependency_probe_failures_total{dependency="postgres", type="database"}
```

## API端点

### 基础健康检查
```bash
GET /health
# Response: {"status": "healthy"}
```
用于K8s liveness probe。

### 就绪检查
```bash
GET /ready
# Response: {"status": "ready"}
```
用于K8s readiness probe，检查关键依赖是否可用。

### 所有依赖健康
```bash
GET /health/dependencies
```
返回所有依赖的健康状态摘要。

**示例响应**:
```json
{
  "overall_status": "healthy",
  "timestamp": "2025-10-16T10:00:00Z",
  "dependencies": {
    "postgres": {
      "status": "healthy",
      "type": "database",
      "response_time_ms": 12.5,
      "critical": true,
      "error": null
    },
    "redis": {
      "status": "healthy",
      "type": "cache",
      "response_time_ms": 3.2,
      "critical": true,
      "error": null
    }
  }
}
```

### 单个依赖健康
```bash
GET /health/dependencies/{dependency_name}
```

### Prometheus Metrics
```bash
GET /metrics
```

## 配置

### dependencies.yaml
```yaml
dependencies:
  - name: postgres
    type: database
    endpoint: postgresql://user:pass@localhost:5432/db
    timeout_seconds: 5.0
    critical: true
  
  - name: redis
    type: cache
    endpoint: redis://localhost:6379/0
    timeout_seconds: 3.0
    critical: true

probe_config:
  interval_seconds: 30
  retry_count: 3
  retry_delay_seconds: 5
```

## Kubernetes集成

### Liveness Probe
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 30
```

### Readiness Probe
```yaml
readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 10
```

当关键依赖不可用时，pod标记为not ready，流量不会路由到该pod。

## 告警规则

### Critical告警
- **关键依赖不可用**: 持续2分钟 → page
- **数据库探针持续失败**: 5分钟内失败>3次 → page

### Warning告警
- **数据库响应慢**: P95延迟>3秒 → warning
- **外部API不可用**: 持续10分钟 → warning
- **整体健康降级**: >2个依赖不健康 → warning

## 故障排查

### 依赖探针失败
1. 检查网络连接：`ping`, `telnet`
2. 验证端点配置：URL、端口、认证
3. 查看依赖服务日志
4. 检查防火墙/网络策略

### 响应时间慢
1. 检查依赖负载
2. 优化查询/请求
3. 增加超时时间
4. 考虑扩容

### 关键依赖不可用
1. 立即触发降级策略
2. 切换到备用依赖（如有）
3. 通知相关团队
4. 启动应急预案

## 最佳实践

1. **区分关键性**: 明确标记关键/非关键依赖
2. **合理超时**: 根据依赖类型设置超时
3. **探测频率**: 平衡及时性和资源消耗
4. **降级策略**: 为非关键依赖准备降级方案
5. **告警分级**: 关键依赖立即page，非关键warning
6. **定期演练**: 定期模拟依赖故障

## 部署

### 1. 构建镜像
```bash
docker build -t qilin-stack/dependency-probe:latest -f Dockerfile.probe .
```

### 2. 部署到K8s
```bash
kubectl apply -f k8s/deployments/dependency-health-probe.yaml
```

### 3. 配置Prometheus抓取
通过ServiceMonitor自动配置。

### 4. 导入告警规则
```bash
kubectl apply -f monitoring/prometheus/rules/dependency_health_alerts.yaml
```

## 监控示例

### 查询依赖健康
```promql
# 所有依赖健康状态
qilin_dependency_health_status

# 不健康的依赖
qilin_dependency_health_status < 1

# 关键依赖健康
qilin_dependency_health_status{critical="true"}
```

### 查询响应时间
```promql
# P95响应时间
histogram_quantile(0.95,
  rate(qilin_dependency_probe_duration_seconds_bucket[5m])
)

# 按依赖分组
histogram_quantile(0.95,
  sum(rate(qilin_dependency_probe_duration_seconds_bucket[5m])) by (dependency, le)
)
```

### 失败率
```promql
# 探针失败率
rate(qilin_dependency_probe_failures_total[5m])
```

## 参考资料
- [Health Checks Best Practices](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)
- [Prometheus Monitoring](https://prometheus.io/docs/practices/instrumentation/)
- [依赖管理策略](https://wiki.qilin.internal/architecture/dependencies)

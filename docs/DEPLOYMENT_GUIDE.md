# 麒麟量化系统部署指南

## 🚀 快速开始

### 前置要求
- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.24+ (生产环境)
- Python 3.10+ (本地开发)

---

## 📦 本地开发环境

### 1. 使用Docker Compose

```bash
# 克隆项目
cd qilin_stack_with_ta

# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f app

# 停止服务
docker-compose down
```

**访问地址**:
- 应用: http://localhost:8000
- 健康检查: http://localhost:8000/health/ready
- Prometheus指标: http://localhost:9090/metrics

### 2. 本地Python开发

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 运行应用
python main.py --mode simulation --log-level INFO
```

---

## 🏭 生产环境部署

### 1. Kubernetes部署

#### 准备工作

```bash
# 创建命名空间
kubectl create namespace production

# 创建Secret (替换为实际值)
kubectl create secret generic qilin-secrets \
  --from-literal=jwt-secret=your-secret-key \
  --from-literal=db-password=your-db-password \
  --namespace=production
```

#### 部署应用

```bash
# 应用所有配置
kubectl apply -f deploy/k8s/ --namespace=production

# 查看部署状态
kubectl get pods -n production
kubectl get svc -n production

# 查看日志
kubectl logs -f deployment/qilin-trading-system -n production

# 查看HPA状态
kubectl get hpa -n production
```

#### 验证部署

```bash
# 获取服务地址
export SERVICE_IP=$(kubectl get svc qilin-service -n production -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# 健康检查
curl http://$SERVICE_IP/health/ready

# 查看metrics
curl http://$SERVICE_IP:9090/metrics
```

### 2. 滚动更新

```bash
# 更新镜像
kubectl set image deployment/qilin-trading-system \
  qilin-app=qilin/trading-system:v2.2 \
  --namespace=production

# 查看更新状态
kubectl rollout status deployment/qilin-trading-system -n production

# 回滚(如果需要)
kubectl rollout undo deployment/qilin-trading-system -n production
```

### 3. 扩缩容

```bash
# 手动扩容
kubectl scale deployment qilin-trading-system --replicas=5 -n production

# 自动扩容已通过HPA配置
# 查看HPA状态
kubectl describe hpa qilin-hpa -n production
```

---

## 📊 监控配置

### 1. Prometheus配置

创建 `prometheus-config.yaml`:

```yaml
scrape_configs:
  - job_name: 'qilin-trading'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - production
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        target_label: __address__
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
```

### 2. Grafana Dashboard

导入Dashboard JSON:
- 系统概览: `deploy/monitoring/grafana-dashboard.json`
- Agent性能: `deploy/monitoring/agent-dashboard.json`

常用指标:
- `qilin_requests_total` - 请求总数
- `qilin_request_duration_seconds` - 请求延迟
- `qilin_agent_analysis_duration_seconds` - Agent分析时长
- `qilin_stocks_analyzed_total` - 分析股票总数
- `qilin_system_health` - 系统健康状态

---

## 🔐 安全配置

### 1. 密钥管理

```bash
# 生成JWT密钥
python -c "import secrets; print(secrets.token_urlsafe(32))"

# 更新Secret
kubectl create secret generic qilin-secrets \
  --from-literal=jwt-secret=<新密钥> \
  --dry-run=client -o yaml | kubectl apply -f - -n production
```

### 2. 网络策略

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: qilin-network-policy
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: qilin-trading
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector: {}
      ports:
        - protocol: TCP
          port: 8000
        - protocol: TCP
          port: 9090
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: redis
      ports:
        - protocol: TCP
          port: 6379
```

---

## 🧪 测试部署

### 1. 冒烟测试

```bash
# 健康检查
curl -f http://$SERVICE_IP/health/ready || exit 1

# 基础功能测试
curl -X POST http://$SERVICE_IP/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol": "000001", "date": "2025-01-15"}'
```

### 2. 负载测试

```bash
# 使用Apache Bench
ab -n 1000 -c 10 http://$SERVICE_IP/health/ready

# 使用Locust
pip install locust
locust -f tests/performance/locustfile.py --host=http://$SERVICE_IP
```

---

## 🔧 故障排查

### 常见问题

**1. Pod无法启动**
```bash
# 查看Pod事件
kubectl describe pod <pod-name> -n production

# 查看日志
kubectl logs <pod-name> -n production
```

**2. 健康检查失败**
```bash
# 进入容器
kubectl exec -it <pod-name> -n production -- /bin/bash

# 手动检查
curl localhost:8000/health/ready
```

**3. 性能问题**
```bash
# 查看资源使用
kubectl top pods -n production
kubectl top nodes

# 查看HPA
kubectl get hpa -n production
```

### 日志收集

```bash
# 收集所有Pod日志
kubectl logs -l app=qilin-trading -n production --tail=1000 > qilin-logs.txt

# 实时查看日志
kubectl logs -f deployment/qilin-trading-system -n production
```

---

## 📝 配置管理

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `ENV` | 运行环境 | `production` |
| `LOG_LEVEL` | 日志级别 | `INFO` |
| `REDIS_URL` | Redis连接 | `redis://localhost:6379` |
| `MAX_POSITIONS` | 最大持仓 | `5` |

### 配置更新

```bash
# 更新ConfigMap
kubectl edit configmap qilin-config -n production

# 重启Pod使配置生效
kubectl rollout restart deployment/qilin-trading-system -n production
```

---

## 🔄 CI/CD集成

### GitHub Actions

工作流已在 `.github/workflows/ci-cd.yml` 配置

**触发条件**:
- Push到 `main` 或 `develop` 分支
- 创建Pull Request

**部署流程**:
1. 代码质量检查
2. 单元测试
3. 安全扫描
4. Docker镜像构建
5. 自动部署到对应环境

### 手动部署

```bash
# 构建镜像
docker build -t qilin/trading-system:v2.1 .

# 推送到registry
docker push qilin/trading-system:v2.1

# 更新Kubernetes
kubectl set image deployment/qilin-trading-system \
  qilin-app=qilin/trading-system:v2.1 \
  -n production
```

---

## 📚 相关文档

- [改进进度报告](IMPROVEMENT_PROGRESS.md)
- [技术架构文档](Technical_Architecture_v2.1_Final.md)
- [API文档](API_DOCUMENTATION.md) (待创建)
- [运维手册](OPERATIONS_MANUAL.md) (待创建)

---

## 🆘 支持

遇到问题？
1. 查看 [FAQ](FAQ.md)
2. 查看日志: `kubectl logs -f deployment/qilin-trading-system -n production`
3. 检查健康状态: `curl http://$SERVICE_IP/health/ready`
4. 查看监控指标: Grafana Dashboard

---

**最后更新**: 2025-10-16
**文档版本**: v1.0

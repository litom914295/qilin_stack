# 麒麟量化交易系统 - 监控体系

完整的监控、追踪、日志和告警解决方案。

## 目录结构

```
monitoring/
├── prometheus/              # Prometheus监控配置
│   ├── prometheus.yml       # 主配置文件
│   └── alerts/              # 告警规则
│       └── system_alerts.yml
├── grafana/                 # Grafana可视化配置
│   ├── dashboards/          # Dashboard配置
│   │   ├── system-overview.json
│   │   └── trading-system.json
│   └── provisioning/        # 数据源配置
│       └── datasources/
│           └── prometheus.yml
├── jaeger/                  # Jaeger追踪配置
│   └── jaeger_tracer.py     # Python追踪器
├── logstash/                # Logstash日志处理
│   ├── pipeline/
│   │   └── qilin-logs.conf
│   └── config/
├── filebeat/                # Filebeat日志收集
│   └── filebeat.yml
├── alertmanager/            # Alertmanager告警管理
│   ├── alertmanager.yml     # 告警配置
│   └── webhook_receiver.py  # Webhook接收服务
└── docker-compose.*.yml     # Docker部署配置
```

## 快速开始

### 1. 部署Prometheus监控系统

```bash
cd monitoring
docker-compose -f docker-compose.prometheus.yml up -d
```

访问: http://localhost:9090

### 2. 部署Grafana可视化平台

```bash
docker-compose -f docker-compose.grafana.yml up -d
```

访问: http://localhost:3000
- 默认账号: admin
- 默认密码: qilin_admin_2024

### 3. 部署Jaeger追踪系统

```bash
docker-compose -f docker-compose.jaeger.yml up -d
```

访问: http://localhost:16686

### 4. 部署ELK日志系统

```bash
docker-compose -f docker-compose.elk.yml up -d
```

- Elasticsearch: http://localhost:9200
- Kibana: http://localhost:5601

### 5. 启动告警Webhook接收服务

```bash
cd alertmanager
pip install flask requests
python webhook_receiver.py
```

访问: http://localhost:5001

## 监控指标

### 系统指标
- CPU使用率
- 内存使用率
- 磁盘使用率
- 网络流量
- 节点健康状态

### 应用指标
- HTTP请求QPS
- HTTP错误率
- 请求延迟（P95, P99）
- 服务健康检查

### 数据库指标
- Redis连接数、内存使用
- PostgreSQL连接数、慢查询
- MongoDB连接数

### 交易系统指标
- 订单量、失败率
- T日候选数、T+1买入数、T+2卖出数
- 模型预测延迟
- 特征提取延迟
- 数据管道处理速率
- 累计盈亏、胜率、平均收益率

## Grafana Dashboard

### 系统概览 (system-overview.json)
- 8个核心面板
- 实时系统资源监控
- 服务健康状态
- HTTP性能指标

### 交易系统 (trading-system.json)
- 13个核心面板
- 订单量和失败率
- T日/T+1/T+2流程监控
- 模型性能监控
- 盈亏统计

## 告警规则

### 系统告警
- CPU/内存/磁盘使用率告警
- 节点宕机告警
- 磁盘IO/网络流量告警

### 应用告警
- 服务不可用告警
- HTTP错误率告警
- 请求延迟告警

### 数据库告警
- 连接数过高告警
- 慢查询告警

### 交易系统告警
- 订单失败率过高告警
- 模型预测延迟告警
- 数据管道停滞告警

## 告警通知渠道

### 邮件通知
配置SMTP服务器发送邮件告警。

### 企业微信
支持企业微信群机器人推送。

### 钉钉
支持钉钉群机器人推送。

### 短信
支持严重告警短信通知（需第三方短信服务）。

## 日志查询

### Kibana查询示例

1. 查询错误日志:
```
log_level:"ERROR" AND service:"auction-engine"
```

2. 查询交易日志:
```
component:"trading" AND message:*T+1*
```

3. 查询性能日志:
```
tags:"performance" AND duration:>1000
```

## Jaeger追踪

### Python应用集成

```python
from monitoring.jaeger.jaeger_tracer import init_tracer, get_tracer

# 初始化追踪器
tracer = init_tracer(service_name="qilin-auction-engine")

# 使用装饰器追踪函数
@tracer.trace(operation_name="predict", tags_dict={"model": "lgb"})
def predict(data):
    return model.predict(data)

# 手动创建span
with tracer.tracer.start_active_span('manual_op') as scope:
    span = scope.span
    span.set_tag('key', 'value')
    # 执行操作
```

## 维护

### 数据保留策略
- Prometheus: 30天
- Elasticsearch: 7天（可调整）
- Jaeger: 按Elasticsearch配置

### 备份
```bash
# 备份Grafana配置
docker exec qilin-grafana grafana-cli admin export /var/lib/grafana/backups/

# 备份Prometheus数据
docker cp qilin-prometheus:/prometheus ./prometheus-backup/
```

### 性能优化
1. 调整Prometheus抓取间隔
2. 优化Elasticsearch索引策略
3. 配置Jaeger采样率
4. 设置日志级别

## 故障排查

### Prometheus无数据
1. 检查target状态: http://localhost:9090/targets
2. 检查服务是否暴露metrics端点
3. 检查网络连接

### Grafana无法连接Prometheus
1. 检查数据源配置
2. 验证Prometheus URL可访问
3. 查看Grafana日志

### 告警不发送
1. 检查Alertmanager状态
2. 验证告警规则配置
3. 检查Webhook接收服务
4. 查看Alertmanager日志

### 日志未收集
1. 检查Filebeat状态
2. 验证日志路径配置
3. 检查Logstash pipeline
4. 查看Elasticsearch健康状态

## 环境变量

```bash
# Jaeger
export JAEGER_AGENT_HOST=localhost
export JAEGER_AGENT_PORT=6831

# Elasticsearch
export ES_JAVA_OPTS="-Xms1g -Xmx1g"

# Logstash
export LS_JAVA_OPTS="-Xms512m -Xmx512m"
```

## 依赖

### Python依赖
```bash
pip install jaeger-client opentracing flask requests
```

### Docker镜像
- prometheus/prometheus:latest
- grafana/grafana:latest
- jaegertracing/all-in-one:latest
- docker.elastic.co/elasticsearch/elasticsearch:8.10.0
- docker.elastic.co/logstash/logstash:8.10.0
- docker.elastic.co/kibana/kibana:8.10.0
- docker.elastic.co/beats/filebeat:8.10.0

## 参考文档

- [Prometheus文档](https://prometheus.io/docs/)
- [Grafana文档](https://grafana.com/docs/)
- [Jaeger文档](https://www.jaegertracing.io/docs/)
- [ELK Stack文档](https://www.elastic.co/guide/)
- [Alertmanager文档](https://prometheus.io/docs/alerting/latest/alertmanager/)

## 支持

如有问题，请联系运维团队或提交Issue。

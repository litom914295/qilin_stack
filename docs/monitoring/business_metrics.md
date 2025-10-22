# 业务金指标监控（P0-11）

## 概述
建立完整的业务指标监控体系，追踪核心量化业务指标：推荐命中率、收益率、信号质量。

## 核心指标

### 1. 推荐命中率（Hit Rate）
- **定义**: T+1验证方向正确的推荐占比
- **目标**: ≥ 60%（1日）、≥ 65%（7日）、≥ 70%（30日）
- **计算**: 正确推荐数 / 已验证推荐数
- **验证规则**:
  - BUY推荐：实际收益 > 0
  - SELL推荐：实际收益 < 0
  - HOLD推荐：实际收益在±2%内

### 2. 平均收益率（Average Return）
- **定义**: T+1实际收益率均值
- **目标**: 1日 > 0%、7日 > 1%、30日 > 2%
- **监控**: 按日/周/月聚合

### 3. 信号质量（Signal Quality）
- **High**: 置信度 ≥ 0.8
- **Medium**: 置信度 0.6-0.8
- **Low**: 置信度 < 0.6
- **目标**: High质量占比 ≥ 30%

### 4. 信号覆盖率（Signal Coverage）
- **定义**: 有推荐信号的股票数 / 股票池总数
- **目标**: ≥ 5%（日均至少覆盖150只股票，假设3000只股票池）

### 5. 推荐数量（Recommendation Volume）
- **目标**: 10-100条/日
- **告警**: < 5条（过低）或 > 500条（异常）

## Prometheus指标

### Counter指标
```promql
# 推荐总数（按动作和质量）
qilin_recommendations_total{action="buy|sell|hold", quality="high|medium|low"}

# 用户关注推荐总数
qilin_recommendation_follows_total{action="buy|sell|hold"}
```

### Gauge指标
```promql
# 推荐命中率
qilin_recommendation_hit_rate{timeframe="1d|7d|30d"}

# 平均收益率（百分比）
qilin_avg_return_percent{timeframe="1d|7d|30d"}

# 信号覆盖率
qilin_signal_coverage

# 当日推荐数量
qilin_daily_recommendations{action="buy|sell|hold"}

# 活跃用户数
qilin_active_users{period="daily|weekly|monthly"}
```

### Histogram指标
```promql
# 推荐收益分布
qilin_recommendation_return_percent_bucket

# 推荐置信度分布
qilin_recommendation_confidence_bucket
```

## API端点

### Prometheus Metrics
```bash
GET /metrics
```
供Prometheus抓取的标准metrics端点。

### 记录推荐
```bash
POST /api/recommendations
{
  "recommendation_id": "rec_20250116_001",
  "stock_code": "000001",
  "action": "buy",
  "confidence": 0.85,
  "target_price": 15.0
}
```

### T+1验证推荐
```bash
POST /api/recommendations/{recommendation_id}/validate
{
  "actual_return": 0.05
}
```

### 查询命中率
```bash
GET /api/metrics/hit-rate?days=7
# Response: {"hit_rate": 0.72, "timeframe": "7d"}
```

### 查询平均收益
```bash
GET /api/metrics/avg-return?days=7
# Response: {"avg_return": 0.035, "avg_return_percent": 3.5, "timeframe": "7d"}
```

### 当日汇总
```bash
GET /api/metrics/summary
```

### 业务报告
```bash
GET /api/metrics/report?days=30
```

## 集成示例

### 在推荐生成时记录
```python
from monitoring.business_metrics import BusinessMetricsCollector, RecommendationAction

collector = BusinessMetricsCollector()

# 生成推荐后记录
recommendation = agent.generate_recommendation(stock_code)
collector.record_recommendation(
    recommendation_id=f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    stock_code=stock_code,
    action=RecommendationAction(recommendation['action']),
    confidence=recommendation['confidence'],
    target_price=recommendation.get('target_price')
)
```

### T+1验证流程
```python
# 定时任务（每日收盘后）
async def validate_yesterday_recommendations():
    yesterday = datetime.now().date() - timedelta(days=1)
    
    # 获取昨日推荐
    recommendations = get_recommendations_by_date(yesterday)
    
    for rec in recommendations:
        # 获取实际收益
        actual_return = calculate_actual_return(
            stock_code=rec.stock_code,
            date=yesterday
        )
        
        # 验证
        collector.validate_recommendation(
            recommendation_id=rec.recommendation_id,
            actual_return=actual_return
        )
```

## 告警规则

### Critical告警
- **命中率 < 40%**: 1日命中率持续30分钟低于40%
- **平均收益 < -5%**: 7日平均收益持续2小时低于-5%

### Warning告警
- **命中率 < 60%**: 1日命中率持续1小时低于60%
- **推荐数量 < 5**: 持续3小时每日推荐数少于5条
- **信号覆盖 < 5%**: 持续2小时覆盖率低于5%
- **高质量占比 < 20%**: 持续6小时高质量推荐占比低于20%

## Grafana仪表盘

仪表盘包含以下面板：
1. **核心指标卡片**: 命中率、收益率、推荐数、覆盖率
2. **趋势图**: 命中率和收益率的1d/7d/30d趋势
3. **推荐分布**: 按动作、质量分类的柱状图
4. **收益分布热力图**: 收益率分布的heatmap
5. **质量对比表**: 不同质量推荐的统计对比

## 部署

### 1. 构建Docker镜像
```bash
docker build -t qilin-stack/business-metrics:latest -f Dockerfile.metrics .
```

### 2. 部署到K8s
```bash
kubectl apply -f k8s/deployments/business-metrics.yaml
```

### 3. 配置Prometheus抓取
Prometheus会通过ServiceMonitor自动发现并抓取metrics。

### 4. 导入Grafana仪表盘
```bash
# 导入 grafana/dashboards/business_metrics.json
```

## 最佳实践

1. **T+1验证**: 每日收盘后自动运行验证任务
2. **数据持久化**: 使用PostgreSQL/Redis持久化推荐记录
3. **指标导出频率**: Prometheus每30秒抓取一次
4. **告警路由**: Critical告警发送到PagerDuty，Warning发送到Slack
5. **报告生成**: 每周自动生成业务指标周报

## 故障排查

### 命中率异常低
1. 检查特征工程是否正常
2. 验证Agent模型版本
3. 检查市场环境变化
4. 查看信号质量分布

### 推荐数量异常
1. 检查数据采集是否正常
2. 验证过滤逻辑
3. 检查Agent置信度阈值

### 收益率为负
1. 分析按动作的收益分布
2. 检查风控规则
3. 回测近期策略表现
4. 考虑市场系统性风险

## 参考资料
- [Prometheus最佳实践](https://prometheus.io/docs/practices/)
- [Grafana仪表盘设计](https://grafana.com/docs/grafana/latest/dashboards/)
- [量化交易指标体系](https://wiki.qilin.internal/quant/metrics)

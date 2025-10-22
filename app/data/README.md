# 数据质量监控系统

完整的数据质量监控解决方案，提供多维度数据质量检查和实时监控告警。

## 功能特性

### 质量检查维度

1. **完整性检查** (Completeness)
   - 必需字段检查
   - 空值比例检查
   - 数据量检查

2. **一致性检查** (Consistency)
   - High/Low关系验证
   - OHLC逻辑顺序
   - 时间戳唯一性

3. **时效性检查** (Timeliness)
   - 数据延迟监控
   - 时间序列连续性
   - 更新频率检查

4. **准确性检查** (Accuracy)
   - 数值范围验证
   - 异常值检测
   - 统计分布分析

5. **有效性检查** (Validity)
   - 负值检测
   - 无穷值检测
   - 数据类型验证

## 快速开始

### 基础使用

```python
from app.data import DataQualityChecker, QualityMonitor
import pandas as pd

# 创建检查器
checker = DataQualityChecker()

# 检查数据
report = checker.check_all(data, data_source="your_source")

# 查看质量分数
print(f"质量分数: {report.quality_score}")

# 查看问题
for issue in report.issues:
    print(f"[{issue.severity.value}] {issue.message}")
```

### 实时监控

```python
# 创建监控器
monitor = QualityMonitor(alert_threshold=80.0)

# 监控数据
report, needs_alert = monitor.monitor(data, "source_name")

if needs_alert:
    print("⚠️ 数据质量告警！")
    print(f"质量分数: {report.quality_score}")

# 获取质量趋势
trend = monitor.get_quality_trend()
print(trend)

# 生成摘要
summary = monitor.generate_summary()
print(summary)
```

## 配置选项

### 默认配置

```python
config = {
    'completeness': {
        'required_fields': ['open', 'high', 'low', 'close', 'volume'],
        'min_completeness_ratio': 0.95
    },
    'timeliness': {
        'max_delay_minutes': 5,
        'check_update_time': True
    },
    'accuracy': {
        'price_range': {'min': 0, 'max': 10000},
        'volume_range': {'min': 0, 'max': 1e12},
        'check_outliers': True,
        'outlier_std_threshold': 3
    },
    'consistency': {
        'check_high_low': True,
        'check_ohlc_order': True,
        'check_duplicate_timestamps': True
    }
}

# 使用自定义配置
checker = DataQualityChecker(config=config)
```

## API参考

### DataQualityChecker

```python
class DataQualityChecker:
    def __init__(self, config: Optional[Dict] = None)
    
    def check_all(
        self,
        data: pd.DataFrame,
        data_source: str = "unknown",
        symbol: Optional[str] = None
    ) -> QualityReport
```

### QualityMonitor

```python
class QualityMonitor:
    def __init__(self, alert_threshold: float = 80.0)
    
    def monitor(
        self,
        data: pd.DataFrame,
        data_source: str,
        symbol: Optional[str] = None
    ) -> Tuple[QualityReport, bool]
    
    def get_quality_trend(
        self,
        data_source: Optional[str] = None,
        window: int = 10
    ) -> pd.DataFrame
    
    def generate_summary(self) -> Dict[str, Any]
```

### QualityReport

```python
@dataclass
class QualityReport:
    data_source: str              # 数据源名称
    check_time: datetime           # 检查时间
    total_records: int            # 总记录数
    passed_checks: int            # 通过检查数
    failed_checks: int            # 失败检查数
    quality_score: float          # 质量分数 (0-100)
    issues: List[QualityIssue]    # 问题列表
    metrics: Dict[str, float]     # 质量指标
```

## 质量分数计算

质量分数采用扣分制，从100分开始：

```
质量分数 = 100 - Σ(问题严重程度 × 权重 × 2)

严重程度权重:
- INFO: 0.1
- WARNING: 0.3
- ERROR: 0.6
- CRITICAL: 1.0
```

## 告警机制

当质量分数低于阈值时触发告警：

```python
monitor = QualityMonitor(alert_threshold=80.0)

report, needs_alert = monitor.monitor(data, "source")

if needs_alert:
    # 自动记录WARNING级别日志
    # 包含关键问题摘要
    pass
```

## 使用示例

### 示例1: 单次检查

```python
import pandas as pd
from app.data import DataQualityChecker

# 准备数据
data = pd.DataFrame({
    'open': [100, 101, 102],
    'high': [105, 106, 107],
    'low': [95, 96, 97],
    'close': [101, 102, 103],
    'volume': [1000000, 1100000, 1200000]
}, index=pd.date_range('2023-01-01', periods=3))

# 执行检查
checker = DataQualityChecker()
report = checker.check_all(data, "example_source")

# 打印结果
print(f"质量分数: {report.quality_score:.2f}")
print(f"问题数量: {len(report.issues)}")

# 打印指标
for metric, value in report.metrics.items():
    print(f"{metric}: {value:.2%}")
```

### 示例2: 持续监控

```python
from app.data import QualityMonitor

# 创建监控器
monitor = QualityMonitor(alert_threshold=85.0)

# 模拟持续监控
for day in range(30):
    # 获取当天数据
    daily_data = fetch_daily_data(day)
    
    # 监控
    report, alert = monitor.monitor(daily_data, f"day_{day}")
    
    if alert:
        print(f"Day {day}: 质量告警! Score={report.quality_score:.1f}")

# 查看趋势
trend = monitor.get_quality_trend(window=30)
print(trend[['quality_score', 'issue_count']])

# 生成摘要
summary = monitor.generate_summary()
print(f"平均质量分数: {summary['recent_avg_score']:.1f}")
print(f"严重性分布: {summary['severity_distribution']}")
```

### 示例3: 多数据源监控

```python
monitor = QualityMonitor()

sources = ['tushare', 'akshare', 'wind']

for source in sources:
    data = fetch_data(source)
    report, _ = monitor.monitor(data, source)
    print(f"{source}: {report.quality_score:.1f}")

# 按数据源查看趋势
for source in sources:
    trend = monitor.get_quality_trend(data_source=source)
    print(f"\n{source} 趋势:")
    print(trend)
```

## 集成到系统

### 与数据获取集成

```python
from app.data import QualityMonitor

class DataFetcher:
    def __init__(self):
        self.quality_monitor = QualityMonitor(alert_threshold=80.0)
    
    def fetch_data(self, source: str, symbol: str):
        # 获取数据
        data = self._fetch_raw_data(source, symbol)
        
        # 质量检查
        report, needs_alert = self.quality_monitor.monitor(
            data, 
            data_source=f"{source}_{symbol}"
        )
        
        if needs_alert:
            # 降级处理
            data = self._apply_fallback(data, report)
        
        return data, report
```

### 与监控系统集成

```python
from app.monitoring import metrics
from app.data import QualityMonitor

monitor = QualityMonitor()

def check_data_quality(data, source):
    report, _ = monitor.monitor(data, source)
    
    # 上报到Prometheus
    metrics.data_quality_score.labels(source=source).set(
        report.quality_score
    )
    
    metrics.data_quality_issues.labels(
        source=source,
        severity='error'
    ).set(
        sum(1 for i in report.issues if i.severity == SeverityLevel.ERROR)
    )
    
    return report
```

## 最佳实践

1. **设置合理的阈值**: 根据业务需求调整告警阈值
2. **定期监控**: 对关键数据源实施持续监控
3. **趋势分析**: 定期查看质量趋势，识别长期问题
4. **自动化处理**: 将质量检查集成到数据管道
5. **降级策略**: 准备数据质量问题的备选方案

## 故障排查

### 常见问题

1. **误报过多**
   - 调整配置阈值
   - 检查数据源特性
   - 自定义检查逻辑

2. **性能问题**
   - 减少异常值检查范围
   - 采样大数据集
   - 异步执行质量检查

3. **历史记录过多**
   - 定期清理history
   - 设置最大记录数
   - 使用数据库存储

## 相关文档

- [监控系统](../monitoring/README.md)
- [异常处理](../core/exceptions.py)
- [测试用例](../../tests/unit/test_data_quality.py)

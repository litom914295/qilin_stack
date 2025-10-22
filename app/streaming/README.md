# 实时数据流处理系统

基于Kafka的实时数据流处理系统，提供流消费、处理管道、窗口聚合和异常检测功能。

## 功能特性

### 核心组件

1. **KafkaConsumer** - Kafka消费者
2. **StreamPipeline** - 流处理管道
3. **WindowAggregator** - 时间窗口聚合
4. **AnomalyDetector** - 实时异常检测
5. **StreamManager** - 流管理器

## 快速开始

### 启动Kafka

```bash
docker-compose up -d zookeeper kafka
```

### 基础使用

```python
import asyncio
from app.streaming import StreamManager, StreamPipeline

async def main():
    # 创建管道
    pipeline = StreamPipeline()
    
    # 创建管理器
    manager = StreamManager(
        kafka_config={
            'bootstrap_servers': 'localhost:9092',
            'group_id': 'qilin_group',
            'topics': ['market_data']
        },
        pipeline=pipeline
    )
    
    # 启动
    await manager.start()
    await manager.run()

asyncio.run(main())
```

## 组件说明

### StreamPipeline

流处理管道支持：
- 过滤器 (Filters)
- 转换器 (Transformers)
- 处理器 (Processors)
- 窗口聚合
- 异常检测

```python
pipeline = StreamPipeline()

# 添加过滤器
pipeline.add_filter(lambda msg: msg.value.get('price', 0) > 0)

# 添加转换器
pipeline.add_transformer(lambda msg: msg)

# 添加处理器
pipeline.add_processor(RealTimeDataValidator())
```

### WindowAggregator

时间窗口聚合：

```python
aggregator = WindowAggregator(
    window_size=60,  # 60秒窗口
    slide_size=10    # 10秒滑动
)

# 添加消息
aggregator.add("symbol_001", message)

# 聚合
result = aggregator.aggregate("symbol_001", lambda msgs: len(msgs))

# 统计
stats = aggregator.get_window_stats("symbol_001")
```

### AnomalyDetector

实时异常检测：

```python
detector = AnomalyDetector(
    std_threshold=3.0,
    window_size=100
)

# 检测异常
is_anomaly = detector.is_anomaly("price", 150.5)

# 添加正常值
detector.add_value("price", 100.0)

# 获取统计
stats = detector.get_stats("price")
```

## 配置

### Docker Compose

```yaml
zookeeper:
  image: confluentinc/cp-zookeeper:latest
  ports:
    - "2181:2181"

kafka:
  image: confluentinc/cp-kafka:latest
  depends_on:
    - zookeeper
  ports:
    - "9092:9092"
```

## 相关文档

- [数据质量](../data/README.md)
- [监控系统](../monitoring/README.md)

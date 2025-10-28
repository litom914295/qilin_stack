"""
实时数据流处理系统
支持Kafka消费、流处理管道、窗口聚合和异常检测
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class StreamMessage:
    """流消息"""
    topic: str
    key: Optional[str]
    value: Dict[str, Any]
    timestamp: datetime
    partition: int = 0
    offset: int = 0
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class ProcessedMessage:
    """处理后的消息"""
    original: StreamMessage
    processed_data: Dict[str, Any]
    processing_time: float
    status: str = "success"
    error: Optional[str] = None


class StreamProcessor(ABC):
    """流处理器抽象基类"""
    
    @abstractmethod
    async def process(self, message: StreamMessage) -> ProcessedMessage:
        """处理单条消息"""
        pass
    
    @abstractmethod
    async def process_batch(self, messages: List[StreamMessage]) -> List[ProcessedMessage]:
        """批量处理消息"""
        pass


class KafkaConsumer:
    """Kafka消费者 (模拟实现)"""
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "qilin_consumer",
        topics: Optional[List[str]] = None,
        auto_offset_reset: str = "latest",
        enable_auto_commit: bool = True
    ):
        """
        初始化Kafka消费者
        
        Args:
            bootstrap_servers: Kafka服务器地址
            group_id: 消费者组ID
            topics: 订阅的主题列表
            auto_offset_reset: 偏移重置策略
            enable_auto_commit: 是否自动提交偏移
        """
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.topics = topics or []
        self.auto_offset_reset = auto_offset_reset
        self.enable_auto_commit = enable_auto_commit
        
        self.running = False
        self.consumer = None
        
        logger.info(f"Kafka consumer initialized: {bootstrap_servers}, group: {group_id}")
    
    async def start(self):
        """启动消费者"""
        try:
            # 在实际实现中，这里会创建真实的Kafka消费者
            # from aiokafka import AIOKafkaConsumer
            # self.consumer = AIOKafkaConsumer(
            #     *self.topics,
            #     bootstrap_servers=self.bootstrap_servers,
            #     group_id=self.group_id,
            #     auto_offset_reset=self.auto_offset_reset,
            #     enable_auto_commit=self.enable_auto_commit
            # )
            # await self.consumer.start()
            
            self.running = True
            logger.info("Kafka consumer started")
            
        except Exception as e:
            logger.error(f"Failed to start Kafka consumer: {e}")
            raise
    
    async def stop(self):
        """停止消费者"""
        self.running = False
        
        if self.consumer:
            # await self.consumer.stop()
            pass
        
        logger.info("Kafka consumer stopped")
    
    async def consume(self) -> Optional[StreamMessage]:
        """
        消费单条消息
        
        Returns:
            StreamMessage或None
        """
        if not self.running:
            return None
        
        try:
            # 模拟从Kafka消费消息
            # 在实际实现中使用:
            # async for msg in self.consumer:
            #     return self._parse_message(msg)
            
            await asyncio.sleep(0.1)  # 模拟IO延迟
            return None
            
        except Exception as e:
            logger.error(f"Error consuming message: {e}")
            return None
    
    def _parse_message(self, kafka_msg) -> StreamMessage:
        """解析Kafka消息"""
        return StreamMessage(
            topic=kafka_msg.topic,
            key=kafka_msg.key.decode() if kafka_msg.key else None,
            value=json.loads(kafka_msg.value.decode()),
            timestamp=datetime.fromtimestamp(kafka_msg.timestamp / 1000),
            partition=kafka_msg.partition,
            offset=kafka_msg.offset,
            headers={k: v.decode() for k, v in kafka_msg.headers}
        )


class WindowAggregator:
    """窗口聚合器"""
    
    def __init__(
        self,
        window_size: int = 60,  # 秒
        slide_size: int = 10     # 秒
    ):
        """
        初始化窗口聚合器
        
        Args:
            window_size: 窗口大小（秒）
            slide_size: 滑动大小（秒）
        """
        self.window_size = timedelta(seconds=window_size)
        self.slide_size = timedelta(seconds=slide_size)
        self.windows: Dict[str, deque] = {}
        
    def add(self, key: str, message: StreamMessage):
        """添加消息到窗口"""
        if key not in self.windows:
            self.windows[key] = deque()
        
        self.windows[key].append(message)
        self._cleanup_old_messages(key)
    
    def _cleanup_old_messages(self, key: str):
        """清理过期消息"""
        if key not in self.windows:
            return
        
        cutoff_time = datetime.now() - self.window_size
        
        while self.windows[key] and self.windows[key][0].timestamp < cutoff_time:
            self.windows[key].popleft()
    
    def aggregate(self, key: str, agg_func: Callable) -> Any:
        """
        聚合窗口数据
        
        Args:
            key: 窗口键
            agg_func: 聚合函数
            
        Returns:
            聚合结果
        """
        if key not in self.windows or not self.windows[key]:
            return None
        
        self._cleanup_old_messages(key)
        messages = list(self.windows[key])
        
        if not messages:
            return None
        
        return agg_func(messages)
    
    def get_window_stats(self, key: str) -> Dict[str, Any]:
        """获取窗口统计信息"""
        if key not in self.windows:
            return {'count': 0, 'window_size': 0}
        
        self._cleanup_old_messages(key)
        messages = list(self.windows[key])
        
        if not messages:
            return {'count': 0, 'window_size': 0}
        
        return {
            'count': len(messages),
            'window_size': self.window_size.total_seconds(),
            'oldest': messages[0].timestamp.isoformat(),
            'newest': messages[-1].timestamp.isoformat(),
            'time_span': (messages[-1].timestamp - messages[0].timestamp).total_seconds()
        }


class AnomalyDetector:
    """异常检测器"""
    
    def __init__(
        self,
        std_threshold: float = 3.0,
        window_size: int = 100
    ):
        """
        初始化异常检测器
        
        Args:
            std_threshold: 标准差阈值
            window_size: 历史窗口大小
        """
        self.std_threshold = std_threshold
        self.window_size = window_size
        self.history: Dict[str, deque] = {}
    
    def add_value(self, key: str, value: float):
        """添加值到历史"""
        if key not in self.history:
            self.history[key] = deque(maxlen=self.window_size)
        
        self.history[key].append(value)
    
    def is_anomaly(self, key: str, value: float) -> bool:
        """
        检测是否为异常值
        
        Args:
            key: 检测键
            value: 待检测值
            
        Returns:
            是否为异常
        """
        if key not in self.history or len(self.history[key]) < 10:
            # 样本不足，不判定为异常
            return False
        
        values = np.array(list(self.history[key]))
        mean = values.mean()
        std = values.std()
        
        if std == 0:
            return False
        
        z_score = abs((value - mean) / std)
        return z_score > self.std_threshold
    
    def get_stats(self, key: str) -> Optional[Dict[str, float]]:
        """获取统计信息"""
        if key not in self.history or len(self.history[key]) == 0:
            return None
        
        values = np.array(list(self.history[key]))
        
        return {
            'mean': float(values.mean()),
            'std': float(values.std()),
            'min': float(values.min()),
            'max': float(values.max()),
            'count': len(values)
        }


class StreamPipeline:
    """流处理管道"""
    
    def __init__(self):
        """初始化管道"""
        self.processors: List[StreamProcessor] = []
        self.filters: List[Callable] = []
        self.transformers: List[Callable] = []
        self.aggregator = WindowAggregator()
        self.anomaly_detector = AnomalyDetector()
        
        self.stats = {
            'messages_processed': 0,
            'messages_filtered': 0,
            'anomalies_detected': 0,
            'errors': 0
        }
    
    def add_processor(self, processor: StreamProcessor):
        """添加处理器"""
        self.processors.append(processor)
        logger.info(f"Added processor: {processor.__class__.__name__}")
    
    def add_filter(self, filter_func: Callable[[StreamMessage], bool]):
        """添加过滤器"""
        self.filters.append(filter_func)
    
    def add_transformer(self, transform_func: Callable[[StreamMessage], StreamMessage]):
        """添加转换器"""
        self.transformers.append(transform_func)
    
    async def process_message(self, message: StreamMessage) -> Optional[ProcessedMessage]:
        """
        处理单条消息
        
        Args:
            message: 输入消息
            
        Returns:
            处理结果或None
        """
        start_time = datetime.now()
        
        try:
            # 1. 过滤
            for filter_func in self.filters:
                if not filter_func(message):
                    self.stats['messages_filtered'] += 1
                    return None
            
            # 2. 转换
            for transform_func in self.transformers:
                message = transform_func(message)
            
            # 3. 异常检测
            if 'price' in message.value:
                price = float(message.value['price'])
                key = f"{message.topic}:price"
                
                if self.anomaly_detector.is_anomaly(key, price):
                    self.stats['anomalies_detected'] += 1
                    logger.warning(f"Anomaly detected: {key} = {price}")
                
                self.anomaly_detector.add_value(key, price)
            
            # 4. 窗口聚合
            if message.key:
                self.aggregator.add(message.key, message)
            
            # 5. 应用处理器
            result = None
            for processor in self.processors:
                result = await processor.process(message)
            
            # 6. 更新统计
            self.stats['messages_processed'] += 1
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessedMessage(
                original=message,
                processed_data=result.processed_data if result else {},
                processing_time=processing_time,
                status="success"
            )
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Error processing message: {e}")
            
            return ProcessedMessage(
                original=message,
                processed_data={},
                processing_time=(datetime.now() - start_time).total_seconds(),
                status="error",
                error=str(e)
            )
            
    def get_stats(self) -> Dict[str, Any]:
        """获取管道统计信息"""
        return {
            **self.stats,
            'processors_count': len(self.processors),
            'filters_count': len(self.filters),
            'transformers_count': len(self.transformers)
        }


class RealTimeDataValidator(StreamProcessor):
    """实时数据验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.validation_rules = {
            'price': lambda x: 0 < x < 10000,
            'volume': lambda x: x >= 0,
            'timestamp': lambda x: x is not None
        }
    
    async def process(self, message: StreamMessage) -> ProcessedMessage:
        """验证单条消息"""
        start_time = datetime.now()
        errors = []
        
        # 验证字段
        for field, rule in self.validation_rules.items():
            if field in message.value:
                try:
                    if not rule(message.value[field]):
                        errors.append(f"Validation failed for {field}")
                except Exception as e:
                    errors.append(f"Validation error for {field}: {e}")
        
        status = "success" if not errors else "validation_failed"
        
        return ProcessedMessage(
            original=message,
            processed_data={
                'validated': status == "success",
                'errors': errors
            },
            processing_time=(datetime.now() - start_time).total_seconds(),
            status=status,
            error="; ".join(errors) if errors else None
        )
    
    async def process_batch(self, messages: List[StreamMessage]) -> List[ProcessedMessage]:
        """批量验证"""
        results = []
        for message in messages:
            result = await self.process(message)
            results.append(result)
        return results


class StreamManager:
    """流管理器"""
    
    def __init__(
        self,
        kafka_config: Optional[Dict] = None,
        pipeline: Optional[StreamPipeline] = None
    ):
        """
        初始化流管理器
        
        Args:
            kafka_config: Kafka配置
            pipeline: 流处理管道
        """
        self.kafka_config = kafka_config or {
            'bootstrap_servers': 'localhost:9092',
            'group_id': 'qilin_consumer',
            'topics': ['market_data', 'trades']
        }
        
        self.pipeline = pipeline or StreamPipeline()
        self.consumer = None
        self.running = False
        self.processed_messages = []
        
    async def start(self):
        """启动流处理"""
        # 创建消费者
        self.consumer = KafkaConsumer(**self.kafka_config)
        await self.consumer.start()
        
        # 添加默认处理器
        self.pipeline.add_processor(RealTimeDataValidator())
        
        self.running = True
        logger.info("Stream manager started")
    
    async def stop(self):
        """停止流处理"""
        self.running = False
        
        if self.consumer:
            await self.consumer.stop()
        
        logger.info("Stream manager stopped")
    
    async def run(self):
        """运行流处理循环"""
        while self.running:
            try:
                # 消费消息
                message = await self.consumer.consume()
                
                if message:
                    # 处理消息
                    result = await self.pipeline.process_message(message)
                    
                    if result:
                        self.processed_messages.append(result)
                        
                        # 限制历史记录
                        if len(self.processed_messages) > 10000:
                            self.processed_messages = self.processed_messages[-5000:]
                
            except Exception as e:
                logger.error(f"Error in stream processing loop: {e}")
                await asyncio.sleep(1)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        pipeline_stats = self.pipeline.get_stats()
        
        return {
            'pipeline': pipeline_stats,
            'processed_messages_count': len(self.processed_messages),
            'running': self.running
        }

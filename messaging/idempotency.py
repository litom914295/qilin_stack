"""
消息幂等性处理系统（P0-14）
防止Kafka消息重复消费，实现at-most-once/exactly-once语义
"""

import logging
import hashlib
import time
from typing import Dict, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import redis.asyncio as aioredis
from enum import Enum

logger = logging.getLogger(__name__)


class MessageStatus(str, Enum):
    """消息处理状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MessageRecord:
    """消息记录"""
    message_id: str
    message_hash: str
    status: MessageStatus
    created_at: datetime
    completed_at: Optional[datetime]
    attempt_count: int
    last_error: Optional[str]


class IdempotencyManager:
    """幂等性管理器"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl_seconds: int = 86400,  # 24小时
        key_prefix: str = "idempotency:"
    ):
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix
        self.redis_client: Optional[aioredis.Redis] = None
    
    async def connect(self):
        """连接Redis"""
        if self.redis_client is None:
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            logger.info("Connected to Redis for idempotency management")
    
    async def disconnect(self):
        """断开连接"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Disconnected from Redis")
    
    def _generate_message_hash(self, message: Dict[str, Any]) -> str:
        """
        生成消息内容哈希
        
        Args:
            message: 消息内容
            
        Returns:
            SHA256哈希值
        """
        # 排序keys确保一致性
        sorted_message = json.dumps(message, sort_keys=True)
        return hashlib.sha256(sorted_message.encode()).hexdigest()
    
    def _get_redis_key(self, message_id: str) -> str:
        """获取Redis key"""
        return f"{self.key_prefix}{message_id}"
    
    async def check_duplicate(
        self,
        message_id: str,
        message: Dict[str, Any]
    ) -> tuple[bool, Optional[MessageRecord]]:
        """
        检查消息是否重复
        
        Args:
            message_id: 消息ID
            message: 消息内容
            
        Returns:
            (是否重复, 消息记录)
        """
        await self.connect()
        
        message_hash = self._generate_message_hash(message)
        redis_key = self._get_redis_key(message_id)
        
        # 检查Redis中是否存在
        existing = await self.redis_client.get(redis_key)
        
        if existing:
            record_data = json.loads(existing)
            record = MessageRecord(
                message_id=record_data['message_id'],
                message_hash=record_data['message_hash'],
                status=MessageStatus(record_data['status']),
                created_at=datetime.fromisoformat(record_data['created_at']),
                completed_at=datetime.fromisoformat(record_data['completed_at']) if record_data.get('completed_at') else None,
                attempt_count=record_data.get('attempt_count', 0),
                last_error=record_data.get('last_error')
            
            # 检查哈希是否匹配
            if record.message_hash == message_hash:
                # 相同消息
                if record.status == MessageStatus.COMPLETED:
                    logger.info(f"Message {message_id} already processed, skipping")
                    return True, record
                elif record.status == MessageStatus.PROCESSING:
                    logger.warning(f"Message {message_id} is being processed")
                    return True, record
            else:
                # ID相同但内容不同，可能是冲突
                logger.warning(f"Message ID conflict: {message_id}")
        
        return False, None
    
    async def mark_processing(
        self,
        message_id: str,
        message: Dict[str, Any]
    ) -> bool:
        """
        标记消息为处理中
        
        Args:
            message_id: 消息ID
            message: 消息内容
            
        Returns:
            是否成功标记
        """
        await self.connect()
        
        message_hash = self._generate_message_hash(message)
        redis_key = self._get_redis_key(message_id)
        
        record = MessageRecord(
            message_id=message_id,
            message_hash=message_hash,
            status=MessageStatus.PROCESSING,
            created_at=datetime.now(),
            completed_at=None,
            attempt_count=1,
            last_error=None
        
        record_data = {
            'message_id': record.message_id,
            'message_hash': record.message_hash,
            'status': record.status.value,
            'created_at': record.created_at.isoformat(),
            'completed_at': None,
            'attempt_count': record.attempt_count,
            'last_error': None
        }
        
        # 使用SETNX确保原子性
        success = await self.redis_client.set(
            redis_key,
            json.dumps(record_data),
            nx=True,  # Only set if not exists
            ex=self.ttl_seconds
        
        if success:
            logger.info(f"Marked message {message_id} as processing")
            return True
        else:
            logger.warning(f"Failed to mark message {message_id} (already exists)")
            return False
    
    async def mark_completed(
        self,
        message_id: str,
        result: Optional[Any] = None
    ):
        """
        标记消息处理完成
        
        Args:
            message_id: 消息ID
            result: 处理结果（可选）
        """
        await self.connect()
        
        redis_key = self._get_redis_key(message_id)
        existing = await self.redis_client.get(redis_key)
        
        if existing:
            record_data = json.loads(existing)
            record_data['status'] = MessageStatus.COMPLETED.value
            record_data['completed_at'] = datetime.now().isoformat()
            
            await self.redis_client.set(
                redis_key,
                json.dumps(record_data),
                ex=self.ttl_seconds
            
            logger.info(f"Marked message {message_id} as completed")
    
    async def mark_failed(
        self,
        message_id: str,
        error: str
    ):
        """
        标记消息处理失败
        
        Args:
            message_id: 消息ID
            error: 错误信息
        """
        await self.connect()
        
        redis_key = self._get_redis_key(message_id)
        existing = await self.redis_client.get(redis_key)
        
        if existing:
            record_data = json.loads(existing)
            record_data['status'] = MessageStatus.FAILED.value
            record_data['last_error'] = error
            record_data['attempt_count'] = record_data.get('attempt_count', 0) + 1
            
            await self.redis_client.set(
                redis_key,
                json.dumps(record_data),
                ex=self.ttl_seconds
            
            logger.warning(f"Marked message {message_id} as failed: {error}")
    
    async def process_with_idempotency(
        self,
        message_id: str,
        message: Dict[str, Any],
        handler: Callable
    ) -> tuple[bool, Optional[Any]]:
        """
        幂等性处理消息
        
        Args:
            message_id: 消息ID
            message: 消息内容
            handler: 处理函数
            
        Returns:
            (是否处理, 处理结果)
        """
        # 1. 检查重复
        is_duplicate, record = await self.check_duplicate(message_id, message)
        
        if is_duplicate and record:
            if record.status == MessageStatus.COMPLETED:
                logger.info(f"Message {message_id} already completed, skipping")
                return False, None
            elif record.status == MessageStatus.PROCESSING:
                logger.warning(f"Message {message_id} is being processed elsewhere")
                return False, None
        
        # 2. 标记为处理中
        marked = await self.mark_processing(message_id, message)
        if not marked:
            logger.warning(f"Failed to acquire lock for message {message_id}")
            return False, None
        
        # 3. 执行处理
        try:
            result = await handler(message)
            await self.mark_completed(message_id, result)
            logger.info(f"Successfully processed message {message_id}")
            return True, result
        except Exception as e:
            await self.mark_failed(message_id, str(e))
            logger.error(f"Failed to process message {message_id}: {e}")
            raise


class KafkaIdempotentConsumer:
    """Kafka幂等消费者"""
    
    def __init__(
        self,
        idempotency_manager: IdempotencyManager,
        message_id_extractor: Optional[Callable] = None
    ):
        self.idempotency_manager = idempotency_manager
        self.message_id_extractor = message_id_extractor or self._default_id_extractor
    
    def _default_id_extractor(self, message: Dict[str, Any]) -> str:
        """默认消息ID提取器"""
        # 尝试常见的ID字段
        for key in ['message_id', 'id', 'event_id', 'uuid']:
            if key in message:
                return str(message[key])
        
        # 如果没有ID，使用内容哈希作为ID
        return hashlib.sha256(
            json.dumps(message, sort_keys=True).encode()
        ).hexdigest()
    
    async def consume_message(
        self,
        message: Dict[str, Any],
        handler: Callable
    ) -> bool:
        """
        消费单条消息（幂等）
        
        Args:
            message: Kafka消息
            handler: 消息处理函数
            
        Returns:
            是否成功处理
        """
        message_id = self.message_id_extractor(message)
        
        try:
            processed, result = await self.idempotency_manager.process_with_idempotency(
                message_id=message_id,
                message=message,
                handler=handler
            
            return processed
            
        except Exception as e:
            logger.error(f"Error consuming message {message_id}: {e}")
            return False


async def example_message_handler(message: Dict[str, Any]) -> Dict[str, Any]:
    """示例消息处理函数"""
    import asyncio
    
    logger.info(f"Processing message: {message}")
    
    # 模拟处理
    await asyncio.sleep(0.1)
    
    # 模拟业务逻辑
    result = {
        'status': 'success',
        'processed_at': datetime.now().isoformat(),
        'data': message.get('data')
    }
    
    return result


async def main():
    """示例用法"""
    logging.basicConfig(level=logging.INFO)
    
    # 创建幂等性管理器
    idempotency_mgr = IdempotencyManager(
        redis_url="redis://localhost:6379/0",
        ttl_seconds=3600  # 1小时
    
    # 创建Kafka消费者
    consumer = KafkaIdempotentConsumer(idempotency_mgr)
    
    # 模拟消息
    test_message = {
        'message_id': 'msg_001',
        'type': 'recommendation',
        'data': {
            'stock_code': '000001',
            'action': 'buy'
        }
    }
    
    print("\n=== First consumption (should process) ===")
    await consumer.consume_message(test_message, example_message_handler)
    
    print("\n=== Second consumption (should skip) ===")
    await consumer.consume_message(test_message, example_message_handler)
    
    print("\n=== Different message (should process) ===")
    test_message2 = {
        'message_id': 'msg_002',
        'type': 'recommendation',
        'data': {
            'stock_code': '000002',
            'action': 'sell'
        }
    }
    await consumer.consume_message(test_message2, example_message_handler)
    
    await idempotency_mgr.disconnect()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

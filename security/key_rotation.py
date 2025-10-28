"""
密钥轮换管理系统
实现JWT密钥、API密钥、加密密钥的自动轮换机制
"""

import os
import base64
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import asyncio
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class KeyType(Enum):
    """密钥类型"""
    JWT_SECRET = "jwt_secret"
    API_KEY = "api_key"
    ENCRYPTION_KEY = "encryption_key"
    SIGNING_KEY = "signing_key"
    DATABASE_PASSWORD = "database_password"


class KeyStatus(Enum):
    """密钥状态"""
    ACTIVE = "active"
    ROTATING = "rotating"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"


@dataclass
class KeyMetadata:
    """密钥元数据"""
    key_id: str
    key_type: KeyType
    status: KeyStatus
    created_at: datetime
    expires_at: datetime
    rotated_at: Optional[datetime] = None
    rotation_count: int = 0
    last_used_at: Optional[datetime] = None
    usage_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class KeyRotationManager:
    """密钥轮换管理器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化密钥轮换管理器
        
        Args:
            config: 配置字典
        """
        self.config = config or self._default_config()
        self.redis_client: Optional[aioredis.Redis] = None
        self.key_store: Dict[str, KeyMetadata] = {}
        self.rotation_schedule: Dict[KeyType, int] = {
            KeyType.JWT_SECRET: 30,  # 30天
            KeyType.API_KEY: 90,  # 90天
            KeyType.ENCRYPTION_KEY: 180,  # 180天
            KeyType.SIGNING_KEY: 60,  # 60天
            KeyType.DATABASE_PASSWORD: 90  # 90天
        }
        
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'redis_url': 'redis://localhost:6379',
            'key_prefix': 'qilin:keys:',
            'grace_period_days': 7,  # 旧密钥保留期
            'alert_before_expiry_days': 3,
            'enable_auto_rotation': True,
            'backup_keys': True
        }
    
    async def initialize(self):
        """初始化连接"""
        self.redis_client = await aioredis.from_url(
            self.config['redis_url'],
            encoding='utf-8',
            decode_responses=True
        )
        logger.info("Key Rotation Manager initialized")
    
    async def generate_key(self, key_type: KeyType, **kwargs) -> Tuple[str, KeyMetadata]:
        """
        生成新密钥
        
        Args:
            key_type: 密钥类型
            **kwargs: 额外参数
            
        Returns:
            (密钥值, 密钥元数据)
        """
        key_id = self._generate_key_id(key_type)
        
        # 根据类型生成密钥
        if key_type == KeyType.JWT_SECRET:
            key_value = self._generate_jwt_secret()
        elif key_type == KeyType.API_KEY:
            key_value = self._generate_api_key()
        elif key_type == KeyType.ENCRYPTION_KEY:
            key_value = self._generate_encryption_key()
        elif key_type == KeyType.SIGNING_KEY:
            key_value = self._generate_signing_key()
        elif key_type == KeyType.DATABASE_PASSWORD:
            key_value = self._generate_database_password()
        else:
            raise ValueError(f"Unsupported key type: {key_type}")
        
        # 创建元数据
        rotation_days = self.rotation_schedule.get(key_type, 90)
        metadata = KeyMetadata(
            key_id=key_id,
            key_type=key_type,
            status=KeyStatus.ACTIVE,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=rotation_days),
            metadata=kwargs
        )
        
        # 存储密钥
        await self._store_key(key_id, key_value, metadata)
        
        logger.info(f"Generated new {key_type.value} key: {key_id}")
        return key_value, metadata
    
    async def rotate_key(self, key_id: str, force: bool = False) -> Tuple[str, KeyMetadata]:
        """
        轮换密钥
        
        Args:
            key_id: 密钥ID
            force: 是否强制轮换
            
        Returns:
            (新密钥值, 新密钥元数据)
        """
        # 获取旧密钥
        old_metadata = await self._get_key_metadata(key_id)
        if not old_metadata:
            raise ValueError(f"Key not found: {key_id}")
        
        # 检查是否需要轮换
        if not force and not self._should_rotate(old_metadata):
            logger.info(f"Key {key_id} does not need rotation yet")
            return await self._get_key_value(key_id), old_metadata
        
        # 标记旧密钥为轮换中
        old_metadata.status = KeyStatus.ROTATING
        await self._update_key_metadata(key_id, old_metadata)
        
        # 生成新密钥
        new_key_value, new_metadata = await self.generate_key(
            old_metadata.key_type,
            previous_key_id=key_id,
            rotation_from=key_id
        )
        
        # 保留旧密钥（宽限期）
        grace_period = self.config['grace_period_days']
        old_metadata.status = KeyStatus.DEPRECATED
        old_metadata.expires_at = datetime.now() + timedelta(days=grace_period)
        await self._update_key_metadata(key_id, old_metadata)
        
        # 记录轮换
        await self._record_rotation(old_metadata, new_metadata)
        
        logger.info(f"Rotated key {key_id} to {new_metadata.key_id}")
        return new_key_value, new_metadata
    
    async def revoke_key(self, key_id: str, reason: str = ""):
        """
        撤销密钥
        
        Args:
            key_id: 密钥ID
            reason: 撤销原因
        """
        metadata = await self._get_key_metadata(key_id)
        if not metadata:
            raise ValueError(f"Key not found: {key_id}")
        
        metadata.status = KeyStatus.REVOKED
        metadata.metadata['revocation_reason'] = reason
        metadata.metadata['revoked_at'] = datetime.now().isoformat()
        
        await self._update_key_metadata(key_id, metadata)
        logger.warning(f"Revoked key {key_id}: {reason}")
    
    async def get_active_key(self, key_type: KeyType) -> Tuple[str, KeyMetadata]:
        """
        获取活跃密钥
        
        Args:
            key_type: 密钥类型
            
        Returns:
            (密钥值, 密钥元数据)
        """
        # 查找所有该类型的活跃密钥
        pattern = f"{self.config['key_prefix']}{key_type.value}:*"
        keys = []
        
        async for key in self.redis_client.scan_iter(match=pattern):
            metadata_json = await self.redis_client.hget(key, 'metadata')
            if metadata_json:
                metadata = self._deserialize_metadata(metadata_json)
                if metadata.status == KeyStatus.ACTIVE:
                    keys.append((key, metadata))
        
        if not keys:
            # 没有活跃密钥，生成新的
            return await self.generate_key(key_type)
        
        # 返回最新的活跃密钥
        keys.sort(key=lambda x: x[1].created_at, reverse=True)
        key_id = keys[0][1].key_id
        key_value = await self._get_key_value(key_id)
        
        # 更新使用统计
        await self._update_usage_stats(key_id)
        
        return key_value, keys[0][1]
    
    async def check_expiring_keys(self) -> List[KeyMetadata]:
        """
        检查即将过期的密钥
        
        Returns:
            即将过期的密钥列表
        """
        expiring_keys = []
        alert_threshold = datetime.now() + timedelta(
            days=self.config['alert_before_expiry_days']
        )
        
        pattern = f"{self.config['key_prefix']}*"
        async for key in self.redis_client.scan_iter(match=pattern):
            metadata_json = await self.redis_client.hget(key, 'metadata')
            if metadata_json:
                metadata = self._deserialize_metadata(metadata_json)
                if (metadata.status == KeyStatus.ACTIVE and 
                    metadata.expires_at <= alert_threshold):
                    expiring_keys.append(metadata)
        
        return expiring_keys
    
    async def auto_rotate_keys(self):
        """自动轮换过期密钥"""
        if not self.config['enable_auto_rotation']:
            logger.info("Auto rotation is disabled")
            return
        
        pattern = f"{self.config['key_prefix']}*"
        rotated_count = 0
        
        async for key in self.redis_client.scan_iter(match=pattern):
            metadata_json = await self.redis_client.hget(key, 'metadata')
            if metadata_json:
                metadata = self._deserialize_metadata(metadata_json)
                
                if self._should_rotate(metadata):
                    try:
                        await self.rotate_key(metadata.key_id)
                        rotated_count += 1
                    except Exception as e:
                        logger.error(f"Failed to rotate key {metadata.key_id}: {e}")
        
        logger.info(f"Auto rotation completed: {rotated_count} keys rotated")
    
    async def cleanup_expired_keys(self):
        """清理过期密钥"""
        now = datetime.now()
        pattern = f"{self.config['key_prefix']}*"
        deleted_count = 0
        
        async for key in self.redis_client.scan_iter(match=pattern):
            metadata_json = await self.redis_client.hget(key, 'metadata')
            if metadata_json:
                metadata = self._deserialize_metadata(metadata_json)
                
                # 删除过期且不活跃的密钥
                if (metadata.status in [KeyStatus.DEPRECATED, KeyStatus.REVOKED] and 
                    metadata.expires_at < now):
                    
                    # 备份后删除
                    if self.config['backup_keys']:
                        await self._backup_key(metadata)
                    
                    await self.redis_client.delete(key)
                    deleted_count += 1
                    logger.info(f"Deleted expired key: {metadata.key_id}")
        
        logger.info(f"Cleanup completed: {deleted_count} keys deleted")
    
    def _generate_key_id(self, key_type: KeyType) -> str:
        """生成密钥ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        random_suffix = secrets.token_hex(4)
        return f"{key_type.value}_{timestamp}_{random_suffix}"
    
    def _generate_jwt_secret(self) -> str:
        """生成JWT密钥"""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
    
    def _generate_api_key(self) -> str:
        """生成API密钥"""
        return f"qilin_{secrets.token_urlsafe(32)}"
    
    def _generate_encryption_key(self) -> str:
        """生成加密密钥"""
        return Fernet.generate_key().decode()
    
    def _generate_signing_key(self) -> str:
        """生成签名密钥"""
        return secrets.token_hex(32)
    
    def _generate_database_password(self) -> str:
        """生成数据库密码"""
        # 生成强密码：大小写字母+数字+特殊字符
        import string
        chars = string.ascii_letters + string.digits + "!@#$%^&*"
        password = ''.join(secrets.choice(chars) for _ in range(32))
        return password
    
    def _should_rotate(self, metadata: KeyMetadata) -> bool:
        """检查是否应该轮换"""
        if metadata.status != KeyStatus.ACTIVE:
            return False
        
        # 检查是否接近过期
        days_until_expiry = (metadata.expires_at - datetime.now()).days
        return days_until_expiry <= 0
    
    async def _store_key(self, key_id: str, key_value: str, metadata: KeyMetadata):
        """存储密钥"""
        redis_key = f"{self.config['key_prefix']}{key_id}"
        
        # 存储密钥值（加密）
        encrypted_value = self._encrypt_key_value(key_value)
        
        # 存储到Redis
        await self.redis_client.hset(redis_key, mapping={
            'value': encrypted_value,
            'metadata': self._serialize_metadata(metadata)
        })
        
        # 设置过期时间
        expire_seconds = int((metadata.expires_at - datetime.now()).total_seconds())
        if expire_seconds > 0:
            await self.redis_client.expire(redis_key, expire_seconds)
    
    async def _get_key_value(self, key_id: str) -> str:
        """获取密钥值"""
        redis_key = f"{self.config['key_prefix']}{key_id}"
        encrypted_value = await self.redis_client.hget(redis_key, 'value')
        
        if not encrypted_value:
            raise ValueError(f"Key not found: {key_id}")
        
        return self._decrypt_key_value(encrypted_value)
    
    async def _get_key_metadata(self, key_id: str) -> Optional[KeyMetadata]:
        """获取密钥元数据"""
        redis_key = f"{self.config['key_prefix']}{key_id}"
        metadata_json = await self.redis_client.hget(redis_key, 'metadata')
        
        if not metadata_json:
            return None
        
        return self._deserialize_metadata(metadata_json)
    
    async def _update_key_metadata(self, key_id: str, metadata: KeyMetadata):
        """更新密钥元数据"""
        redis_key = f"{self.config['key_prefix']}{key_id}"
        await self.redis_client.hset(
            redis_key,
            'metadata',
            self._serialize_metadata(metadata)
        )
    
    async def _update_usage_stats(self, key_id: str):
        """更新使用统计"""
        metadata = await self._get_key_metadata(key_id)
        if metadata:
            metadata.usage_count += 1
            metadata.last_used_at = datetime.now()
            await self._update_key_metadata(key_id, metadata)
    
    async def _record_rotation(self, old_metadata: KeyMetadata, new_metadata: KeyMetadata):
        """记录轮换历史"""
        rotation_record = {
            'old_key_id': old_metadata.key_id,
            'new_key_id': new_metadata.key_id,
            'key_type': old_metadata.key_type.value,
            'rotated_at': datetime.now().isoformat(),
            'old_key_created': old_metadata.created_at.isoformat(),
            'old_key_usage': old_metadata.usage_count
        }
        
        # 存储到轮换历史
        history_key = f"{self.config['key_prefix']}rotation_history"
        await self.redis_client.lpush(
            history_key,
            json.dumps(rotation_record)
        )
        
        # 只保留最近1000条记录
        await self.redis_client.ltrim(history_key, 0, 999)
    
    async def _backup_key(self, metadata: KeyMetadata):
        """备份密钥"""
        backup_key = f"{self.config['key_prefix']}backup:{metadata.key_id}"
        await self.redis_client.hset(
            backup_key,
            'metadata',
            self._serialize_metadata(metadata)
        )
        # 备份保留1年
        await self.redis_client.expire(backup_key, 365 * 24 * 3600)
    
    def _encrypt_key_value(self, value: str) -> str:
        """加密密钥值"""
        # 使用主密钥加密（从环境变量获取）
        master_key = os.getenv('MASTER_ENCRYPTION_KEY', 'default-master-key')
        cipher = Fernet(self._derive_master_key(master_key))
        return cipher.encrypt(value.encode()).decode()
    
    def _decrypt_key_value(self, encrypted_value: str) -> str:
        """解密密钥值"""
        master_key = os.getenv('MASTER_ENCRYPTION_KEY', 'default-master-key')
        cipher = Fernet(self._derive_master_key(master_key))
        return cipher.decrypt(encrypted_value.encode()).decode()
    
    def _derive_master_key(self, password: str) -> bytes:
        """派生主密钥"""
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'qilin_master_salt',
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def _serialize_metadata(self, metadata: KeyMetadata) -> str:
        """序列化元数据"""
        data = {
            'key_id': metadata.key_id,
            'key_type': metadata.key_type.value,
            'status': metadata.status.value,
            'created_at': metadata.created_at.isoformat(),
            'expires_at': metadata.expires_at.isoformat(),
            'rotated_at': metadata.rotated_at.isoformat() if metadata.rotated_at else None,
            'rotation_count': metadata.rotation_count,
            'last_used_at': metadata.last_used_at.isoformat() if metadata.last_used_at else None,
            'usage_count': metadata.usage_count,
            'metadata': metadata.metadata
        }
        return json.dumps(data)
    
    def _deserialize_metadata(self, data: str) -> KeyMetadata:
        """反序列化元数据"""
        obj = json.loads(data)
        return KeyMetadata(
            key_id=obj['key_id'],
            key_type=KeyType(obj['key_type']),
            status=KeyStatus(obj['status']),
            created_at=datetime.fromisoformat(obj['created_at']),
            expires_at=datetime.fromisoformat(obj['expires_at']),
            rotated_at=datetime.fromisoformat(obj['rotated_at']) if obj.get('rotated_at') else None,
            rotation_count=obj['rotation_count'],
            last_used_at=datetime.fromisoformat(obj['last_used_at']) if obj.get('last_used_at') else None,
            usage_count=obj['usage_count'],
            metadata=obj['metadata']
        )

class RotationScheduler:
    """轮换调度器"""
    
    def __init__(self, rotation_manager: KeyRotationManager):
        self.rotation_manager = rotation_manager
        self.running = False
    
    async def start(self):
        """启动调度器"""
        self.running = True
        logger.info("Rotation scheduler started")
        
        # 定期执行轮换检查
        while self.running:
            try:
                # 检查即将过期的密钥
                expiring_keys = await self.rotation_manager.check_expiring_keys()
                if expiring_keys:
                    logger.warning(f"Found {len(expiring_keys)} expiring keys")
                    for key in expiring_keys:
                        logger.warning(
                            f"Key {key.key_id} expires at {key.expires_at}"
                        )
                
                # 自动轮换
                await self.rotation_manager.auto_rotate_keys()
                
                # 清理过期密钥
                await self.rotation_manager.cleanup_expired_keys()
                
            except Exception as e:
                logger.error(f"Rotation scheduler error: {e}")
            
            # 每小时检查一次
            await asyncio.sleep(3600)
    
    def stop(self):
        """停止调度器"""
        self.running = False
        logger.info("Rotation scheduler stopped")


if __name__ == "__main__":
    # 测试代码
    async def main():
        config = {
            'redis_url': 'redis://localhost:6379',
            'enable_auto_rotation': True
        }
        
        manager = KeyRotationManager(config)
        await manager.initialize()
        
        # 生成JWT密钥
        jwt_key, metadata = await manager.generate_key(KeyType.JWT_SECRET)
        print(f"Generated JWT key: {metadata.key_id}")
        print(f"Expires at: {metadata.expires_at}")
        
        # 获取活跃密钥
        active_key, active_metadata = await manager.get_active_key(KeyType.JWT_SECRET)
        print(f"Active key: {active_metadata.key_id}")
        
        # 启动调度器
        scheduler = RotationScheduler(manager)
        # await scheduler.start()  # 实际使用时取消注释
    
    asyncio.run(main())

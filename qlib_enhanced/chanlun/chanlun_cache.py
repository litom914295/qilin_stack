"""
缠论缓存管理器 - Phase 4.3.1

功能:
- 支持Redis缓存 (生产环境)
- 支持本地文件缓存 (开发/回测环境)
- LRU淘汰策略
- 自动过期管理
- 命中率统计

双模式复用:
- Qlib系统: 回测时缓存大量历史分析结果
- 独立系统: 实时交易时缓存最近分析结果

作者: Warp AI Assistant
日期: 2025-01
版本: v1.6
"""

import json
import pickle
import hashlib
import time
from pathlib import Path
from typing import Any, Optional, Dict, List
from dataclasses import dataclass, asdict
from collections import OrderedDict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """缓存统计信息"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ChanLunCache:
    """
    缠论分析结果缓存管理器
    
    支持两种后端:
    1. Redis (推荐生产环境)
    2. 本地文件 (开发/回测环境)
    
    Examples:
        >>> # Redis缓存
        >>> cache = ChanLunCache(backend='redis', redis_url='redis://localhost:6379/0')
        >>> cache.set('stock_000001_20240101', analysis_result, ttl=3600)
        >>> result = cache.get('stock_000001_20240101')
        
        >>> # 文件缓存
        >>> cache = ChanLunCache(backend='file', cache_dir='./cache')
        >>> cache.set('stock_000001_20240101', analysis_result)
    """
    
    def __init__(
        self,
        backend: str = 'file',
        redis_url: Optional[str] = None,
        redis_db: int = 0,
        cache_dir: str = './cache/chanlun',
        max_memory_items: int = 1000,
        default_ttl: int = 3600,
        enable_compression: bool = True
    ):
        """
        初始化缓存管理器
        
        Args:
            backend: 缓存后端 ('redis' | 'file')
            redis_url: Redis连接URL (backend='redis'时必需)
            redis_db: Redis数据库编号
            cache_dir: 文件缓存目录 (backend='file'时使用)
            max_memory_items: 内存LRU缓存最大条目数
            default_ttl: 默认过期时间(秒), 0表示永不过期
            enable_compression: 是否启用压缩 (文件缓存)
        """
        self.backend = backend
        self.default_ttl = default_ttl
        self.enable_compression = enable_compression
        
        # 统计信息
        self.stats = CacheStats()
        
        # 内存LRU缓存 (一级缓存)
        self._memory_cache: OrderedDict = OrderedDict()
        self._max_memory_items = max_memory_items
        
        # 初始化后端
        if backend == 'redis':
            self._init_redis(redis_url, redis_db)
        elif backend == 'file':
            self._init_file(cache_dir)
        else:
            raise ValueError(f"不支持的缓存后端: {backend}")
        
        logger.info(f"缠论缓存初始化完成 - 后端: {backend}, TTL: {default_ttl}s")
    
    def _init_redis(self, redis_url: Optional[str], redis_db: int):
        """初始化Redis后端"""
        try:
            import redis
            if redis_url is None:
                redis_url = f'redis://localhost:6379/{redis_db}'
            self.redis_client = redis.from_url(redis_url, decode_responses=False)
            self.redis_client.ping()
            logger.info(f"Redis连接成功: {redis_url}")
        except ImportError:
            raise ImportError("需要安装redis: pip install redis")
        except Exception as e:
            raise ConnectionError(f"Redis连接失败: {e}")
    
    def _init_file(self, cache_dir: str):
        """初始化文件缓存后端"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.meta_file = self.cache_dir / '_cache_meta.json'
        
        # 加载元数据
        if self.meta_file.exists():
            with open(self.meta_file, 'r', encoding='utf-8') as f:
                self._file_meta = json.load(f)
        else:
            self._file_meta = {}
        
        logger.info(f"文件缓存目录: {self.cache_dir}")
    
    def _make_key(self, key: str) -> str:
        """生成缓存键 (支持命名空间)"""
        return f"chanlun:{key}"
    
    def _hash_key(self, key: str) -> str:
        """对键进行hash (用于文件名)"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取缓存
        
        Args:
            key: 缓存键
            default: 缓存不存在时的默认值
        
        Returns:
            缓存的值或默认值
        """
        full_key = self._make_key(key)
        
        # 1. 尝试从内存缓存获取
        if full_key in self._memory_cache:
            value, expire_at = self._memory_cache[full_key]
            if expire_at is None or expire_at > time.time():
                # 更新LRU顺序
                self._memory_cache.move_to_end(full_key)
                self.stats.hits += 1
                return value
            else:
                # 已过期
                del self._memory_cache[full_key]
        
        # 2. 从后端获取
        value = None
        if self.backend == 'redis':
            value = self._redis_get(full_key)
        else:
            value = self._file_get(full_key)
        
        if value is not None:
            self.stats.hits += 1
            # 放入内存缓存
            self._memory_set(full_key, value, None)
            return value
        else:
            self.stats.misses += 1
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        设置缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间(秒), None使用默认值, 0表示永不过期
        
        Returns:
            是否设置成功
        """
        full_key = self._make_key(key)
        ttl = ttl if ttl is not None else self.default_ttl
        
        # 1. 设置内存缓存
        expire_at = time.time() + ttl if ttl > 0 else None
        self._memory_set(full_key, value, expire_at)
        
        # 2. 设置后端缓存
        success = False
        if self.backend == 'redis':
            success = self._redis_set(full_key, value, ttl)
        else:
            success = self._file_set(full_key, value, ttl)
        
        if success:
            self.stats.sets += 1
            self.stats.size += 1
        
        return success
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        full_key = self._make_key(key)
        
        # 1. 删除内存缓存
        if full_key in self._memory_cache:
            del self._memory_cache[full_key]
        
        # 2. 删除后端缓存
        success = False
        if self.backend == 'redis':
            success = self._redis_delete(full_key)
        else:
            success = self._file_delete(full_key)
        
        if success:
            self.stats.deletes += 1
            self.stats.size -= 1
        
        return success
    
    def clear(self) -> bool:
        """清空所有缓存"""
        self._memory_cache.clear()
        
        if self.backend == 'redis':
            # 只删除chanlun命名空间的键
            pattern = self._make_key('*')
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
        else:
            # 删除所有缓存文件
            for file in self.cache_dir.glob('*.cache'):
                file.unlink()
            self._file_meta.clear()
            self._save_file_meta()
        
        self.stats = CacheStats()
        logger.info("缓存已清空")
        return True
    
    def get_stats(self) -> CacheStats:
        """获取缓存统计信息"""
        return self.stats
    
    # ========== 内存LRU缓存 ==========
    
    def _memory_set(self, key: str, value: Any, expire_at: Optional[float]):
        """设置内存缓存 (LRU)"""
        self._memory_cache[key] = (value, expire_at)
        self._memory_cache.move_to_end(key)
        
        # LRU淘汰
        while len(self._memory_cache) > self._max_memory_items:
            self._memory_cache.popitem(last=False)
    
    # ========== Redis后端 ==========
    
    def _redis_get(self, key: str) -> Any:
        """从Redis获取"""
        try:
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Redis读取失败: {e}")
        return None
    
    def _redis_set(self, key: str, value: Any, ttl: int) -> bool:
        """设置Redis缓存"""
        try:
            data = pickle.dumps(value)
            if ttl > 0:
                self.redis_client.setex(key, ttl, data)
            else:
                self.redis_client.set(key, data)
            return True
        except Exception as e:
            logger.error(f"Redis写入失败: {e}")
            return False
    
    def _redis_delete(self, key: str) -> bool:
        """删除Redis缓存"""
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis删除失败: {e}")
            return False
    
    # ========== 文件后端 ==========
    
    def _file_get(self, key: str) -> Any:
        """从文件获取"""
        hash_key = self._hash_key(key)
        cache_file = self.cache_dir / f"{hash_key}.cache"
        
        if not cache_file.exists():
            return None
        
        # 检查是否过期
        meta = self._file_meta.get(hash_key, {})
        expire_at = meta.get('expire_at')
        if expire_at and time.time() > expire_at:
            # 已过期,删除
            self._file_delete(key)
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = f.read()
            
            if self.enable_compression:
                import zlib
                data = zlib.decompress(data)
            
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"文件读取失败: {e}")
            return None
    
    def _file_set(self, key: str, value: Any, ttl: int) -> bool:
        """设置文件缓存"""
        hash_key = self._hash_key(key)
        cache_file = self.cache_dir / f"{hash_key}.cache"
        
        try:
            data = pickle.dumps(value)
            
            if self.enable_compression:
                import zlib
                data = zlib.compress(data)
            
            with open(cache_file, 'wb') as f:
                f.write(data)
            
            # 更新元数据
            expire_at = time.time() + ttl if ttl > 0 else None
            self._file_meta[hash_key] = {
                'key': key,
                'created_at': time.time(),
                'expire_at': expire_at,
                'size': len(data)
            }
            self._save_file_meta()
            
            return True
        except Exception as e:
            logger.error(f"文件写入失败: {e}")
            return False
    
    def _file_delete(self, key: str) -> bool:
        """删除文件缓存"""
        hash_key = self._hash_key(key)
        cache_file = self.cache_dir / f"{hash_key}.cache"
        
        try:
            if cache_file.exists():
                cache_file.unlink()
            if hash_key in self._file_meta:
                del self._file_meta[hash_key]
                self._save_file_meta()
            return True
        except Exception as e:
            logger.error(f"文件删除失败: {e}")
            return False
    
    def _save_file_meta(self):
        """保存文件缓存元数据"""
        try:
            with open(self.meta_file, 'w', encoding='utf-8') as f:
                json.dump(self._file_meta, f, indent=2)
        except Exception as e:
            logger.error(f"元数据保存失败: {e}")
    
    def cleanup_expired(self) -> int:
        """清理过期缓存 (仅文件缓存)"""
        if self.backend != 'file':
            return 0
        
        count = 0
        current_time = time.time()
        expired_keys = []
        
        for hash_key, meta in self._file_meta.items():
            expire_at = meta.get('expire_at')
            if expire_at and current_time > expire_at:
                expired_keys.append(meta['key'])
        
        for key in expired_keys:
            if self._file_delete(key):
                count += 1
        
        if count > 0:
            logger.info(f"清理过期缓存: {count}条")
        
        return count


# ========== 工具函数 ==========

def create_cache_from_config(config: Dict) -> ChanLunCache:
    """从配置创建缓存实例"""
    return ChanLunCache(
        backend=config.get('backend', 'file'),
        redis_url=config.get('redis_url'),
        redis_db=config.get('redis_db', 0),
        cache_dir=config.get('cache_dir', './cache/chanlun'),
        max_memory_items=config.get('max_memory_items', 1000),
        default_ttl=config.get('default_ttl', 3600),
        enable_compression=config.get('enable_compression', True)
    )


# ========== 测试代码 ==========

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建文件缓存
    cache = ChanLunCache(backend='file', cache_dir='./test_cache', default_ttl=10)
    
    # 测试数据
    test_data = {
        'bi_list': [{'start': 100, 'end': 110, 'direction': 'up'}],
        'xd_list': [{'start': 100, 'end': 120, 'direction': 'up'}],
        'buy_points': [(105, 'I类买点')],
        'df': pd.DataFrame({'close': np.random.rand(10)})
    }
    
    # 测试基本功能
    print("\n=== 测试缓存基本功能 ===")
    
    # 1. 设置缓存
    cache.set('test_stock_20240101', test_data, ttl=5)
    print("✅ 缓存设置成功")
    
    # 2. 获取缓存 (应该命中)
    result = cache.get('test_stock_20240101')
    print(f"✅ 缓存获取成功: {result is not None}")
    
    # 3. 再次获取 (命中内存缓存)
    result = cache.get('test_stock_20240101')
    print(f"✅ 内存缓存命中: {result is not None}")
    
    # 4. 获取不存在的缓存
    result = cache.get('not_exist', default='DEFAULT')
    print(f"✅ 默认值: {result}")
    
    # 5. 统计信息
    stats = cache.get_stats()
    print(f"\n=== 缓存统计 ===")
    print(f"命中次数: {stats.hits}")
    print(f"未命中次数: {stats.misses}")
    print(f"命中率: {stats.hit_rate:.2%}")
    print(f"设置次数: {stats.sets}")
    print(f"当前条目数: {stats.size}")
    
    # 6. 测试过期
    print("\n=== 测试过期机制 ===")
    print("等待6秒...")
    time.sleep(6)
    result = cache.get('test_stock_20240101')
    print(f"过期后获取: {result}")
    
    # 7. 清空缓存
    cache.clear()
    print("\n✅ 缓存已清空")
    
    print("\n=== 测试完成 ===")

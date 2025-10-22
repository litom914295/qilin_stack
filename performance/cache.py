"""
多级缓存策略 - 内存+Redis
"""
import json
import time
from typing import Any, Optional, Callable
from functools import wraps
import hashlib


class MemoryCache:
    """内存缓存"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache: dict = {}
        self.max_size = max_size
        self.ttl = ttl
        self.access_times: dict = {}
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if key in self.cache:
            data, expire_time = self.cache[key]
            if time.time() < expire_time:
                self.access_times[key] = time.time()
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存"""
        if len(self.cache) >= self.max_size:
            self._evict()
        
        expire_time = time.time() + (ttl or self.ttl)
        self.cache[key] = (value, expire_time)
        self.access_times[key] = time.time()
    
    def _evict(self):
        """LRU淘汰"""
        if not self.access_times:
            return
        oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()


class RedisCache:
    """Redis缓存（模拟实现）"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, ttl: int = 3600):
        self.host = host
        self.port = port
        self.ttl = ttl
        self.client = None
        self._connect()
    
    def _connect(self):
        """连接Redis"""
        try:
            import redis
            self.client = redis.Redis(host=self.host, port=self.port, decode_responses=True)
        except ImportError:
            print("⚠️ Redis未安装，使用内存缓存")
            self.client = None
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if self.client is None:
            return None
        
        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            print(f"Redis get error: {e}")
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存"""
        if self.client is None:
            return
        
        try:
            self.client.setex(key, ttl or self.ttl, json.dumps(value))
        except Exception as e:
            print(f"Redis set error: {e}")
    
    def delete(self, key: str):
        """删除缓存"""
        if self.client:
            self.client.delete(key)


class MultiLevelCache:
    """多级缓存"""
    
    def __init__(self, use_redis: bool = True):
        self.l1_cache = MemoryCache(max_size=1000, ttl=300)  # 5分钟
        self.l2_cache = RedisCache(ttl=3600) if use_redis else None  # 1小时
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存（L1 -> L2）"""
        # 先查L1
        value = self.l1_cache.get(key)
        if value is not None:
            return value
        
        # 再查L2
        if self.l2_cache:
            value = self.l2_cache.get(key)
            if value is not None:
                # 回填L1
                self.l1_cache.set(key, value)
                return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存（L1 + L2）"""
        self.l1_cache.set(key, value, ttl)
        if self.l2_cache:
            self.l2_cache.set(key, value, ttl)
    
    def invalidate(self, key: str):
        """失效缓存"""
        self.l1_cache.cache.pop(key, None)
        if self.l2_cache:
            self.l2_cache.delete(key)
    
    def clear_all(self):
        """清空所有缓存"""
        self.l1_cache.clear()


# 全局缓存实例
_cache = None

def get_cache() -> MultiLevelCache:
    """获取全局缓存实例"""
    global _cache
    if _cache is None:
        _cache = MultiLevelCache(use_redis=False)  # 默认只用内存
    return _cache


def cached(ttl: int = 3600, key_prefix: str = ""):
    """缓存装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成缓存key
            cache_key = _generate_cache_key(key_prefix or func.__name__, args, kwargs)
            
            # 尝试从缓存获取
            cache = get_cache()
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # 调用原函数
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # 存入缓存
            cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator


def _generate_cache_key(prefix: str, args: tuple, kwargs: dict) -> str:
    """生成缓存key"""
    key_parts = [prefix]
    key_parts.extend(str(arg) for arg in args)
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_str = ":".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


# 使用示例
@cached(ttl=600, key_prefix="market_data")
async def get_market_data(symbol: str, date: str):
    """获取市场数据（带缓存）"""
    # 实际数据获取逻辑
    await asyncio.sleep(0.1)  # 模拟IO
    return {"symbol": symbol, "date": date, "price": 100.0}
